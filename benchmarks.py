import torch
from torch.profiler import profile, record_function, ProfilerActivity
import gc
import psutil
import os
import sys
from contextlib import contextmanager

from functions import (
    SigmoidCELossFunction, sigmoid_cross_entropy_loss
)

@contextmanager
def _suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


class CUDAKernelTester:
    """
    A class to generalize the testing of CUDA kernels against a Python implementation.
    It benchmarks performance and memory, checks for correctness, and presents
    the results in a comparison table.
    """

    def __init__(self, cuda_function, python_function, input_creators, arg_order, device="cuda", dtype=torch.float32):
        """
        Initializes the tester.

        Args:
            cuda_function: The CUDA kernel function to be tested (e.g., MyFunc.apply).
            python_function: The reference Python implementation of the kernel.
            input_creators: A dictionary of lambda functions to generate input tensors.
            arg_order (list[str]): A list of argument names in the exact order the function expects them.
            device (str, optional): The device to run the tests on. Defaults to "cuda".
            dtype (torch.dtype, optional): The data type for the tensors. Defaults to torch.float32.
        """
        self.cuda_function = cuda_function
        self.python_function = python_function
        self.input_creators = input_creators
        self.arg_order = arg_order
        self.device = torch.device(device)
        self.dtype = dtype

    def _measure_pass(self, func, *args):
        """
        Profiles a function to get total CUDA time and peak memory usage.
        
        This function sums the CUDA time of ALL kernels launched during the
        profiler's context, providing an accurate measurement for asynchronous operations.
        """
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        # Suppress verbose profiler logs
        with _suppress_stdout_stderr():
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                output = func(*args)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
        
        # Sum up all CUDA time from all events in the profile
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        total_cuda_time_ms = sum([e.cuda_time for e in prof.key_averages()]) / 1000.0
        peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == 'cuda' else 0
        
        return output, total_cuda_time_ms, peak_mem_mb

    def _check_correctness(self, tensor_cuda, tensor_python, label, atol, rtol):
        """Checks if two tensors are close and prints detailed differences if they are not."""
        are_close = torch.allclose(tensor_cuda, tensor_python.to(self.device), atol=atol, rtol=rtol)
        if not are_close:
            print(f"\n--- Correctness FAILED for '{label}' ---")
            abs_diff = (tensor_cuda - tensor_python.to(self.device)).abs()
            rel_diff = abs_diff / (tensor_python.to(self.device).abs() + 1e-8)
            print(f"  Max Absolute Difference: {abs_diff.max().item():.6e}")
            print(f"  Mean Absolute Difference: {abs_diff.mean().item():.6e}")
            print(f"  Max Relative Difference: {rel_diff.max().item():.6e}")
            print("-" * 40)
        return are_close

    def run(self, fwd_atol=1e-5, fwd_rtol=1e-4, bwd_atol=1e-4, bwd_rtol=1e-3, **kwargs):
        """
        Executes the full testing pipeline: performance, memory, and correctness.
        """
        function_name = self.cuda_function.__self__.__name__
        
        print(f"\n{'='*80}")
        print(f"Starting test for {function_name}")
        print(f"{'='*80}")

        # Initialize results dictionary
        results = {
            'pytorch': {'fwd_time_ms': 0.0, 'bwd_time_ms': 0.0, 'fwd_mem_mb': 0.0, 'bwd_mem_mb': 0.0},
            'cuda':    {'fwd_time_ms': 0.0, 'bwd_time_ms': 0.0, 'fwd_mem_mb': 0.0, 'bwd_mem_mb': 0.0},
            'correctness': {'fwd': False, 'bwd': True}
        }

        # Create base input tensors once
        base_inputs = {name: creator(self.device, self.dtype) for name, creator in self.input_creators.items()}
        
        # --- Python Implementation ---
        inputs_py = {
            name: (t.clone().requires_grad_(True) if torch.is_floating_point(t) else t) 
            for name, t in base_inputs.items()
        }
        all_args_py = {**inputs_py, **kwargs}
        ordered_args_py = [all_args_py[name] for name in self.arg_order]
        
        output_py, fwd_time_py, fwd_mem_py = self._measure_pass(self.python_function, *ordered_args_py)
        results['pytorch']['fwd_time_ms'] = fwd_time_py
        results['pytorch']['fwd_mem_mb'] = fwd_mem_py
        loss_py = output_py.sum()

        _, bwd_time_py, bwd_mem_py = self._measure_pass(lambda: loss_py.backward())
        results['pytorch']['bwd_time_ms'] = bwd_time_py
        results['pytorch']['bwd_mem_mb'] = max(fwd_mem_py, bwd_mem_py) # Peak memory is the max of fwd and bwd

        # --- CUDA Implementation ---
        inputs_cu = {
            name: (t.clone().requires_grad_(True) if torch.is_floating_point(t) else t) 
            for name, t in base_inputs.items()
        }
        all_args_cu = {**inputs_cu, **kwargs}
        ordered_args_cu = [all_args_cu[name] for name in self.arg_order]
        
        output_cu, fwd_time_cu, fwd_mem_cu = self._measure_pass(self.cuda_function, *ordered_args_cu)
        results['cuda']['fwd_time_ms'] = fwd_time_cu
        results['cuda']['fwd_mem_mb'] = fwd_mem_cu
        loss_cu = output_cu.sum()
        
        _, bwd_time_cu, bwd_mem_cu = self._measure_pass(lambda: loss_cu.backward())
        results['cuda']['bwd_time_ms'] = bwd_time_cu
        results['cuda']['bwd_mem_mb'] = max(fwd_mem_cu, bwd_mem_cu)
        
        # --- Correctness Checks ---
        results['correctness']['fwd'] = self._check_correctness(output_cu, output_py, "Forward Pass Output", fwd_atol, fwd_rtol)
        
        for name in base_inputs:
            if inputs_cu[name].grad is not None and inputs_py[name].grad is not None:
                is_grad_correct = self._check_correctness(inputs_cu[name].grad, inputs_py[name].grad, f"Gradient of '{name}'", bwd_atol, bwd_rtol)
                results['correctness']['bwd'] &= is_grad_correct

        # Define column widths
        pass_w, impl_w, time_w, mem_w, corr_w = 10, 14, 15, 20, 10
        
        # Create header and separator strings based on widths
        header = f"| {'Pass':<{pass_w}} | {'Implementation':<{impl_w}} | {'Time (ms)':>{time_w}} | {'Peak GPU Mem (MB)':>{mem_w}} | {'Correct?':>{corr_w}} |"
        separator = f"|{'-'*(pass_w+2)}|{'-'*(impl_w+2)}|{'-'*(time_w+2)}|{'-'*(mem_w+2)}|{'-'*(corr_w+2)}|"
        
        print(separator)
        print(header)
        print(separator)
        
        fwd_correct = 'PASS' if results['correctness']['fwd'] else 'FAIL'
        bwd_correct = 'PASS' if results['correctness']['bwd'] else 'FAIL'

        # Forward Pass Data
        print(f"| {'Forward':<{pass_w}} | {'Pytorch':<{impl_w}} | {results['pytorch']['fwd_time_ms']:>{time_w}.4f} | {results['pytorch']['fwd_mem_mb']:>{mem_w}.2f} | {'':>{corr_w}} |")
        print(f"| {'Forward':<{pass_w}} | {'CUDA':<{impl_w}} | {results['cuda']['fwd_time_ms']:>{time_w}.4f} | {results['cuda']['fwd_mem_mb']:>{mem_w}.2f} | {fwd_correct:>{corr_w}} |")
        print(separator)
        # Backward Pass Data
        print(f"| {'Backward':<{pass_w}} | {'Pytorch':<{impl_w}} | {results['pytorch']['bwd_time_ms']:>{time_w}.4f} | {results['pytorch']['bwd_mem_mb']:>{mem_w}.2f} | {'':>{corr_w}} |")
        print(f"| {'Backward':<{pass_w}} | {'CUDA':<{impl_w}} | {results['cuda']['bwd_time_ms']:>{time_w}.4f} | {results['cuda']['bwd_mem_mb']:>{mem_w}.2f} | {bwd_correct:>{corr_w}} |")
        print(separator)


def test_sigmoid_ce_loss():
    """Test the CUDA implementation of sigmoid cross-entropy loss"""
    B, C, H, W = 16, 256, 64, 64
    H_t, W_t = 512, 512
    
    input_creators = {
        "logits": lambda device, dtype: torch.randn(B, C, H, W, device=device, dtype=dtype),
        "targets": lambda device, dtype: torch.randint(0, C, (B, H_t, W_t), device=device, dtype=torch.long),
    }
    
    arg_order = ["logits", "targets"]
    
    tester = CUDAKernelTester(
        cuda_function=SigmoidCELossFunction.apply,
        python_function=sigmoid_cross_entropy_loss,
        input_creators=input_creators,
        arg_order=arg_order
    )
    
    tester.run()

if __name__ == "__main__":
    test_sigmoid_ce_loss()
