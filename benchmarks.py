import torch
from torch.profiler import profile, record_function, ProfilerActivity
import gc
import psutil
import os
import sys
from contextlib import contextmanager

torch.manual_seed(2)

from functions import *

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

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == 'cuda' else 0

        # Suppress verbose profiler logs
        #with _suppress_stdout_stderr():
        start_event.record()
        output = func(*args)
        end_event.record()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        #print(torch.cuda.memory_summary(abbreviated=True))
        # Sum up all CUDA time from all events in the profile
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        #total_cuda_time_ms = sum([e.cpu_time for e in prof.key_averages()]) / 1000.0
        total_cuda_time_ms = start_event.elapsed_time(end_event)
        end_peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2 if self.device.type == 'cuda' else 0
        
        used_mem_mb = end_peak_mem_mb - start_peak_mem_mb

        return output, total_cuda_time_ms, used_mem_mb

    def _check_correctness(self, tensor_cuda, tensor_python, label, atol, rtol):
        """Checks if two tensors are close and prints detailed differences if they are not."""        
        are_close = torch.allclose(tensor_cuda, tensor_python.to(self.device), atol=atol, rtol=rtol)
        n_different_sign = (tensor_cuda.sign() != tensor_python.to(self.device).sign()).sum().item()
        n_different_inf = ((torch.isinf(tensor_cuda) != torch.isinf(tensor_python.to(self.device)))).sum().item()
        if not are_close or n_different_sign > 0 or n_different_inf > 0:
            print(f"\n--- Correctness FAILED for '{label}' ---")
            abs_diff = (tensor_cuda - tensor_python.to(self.device)).abs()
            rel_diff = abs_diff / (tensor_python.to(self.device).abs() + 1e-8)
            print(f"  Number of infinite elements in CUDA tensor: {torch.isinf(tensor_cuda).sum().item()}")
            print(f"  Number of infinite elements in Python tensor: {torch.isinf(tensor_python).sum().item()}")
            print(f"  Number of positive elements in CUDA tensor: {(tensor_cuda > 0).sum().item()}")
            print(f"  Number of positive elements in Python tensor: {(tensor_python > 0).sum().item()}")
            print(f"  Max Absolute Difference: {abs_diff[torch.isfinite(abs_diff)].max().item():.6e}")
            print(f"  Mean Absolute Difference: {abs_diff[torch.isfinite(abs_diff)].float().mean().item():.6e}")
            print(f"  Max Relative Difference: {rel_diff[torch.isfinite(rel_diff)].max().item():.6e}")
            print(f"  Mean Relative Difference: {rel_diff[torch.isfinite(rel_diff)].float().mean().item():.6e}")
        return are_close

    def run(self, fwd_atol=1e-6, fwd_rtol=1e-6, bwd_atol=1e-6, bwd_rtol=1e-6, **kwargs):
        """
        Executes the full testing pipeline: performance, memory, and correctness.
        Supports forward functions that return a single tensor or a tuple/list of tensors.
        All outputs with gradients are reduced (sum) and then summed together for the loss.
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

        # --- Helpers for multi-output handling ---
        def _loss_from_outputs(out):
            """Return a scalar loss built by summing .sum() of all grad-requiring outputs.
            Returns None if no grad-requiring tensors are present."""
            if torch.is_tensor(out):
                return out.sum() if (out.requires_grad and out.is_floating_point()) else None
            if isinstance(out, (tuple, list)):
                terms = []
                for o in out:
                    if torch.is_tensor(o) and o.is_floating_point() and o.requires_grad:
                        terms.append(o.sum())
                return sum(terms) if terms else None
            return None

        def _has_grad_outputs(out):
            """True if any output tensor requires grad."""
            if torch.is_tensor(out):
                return bool(out.is_floating_point() and out.requires_grad)
            if isinstance(out, (tuple, list)):
                return any(torch.is_tensor(o) and o.is_floating_point() and o.requires_grad for o in out)
            return False

        def _check_forward_correctness(out_cu, out_py):
            """Element-wise correctness for tuple/list outputs, fallback to single tensor check."""
            # Single-tensor or non-sequence case: rely on existing checker
            if not isinstance(out_py, (tuple, list)) and not isinstance(out_cu, (tuple, list)):
                return self._check_correctness(out_cu, out_py, "Forward Pass Output", fwd_atol, fwd_rtol)

            # If one is sequence and the other isn't -> fail
            if isinstance(out_py, (tuple, list)) != isinstance(out_cu, (tuple, list)):
                print("Forward Pass Output: type mismatch between CUDA and PyTorch outputs (tuple/list vs non-sequence).")
                return False

            # Both are sequences: lengths must match
            if len(out_py) != len(out_cu):
                print(f"Forward Pass Output: length mismatch (PyTorch={len(out_py)}, CUDA={len(out_cu)}).")
                return False

            ok = True
            for i, (cu_i, py_i) in enumerate(zip(out_cu, out_py)):
                if torch.is_tensor(cu_i) and torch.is_tensor(py_i):
                    ok &= self._check_correctness(cu_i, py_i, f"Forward Pass Output[{i}]", fwd_atol, fwd_rtol)
                else:
                    # For non-tensor elements, do a simple equality check when possible.
                    try:
                        same = (cu_i == py_i)
                        # If it's a numpy/bool scalar this may be array-like; coerce to bool when possible.
                        same = bool(same) if hasattr(same, "__bool__") else bool(same)
                    except Exception:
                        same = False
                    if not same:
                        print(f"Forward Pass Output[{i}]: non-tensor mismatch.")
                    ok &= same
            return ok

        # Create base input tensors once
        base_inputs = {
            name: creator(self.device, self.dtype) if callable(creator) else creator
            for name, creator in self.input_creators.items()
        }
        
        # --- Python Implementation ---
        inputs_py = {
            name: (t.clone().requires_grad_(True) if torch.is_tensor(t) and torch.is_floating_point(t) else t) 
            for name, t in base_inputs.items()
        }
        all_args_py = {**inputs_py, **kwargs}
        ordered_args_py = [all_args_py[name] for name in self.arg_order]
        
        output_py, fwd_time_py, fwd_mem_py = self._measure_pass(self.python_function, *ordered_args_py)
        results['pytorch']['fwd_time_ms'] = fwd_time_py
        results['pytorch']['fwd_mem_mb'] = fwd_mem_py
        
        # Determine backward from PyTorch outputs
        run_backward = _has_grad_outputs(output_py)

        if run_backward:
            loss_py = _loss_from_outputs(output_py)
            # If for some reason nothing required grad, skip bwd
            if loss_py is not None:
                _, bwd_time_py, bwd_mem_py = self._measure_pass(lambda: loss_py.backward())
                results['pytorch']['bwd_time_ms'] = bwd_time_py
                results['pytorch']['bwd_mem_mb'] = max(fwd_mem_py, bwd_mem_py) # Peak memory is the max of fwd and bwd
            else:
                run_backward = False
                results['pytorch']['bwd_mem_mb'] = fwd_mem_py
        else:
            results['pytorch']['bwd_mem_mb'] = fwd_mem_py

        # --- CUDA Implementation ---
        inputs_cu = {
            name: (t.clone().requires_grad_(True) if torch.is_tensor(t) and torch.is_floating_point(t) else t) 
            for name, t in base_inputs.items()
        }
        all_args_cu = {**inputs_cu, **kwargs}
        ordered_args_cu = [all_args_cu[name] for name in self.arg_order]
        
        output_cu, fwd_time_cu, fwd_mem_cu = self._measure_pass(self.cuda_function, *ordered_args_cu)
        results['cuda']['fwd_time_ms'] = fwd_time_cu
        results['cuda']['fwd_mem_mb'] = fwd_mem_cu
        
        if run_backward:
            loss_cu = _loss_from_outputs(output_cu)
            if loss_cu is not None:
                _, bwd_time_cu, bwd_mem_cu = self._measure_pass(lambda: loss_cu.backward())
                results['cuda']['bwd_time_ms'] = bwd_time_cu
                results['cuda']['bwd_mem_mb'] = max(fwd_mem_cu, bwd_mem_cu)
            else:
                # PyTorch said we should run backward, but CUDA produced no grad-requiring outputs
                # We'll mark backward correctness as False later when grads are compared.
                results['cuda']['bwd_mem_mb'] = fwd_mem_cu
        else:
            results['cuda']['bwd_mem_mb'] = fwd_mem_cu
        
        # --- Correctness Checks ---
        results['correctness']['fwd'] = _check_forward_correctness(output_cu, output_py)
        
        if run_backward:
            for name in base_inputs:
                if not torch.is_tensor(inputs_cu[name]) or not torch.is_tensor(inputs_py[name]):
                    continue
                if inputs_cu[name].grad is None or inputs_py[name].grad is None:
                    # If one side has missing grad while the other has it, that's a failure for that grad.
                    if (inputs_cu[name].grad is None) != (inputs_py[name].grad is None):
                        print(f"Gradient of '{name}': mismatch in gradient presence (one is None).")
                        results['correctness']['bwd'] &= False
                    continue
                is_grad_correct = self._check_correctness(
                    inputs_cu[name].grad, inputs_py[name].grad, f"Gradient of '{name}'", bwd_atol, bwd_rtol
                )
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
        
        # Backward Pass Data (only if backward was run)
        if run_backward:
            print(f"| {'Backward':<{pass_w}} | {'Pytorch':<{impl_w}} | {results['pytorch']['bwd_time_ms']:>{time_w}.4f} | {results['pytorch']['bwd_mem_mb']:>{mem_w}.2f} | {'':>{corr_w}} |")
            print(f"| {'Backward':<{pass_w}} | {'CUDA':<{impl_w}} | {results['cuda']['bwd_time_ms']:>{time_w}.4f} | {results['cuda']['bwd_mem_mb']:>{mem_w}.2f} | {bwd_correct:>{corr_w}} |")
            print(separator)
        else:
            print(f"| {'Backward':<{pass_w}} | {'N/A (no grad)':<{impl_w}} | {'':>{time_w}} | {'':>{mem_w}} | {'SKIP':>{corr_w}} |")
            print(separator)


def create_gaussian_blob_targets(B, Q, C, Ht, Wt, device):
    """
    Creates a target tensor with random Gaussian blobs representing different classes.
    """
    # Start with a background of class 0
    torch.manual_seed(0)
    mask_targets = torch.zeros(B, Ht, Wt, dtype=torch.long, device=device)
    
    # Generate blobs for each image in the batch
    for b in range(B):
        # Add a random number of blobs to each image
        num_blobs = torch.randint(10, 20, (1,)).item()
        
        for _ in range(num_blobs):
            # Assign a random object ID (1 to Q-1, since 0 is background)
            query_id = torch.randint(1, Q, (1,)).item()
            
            # Define random center and size for the Gaussian blob
            center_x = torch.randint(0, Wt, (1,)).item()
            center_y = torch.randint(0, Ht, (1,)).item()
            sigma_x = torch.randint(int(Wt/25), int(Wt/8), (1,)).item()
            sigma_y = torch.randint(int(Ht/25), int(Ht/8), (1,)).item()

            # Create coordinate grid
            x = torch.arange(0, Wt, device=device).view(1, Wt)
            y = torch.arange(0, Ht, device=device).view(Ht, 1)

            # Calculate the Gaussian distribution
            dist_x = (x - center_x) ** 2 / (2 * sigma_x ** 2)
            dist_y = (y - center_y) ** 2 / (2 * sigma_y ** 2)
            gaussian_blob = torch.exp(-(dist_y + dist_x))

            # Create a mask where the blob is most prominent
            mask = gaussian_blob > 0.5
            
            # "Paint" the object ID onto the target tensor where the mask is true
            mask_targets[b, mask] = query_id
    
    # Per-query class assignment; -1 means "no object for this query in this image"
    max_num_blobs = mask_targets.max().item()
    cls_targets = torch.full((B, max_num_blobs+1), -1, dtype=torch.long, device=device)

    for b in range(B):
        present_qids = mask_targets[b].unique()
        present_qids = present_qids[present_qids > 0]  # exclude background
        if present_qids.numel() > 0:
            # Random classes in [0, C-1] for the queries that actually appear
            rand_classes = torch.randint(0, C, (present_qids.numel(),), device=device)
            cls_targets[b, present_qids] = rand_classes

    return mask_targets, cls_targets

def create_label_targets(B, C, device):
    """
    Creates a target tensor with random Gaussian blobs representing different classes.
    """
    # Start with a background of class 0
    max_num_blobs = 30
    targets = torch.zeros(B, max_num_blobs, dtype=torch.long, device=device)
    
    # Generate blobs for each image in the batch
    for b in range(B):
        # Add a random number of blobs to each image
        num_blobs = torch.randint(10, max_num_blobs-1, (1,)).item()
        
        for i in range(num_blobs):
            # Assign a random class ID (1 to C-1, since 0 is background)
            class_id = torch.randint(1, C, (1,)).item()            
            # "Paint" the class ID onto the target tensor where the mask is true
            targets[b, i] = class_id
            
    return targets

def test_sigmoid_ce_loss():
    """Test the CUDA implementation of sigmoid cross-entropy loss"""
    L, B, C, H, W = 1, 1, 128, 256, 256
    H_t, W_t = 1024, 1024
    
    input_creators = {
        "logits": lambda device, dtype: torch.randn(L, B, C, H, W, device=device, dtype=dtype),
        "targets": lambda device, dtype: create_gaussian_blob_targets(B, C, H_t, W_t, device),
        "num_masks": 5,
        "scale": 1.0,
    }
    
    arg_order = ["logits", "targets", "num_masks", "scale"]
    
    tester = CUDAKernelTester(
        cuda_function=SigmoidCELossFunction.apply,
        python_function=sigmoid_cross_entropy_loss_py,
        input_creators=input_creators,
        arg_order=arg_order
    )
    
    tester.run()

def test_pw_sigmoid_ce_loss():
    """Test the CUDA implementation of sigmoid cross-entropy loss"""
    L, B, C, H, W = 10, 16, 128, 256, 256
    H_t, W_t = 1024, 1024

    input_creators = {
        "logits": lambda device, dtype: torch.randn(L, B, C, H, W, device=device, dtype=dtype),
        "targets": lambda device, dtype: create_gaussian_blob_targets(B, C, H_t, W_t, device),
        "background_index": 0,
        "scale": 1.0,
        "focal_gamma": 0.0,
        "focal_alpha": None,
    }

    arg_order = ["logits", "targets", "background_index", "scale", "focal_gamma", "focal_alpha"]
    
    tester = CUDAKernelTester(
        cuda_function=PairwiseSigmoidCELossFunction.apply,
        python_function=pairwise_sigmoid_cross_entropy_loss_py,
        input_creators=input_creators,
        arg_order=arg_order
    )
    
    tester.run()

def test_dice_loss():
    """Test the CUDA implementation of sigmoid dice loss"""
    L, B, C, H, W = 1, 1, 133, 256, 256
    H_t, W_t = 1024, 1024
    
    input_creators = {
        "logits": lambda device, dtype: torch.randn(L, B, C, H, W, device=device, dtype=dtype),
        #"targets": lambda device, dtype: torch.randint(0, C, (B, H_t, W_t), device=device, dtype=torch.long),
        "targets": lambda device, dtype: create_gaussian_blob_targets(B, C, H_t, W_t, device),
        "smooth": 1.0,
        "num_masks": 1
    }
    
    arg_order = ["logits", "targets", "smooth", "num_masks"]
    
    tester = CUDAKernelTester(
        cuda_function=DiceLossFunction.apply,
        python_function=dice_loss_py,
        input_creators=input_creators,
        arg_order=arg_order
    )
    
    tester.run()

def test_pw_dice_loss():
    """Test the CUDA implementation of sigmoid cross-entropy loss"""
    L, B, C, H, W = 1, 2, 128, 256, 256
    H_t, W_t = 1024, 1024

    input_creators = {
        "logits": lambda device, dtype: torch.randn(L, B, C, H, W, device=device, dtype=dtype),
        "targets": lambda device, dtype: create_gaussian_blob_targets(B, C, H_t, W_t, device),
        "smooth": 1.0,
        "background_index": 1,
        "scale": 1.0
    }
    
    arg_order = ["logits", "targets", "smooth", "background_index", "scale"]
    
    tester = CUDAKernelTester(
        cuda_function=PairwiseDiceLossFunction.apply,
        python_function=pairwise_dice_loss_py,
        input_creators=input_creators,
        arg_order=arg_order
    )
    
    tester.run()

def test_pw_label_loss():
    """Test the CUDA implementation of sigmoid cross-entropy loss"""
    L, B, Q, C = 10, 16, 300, 128

    input_creators = {
        "logits": lambda device, dtype: torch.randn(L, B, Q, C, device=device, dtype=dtype),
        "targets": lambda device, dtype: create_label_targets(B, C, device),
        "background_index": 0,
        "scale": 1.0
    }
    
    arg_order = ["logits", "targets", "background_index", "scale"]
    
    tester = CUDAKernelTester(
        cuda_function=PairwiseSigmoidCELossFunction.apply,
        python_function=pairwise_label_loss_py,
        input_creators=input_creators,
        arg_order=arg_order
    )
    
    tester.run() 

def test_pw_mask_loss():
    """Test the CUDA implementation of pairwise mask loss"""
    L, B, Q, C, H, W = 1, 2, 128, 128, 256, 256
    H_t, W_t = 1024, 1024

    input_creators = {
        "mask_logits": lambda device, dtype: torch.randn(L, B, Q, H, W, device=device, dtype=dtype),
        "mask_targets": lambda device, dtype: create_gaussian_blob_targets(B, Q, C, H_t, W_t, device)[0],
        "cls_logits": lambda device, dtype: torch.randn(L, B, Q, C, device=device, dtype=dtype),
        "cls_targets": lambda device, dtype: create_gaussian_blob_targets(B, Q, C, H_t, W_t, device)[1],
        "smooth": 1.0,
        "sigmoid_scale": 1.0,
        "dice_scale": 1.0,
        "cls_scale": 1.0,
        "background_index": 0
    }
    
    arg_order = [
        "mask_logits", 
        "mask_targets", 
        "cls_logits",
        "cls_targets",
        "smooth", 
        "sigmoid_scale", 
        "dice_scale", 
        "cls_scale",
        "background_index"
    ]
    
    tester = CUDAKernelTester(
        cuda_function=PairwiseMaskLossFunction.apply,
        python_function=pairwise_mask_loss_py,
        input_creators=input_creators,
        arg_order=arg_order
    )
    
    tester.run()

def test_mask_matching():
    """Test the CUDA implementation of mask matching"""
    L, B, Q, C, H, W = 10, 8, 128, 128, 256, 256
    H_t, W_t = 1024, 1024

    input_creators = {
        "mask_logits": lambda device, dtype: torch.randn(L, B, Q, H, W, device=device, dtype=dtype),
        "mask_targets": lambda device, dtype: create_gaussian_blob_targets(B, Q, C, H_t, W_t, device)[0],
        "cls_logits": lambda device, dtype: torch.randn(L, B, Q, C, device=device, dtype=dtype),
        "cls_targets": lambda device, dtype: create_gaussian_blob_targets(B, Q, C-1, H_t, W_t, device)[1],
        "smooth": 1.0,
        "sigmoid_scale": 1.0,
        "dice_scale": 1.0,
        "cls_scale": 1.0,
        "background_index": 0,
        "inf_thresh": 1e30,
        "num_masks": 5,
        "force_unmatched_class_to_background": False,
        "force_unmatched_masks_to_empty": False,
        "K": 1,
        "assignment_strategy": "global",
        "focal_gamma": 2.0,
        "focal_alpha": 0.25,
        "void_class_index": C-1,
    }

    arg_order = [
        "mask_logits",
        "mask_targets",
        "cls_logits",
        "cls_targets",
        "smooth", 
        "sigmoid_scale", 
        "dice_scale", 
        "cls_scale",
        "background_index",
        "inf_thresh",
        "num_masks",
        "force_unmatched_class_to_background",
        "force_unmatched_masks_to_empty",
        "K",
        "assignment_strategy",
        "focal_gamma",
        "focal_alpha",
        "void_class_index",
    ]
    
    tester = CUDAKernelTester(
        cuda_function=MaskMatchingFunction.apply,
        python_function=mask_matching_py,
        input_creators=input_creators,
        arg_order=arg_order
    )
    
    tester.run()

if __name__ == "__main__":
    test_mask_matching()
    #test_pw_sigmoid_ce_loss()
