sudo -E "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" \
    /usr/local/cuda/bin/ncu \
    -o report_backwards \
    --target-processes all \
    --set full \
    -k box_pair_gaussian_backward_grad1_kernel \
    -f python3 benchmarks.py

