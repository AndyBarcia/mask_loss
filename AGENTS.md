This is an efficient implementation of common Mask2Former losses in CUDA, including:
- Dice loss.
- Binary cross entropy loss for masks.
- Binary cross entropy loss for classification.
- Hungarian matching.

The structure is as follows:
- `/functions`: Python reference implementations of the functions and entry-points for the CUDA implementations.
    - `dice`: dice loss and pairwise dice loss.
    - `label`: pairwise bce loss for classification.
    - `sigmoid`: bce loss and pairwise bce loss for masks.
    - `mask`: all-in-one pairwise dice+label+sigmoid loss, along with hunarian matching.
- `/src`: the C++ and CUDA implementations.
    - `bindings.cpp`: the header definitions of all the functions exposed to Python.
    - `cpu`: C++ implementations.
    - `cuda`: CUDA implementations.

All tensors or vectors in code should be annotated in a comment in both Python and CUDA code, like this

```
my_logits = some_tensor.some_operation() // (L,B,Q,H,W), float
```

Running tests on this repo is not possible in your test environment because you need CUDA and other heavy dependencies, so don't bother. Just try to write correct code to the best of your ability; your work will be tested later.