# BP_Tests
Implementations of segmentation algorithms that I was testing against in my Thesis

## Cuda Quickshift implementation
Paper: Really Quick Shift

Brian Fulkerson
        
Taken from https://github.com/amueller/GPU-Quickshift-Python-Bindings

## Cuda CCL implementation
Paper: Parallel graph component labelling with GPUs and CUDA

K. Hawick, A. Leist and D. Playne
        
Taken from https://github.com/XiangyuBi/CUCCL

Input map was changed to int to provide normalization, other algorithms use int/float as base so 4bytes values

## Cuda Wathershed implementation
Paper: Fast image segmentation by watershed transform on graphical hardware

Vitor B, Körbes A.
        
Taken from https://github.com/louismullie/watershed-cuda

Only kernel were used, python was replaced by C++ in test implementation

