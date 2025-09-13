# Linear Optimized

## Overview
This project is an optimized CUDA implementation of LU decomposition with residual norm computation.  
It builds on the baseline `linearSolver.cu` by reducing kernel overhead and improving numerical stability.

## Key Improvements
- **cuSOLVER Integration**: Uses NVIDIA’s cuSOLVER library for LU decomposition, which provides built-in partial pivoting for better numerical stability.  
- **Memory Management**: Allocates and reuses GPU buffers outside of loops to avoid repeated device memory operations.  
- **Batched Residual Norms**: Residual computations are performed in parallel on the GPU, reducing host-device transfers and CPU serial work.  

## Build
Requires the NVIDIA CUDA Toolkit. Example:
```bash
nvcc -O3 -arch=sm_86 src/linearOptimized.cu -o linearOptimized -lcusolver -lcublas
