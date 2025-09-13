# Linear Solver

## Overview
This project implements a baseline CUDA LU decomposition with residual calculation.

## How to Build
nvcc -O3 -arch=sm_86 src/linearSolver.cu -o linearSolver

## How to Run
./linearSolver
./linearSolver 512   # run with matrix size 512

## What I Learned
- Basics of CUDA kernel launches
- Residual computation and numerical accuracy
