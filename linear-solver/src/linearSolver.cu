#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>

// Kernel: divide step of LU decomposition
// For column k, divides each element A[row,k] by the pivot on the diagonal
__global__ void kernel_div(float* A, int n, int k, float pivot) {
    int row = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) A[row * n + k] /= pivot;
}

// Kernel: update step of LU decomposition
// For each (row,col) in the submatrix, subtract L[row,k] * U[k,col]
__global__ void kernel_update(float* A, int n, int k) {
    int row = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    int col = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float l = A[row * n + k];  // L element
        float u = A[k * n + col];  // U element
        A[row * n + col] -= l * u;
    }
}

// Performs in-place LU decomposition on device matrix d_A of size n×n
void lu_decompose(float* d_A, int n) {
    for (int k = 0; k < n; ++k) {
        // Copy pivot from device to host
        float pivot;
        cudaMemcpy(&pivot, d_A + k * n + k, sizeof(float), cudaMemcpyDeviceToHost);
        
        // Launch divide kernel
        int t = 256;
        int b = (n - k - 1 + t - 1) / t;
        kernel_div<<<b, t>>>(d_A, n, k, pivot);

        // Launch update kernel with 2D grid
        dim3 t2(16, 16);
        dim3 b2((n - k - 1 + t2.x - 1) / t2.x,
                (n - k - 1 + t2.y - 1) / t2.y);
        kernel_update<<<b2, t2>>>(d_A, n, k);
    }
}

// Host-side triangular solve using the LU-packed matrix in A
// Solves Ly = b (forward substitution), then Ux = y (back substitution)
void host_tri_solve(float* A, float* b, float* x, int n) {
    // Forward substitution: solve L y = b, writing y into b
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            b[i] -= A[i * n + j] * b[j];
        }
    }
    // Back substitution: solve U x = y, writing x into b
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) {
            b[i] -= A[i * n + j] * b[j];
        }
        b[i] /= A[i * n + i];
    }
    // Copy solution into x
    for (int i = 0; i < n; ++i) x[i] = b[i];
}

int main(int argc, char** argv) {
    // Parse matrix size from arguments or default to 1024
    int n = argc > 1 ? atoi(argv[1]) : 1024;
    size_t sz = n * n * sizeof(float);

    // Allocate host arrays: A (matrix), b (RHS), x (solution)
    float* h_A = (float*)malloc(sz);
    float* h_b = (float*)malloc(n * sizeof(float));
    float* h_x = (float*)malloc(n * sizeof(float));
    if (!h_A || !h_b || !h_x) {
        fprintf(stderr, "Host malloc failed\n");
        return -1;
    }

    // Initialize A with random values + diagonal dominance, b with random values
    srand(0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float v = (float)rand() / RAND_MAX;
            h_A[i * n + j] = v + (i == j ? n : 0);
        }
        h_b[i] = (float)rand() / RAND_MAX;
    }
    
    // Backup original A and b for residual computation
    float* h_A0 = (float*)malloc(sz);
    float* h_b0 = (float*)malloc(n * sizeof(float));
    if (!h_A0 || !h_b0) {
        fprintf(stderr, "Host malloc failed for backups\n");
        return -1;
    }
    memcpy(h_A0, h_A, sz);
    memcpy(h_b0, h_b, n * sizeof(float));

    // Allocate device memory for A
    float* d_A;
    cudaMalloc(&d_A, sz);

    // Copy A to device
    cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);

    // Start timer for the solve phase
    auto t0 = std::chrono::high_resolution_clock::now();

    // Perform LU decomposition and triangular solve
    lu_decompose(d_A, n);
    cudaMemcpy(h_A, d_A, sz, cudaMemcpyDeviceToHost);
    host_tri_solve(h_A, h_b, h_x, n);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Stop timer and print solve time
    auto t1 = std::chrono::high_resolution_clock::now();
    double solve_secs = std::chrono::duration<double>(t1 - t0).count();
    printf("total solve time = %f s\n", solve_secs);

    // Compute relative residual: ||A0*x - b0|| / ||b0||
    float err = 0, norm = 0;
    for (int i = 0; i < n; ++i) {
        float s = 0;
        for (int j = 0; j < n; ++j) {
            s += h_A0[i * n + j] * h_x[j];
        }
        float r = s - h_b0[i];
        err += r * r;
        norm += h_b0[i] * h_b0[i];
    }
    printf("relative residual %e\n", sqrt(err / norm));

    // Free device and host memory
    cudaFree(d_A);
    free(h_A); free(h_b); free(h_x);
    free(h_A0); free(h_b0);
    return 0;
}
