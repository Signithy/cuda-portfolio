#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>

// Tiling dimension for update kernel
#define TILE_DIM 16

// Error-checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel: Division step of LU (in-place) for column k
__global__ void kernel_div(float* A, int n, int k) {
    __shared__ float pivot;
    if (threadIdx.x == 0) {
        pivot = A[k * n + k];
    }
    __syncthreads();

    int idx = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        A[idx * n + k] /= pivot;
    }
}

// Kernel: Update step of LU (in-place) for submatrix starting at (row,col)
__global__ void kernel_update(float* A, int n, int k) {
    __shared__ float s_l[TILE_DIM];
    __shared__ float s_u[TILE_DIM];

    int row = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    int col = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;

    // Load column factor and row element into shared memory
    if (threadIdx.x == 0 && row < n) {
        s_l[threadIdx.y] = A[row * n + k];
    }
    if (threadIdx.y == 0 && col < n) {
        s_u[threadIdx.x] = A[k * n + col];
    }
    __syncthreads();

    // Apply rank-1 update
    if (row < n && col < n) {
        A[row * n + col] -= s_l[threadIdx.y] * s_u[threadIdx.x];
    }
}

// Performs in-place LU decomposition using the above kernels
void lu_decompose(float* d_A, int n) {
    const int divBlockSize = 256;
    dim3 updBlock(TILE_DIM, TILE_DIM);

    // Only loop k from 0 to n-2 to avoid zero-size launches
    for (int k = 0; k < n - 1; ++k) {
        // Division kernel
        int rows = n - k - 1;
        int divGridSize = (rows + divBlockSize - 1) / divBlockSize;
        kernel_div<<<divGridSize, divBlockSize>>>(d_A, n, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update kernel
        dim3 updGrid(
            (rows + TILE_DIM - 1) / TILE_DIM,
            (rows + TILE_DIM - 1) / TILE_DIM
        );
        kernel_update<<<updGrid, updBlock>>>(d_A, n, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Host-side triangular solve: Ly = b, then Ux = y (overwrites b with x)
void host_tri_solve(const float* A, float* b, int n) {
    // Forward substitution (L has unit diagonal)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            b[i] -= A[i * n + j] * b[j];
        }
    }
    // Back substitution
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) {
            b[i] -= A[i * n + j] * b[j];
        }
        b[i] /= A[i * n + i];
    }
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t size = n * n * sizeof(float);

    // Host allocations
    float* h_A  = (float*)malloc(size);
    float* h_b  = (float*)malloc(n * sizeof(float));
    if (!h_A || !h_b) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize A with random values + diagonal dominance, b with random values
    srand(0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_A[i * n + j] = ((float)rand() / RAND_MAX) + (i == j ? n : 0);
        }
        h_b[i] = (float)rand() / RAND_MAX;
    }

    // Backup for residual computation
    float* h_A0 = (float*)malloc(size);
    float* h_b0 = (float*)malloc(n * sizeof(float));
    memcpy(h_A0, h_A, size);
    memcpy(h_b0, h_b, n * sizeof(float));

    // Allocate device memory and copy A
    float* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    // Time LU + solve
    auto t0 = std::chrono::high_resolution_clock::now();

    lu_decompose(d_A, n);
    CUDA_CHECK(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));

    // Host solve
    host_tri_solve(h_A, h_b, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    printf("Solve time: %.6f sec\n", sec);

    // Compute relative residual
    double err = 0.0, norm = 0.0;
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += h_A0[i * n + j] * h_b[j];
        }
        double r = sum - h_b0[i];
        err  += r * r;
        norm += (double)h_b0[i] * h_b0[i];
    }
    printf("Relative residual: %e\n", sqrt(err / norm));

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    free(h_A); free(h_b);
    free(h_A0); free(h_b0);
    return EXIT_SUCCESS;
}

