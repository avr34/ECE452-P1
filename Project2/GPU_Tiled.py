import os
import sys
import timeit
import argparse
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

# 1. Standard CuPy GEMM (Highly optimized)
def gemm_cupy_standard(A, B, C):
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    
    start.record()
    # GPU standard multiplication
    ret = cp.matmul(A, B) + C
    end.record()
    
    end.synchronize()
    elapsed = cp.cuda.get_elapsed_time(start, end) / 1000.0 # Convert ms to seconds
    return ret, elapsed

# 2. Tiled CuPy GEMM (Python loops + CuPy slicing)
def gemm_cupy_tiled(A, B, C, b_size):
    rows_A, cols_A = A.shape
    cols_B = B.shape[1]
    ret = cp.copy(C)
    
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    
    start.record()
    for i in range(0, rows_A, b_size):
        for j in range(0, cols_B, b_size):
            for k in range(0, cols_A, b_size):
                # Slicing on GPU
                ret[i:i+b_size, j:j+b_size] += (
                    A[i:i+b_size, k:k+b_size] @ B[k:k+b_size, j:j+b_size]
                )
    end.record()
    
    end.synchronize()
    return ret, cp.cuda.get_elapsed_time(start, end) / 1000.0

# 3. CUDA Tiled Kernel (The "Native" GPU Tiling approach)
# This implements the logic inside a single CUDA kernel
tiled_kernel = cp.RawKernel(r'''
extern "C" __global__
void tiled_gemm(const float* A, const float* B, float* C, int N, int b_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float val = 0;
        for (int k = 0; k < N; k++) {
            val += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] += val;
    }
}
''', 'tiled_gemm')

def gemm_cuda_kernel(A, B, C, b_size):
    N = A.shape[0]
    ret = cp.copy(C)
    
    # Configure grid/block
    threads_per_block = (16, 16)
    grid_size = (int(np.ceil(N / 16)), int(np.ceil(N / 16)))
    
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    
    start.record()
    tiled_kernel(grid_size, threads_per_block, (A, B, ret, N, b_size))
    end.record()
    
    end.synchronize()
    return ret, cp.cuda.get_elapsed_time(start, end) / 1000.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', type=int, default=1024)
    parser.add_argument('-i', '--iterations', type=int, default=10)
    parser.add_argument('-b', '--block-size', type=int, default=32)
    args = parser.parse_args()

    # Create data directly on GPU
    A = cp.random.rand(args.size, args.size, dtype=cp.float32)
    B = cp.random.rand(args.size, args.size, dtype=cp.float32)
    C = cp.random.rand(args.size, args.size, dtype=cp.float32)

    print(f"Benchmarking GPU (T4) | Size: {args.size}x{args.size} | Block: {args.block_size}")
    
    for i in range(args.iterations):
        _, t_std = gemm_cupy_standard(A, B, C)
        _, t_tile = gemm_cupy_tiled(A, B, C, args.block_size)
        _, t_ker = gemm_cuda_kernel(A, B, C, args.block_size)
        
        print(f"Iter {i+1} | CuPy Standard: {t_std:.6f}s | CuPy Tiled: {t_tile:.6f}s | CUDA Kernel: {t_ker:.6f}s")