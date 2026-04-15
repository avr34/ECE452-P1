import os
import sys
import timeit
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Tiled GEMM using Numpy Slicing (3 loops)
def tiled_gemm_numpy(A, B, C, b_size) -> tuple[np.ndarray, float]:
    try:
        rows_A, cols_A = A.shape
        cols_B = B.shape[1]
        ret = np.copy(C).astype(np.float64)

        start = timeit.default_timer()
        for i in range(0, rows_A, b_size):
            for j in range(0, cols_B, b_size):
                for k in range(0, cols_A, b_size):
                    # Numpy handles the inner multiplication of the blocks
                    ret[i:i+b_size, j:j+b_size] += (
                        A[i:i+b_size, k:k+b_size] @ B[k:k+b_size, j:j+b_size]
                    )
        stop = timeit.default_timer()
        return ret, stop - start
    except Exception as e:
        print(f'Error in Tiled Numpy: {e}')
        sys.exit(1)

# Tiled GEMM using Native Python (6 loops)
def tiled_gemm_native(A, B, C, b_size) -> tuple[np.ndarray, float]:
    try:
        rows_A, cols_A = A.shape
        cols_B = B.shape[1]
        ret = np.copy(C).astype(np.float64)

        start = timeit.default_timer()
        for i_t in range(0, rows_A, b_size):
            for j_t in range(0, cols_B, b_size):
                for k_t in range(0, cols_A, b_size):
                    # Inner element-wise loops
                    for i in range(i_t, min(i_t + b_size, rows_A)):
                        for j in range(j_t, min(j_t + b_size, cols_B)):
                            for k in range(k_t, min(k_t + b_size, cols_A)):
                                ret[i, j] += A[i, k] * B[k, j]
        stop = timeit.default_timer()
        return ret, stop - start
    except Exception as e:
        print(f'Error in Tiled Native: {e}')
        sys.exit(1)

def plot_tiled_results(results, args):
    dirr = 'plots/CPU_Tiled'
    os.makedirs(dirr, exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    for label, times in results.items():
        avg = sum(times)/len(times)
        plt.plot(times, marker='o', label=f'{label} (Avg: {avg:.4f}s)')
    
    plt.title(f'Tiled Performance (Size: {args.size}, Block: {args.block_size})')
    plt.xlabel('Iteration')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(dirr, f'tiled_{args.size}_b{args.block_size}.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', type=int, default=128)
    parser.add_argument('-i', '--iterations', type=int, default=5)
    parser.add_argument('-b', '--block-size', type=int, default=32)
    parser.add_argument('-p', '--plot', action='store_true')
    args = parser.parse_args()

    A = np.random.rand(args.size, args.size)
    B = np.random.rand(args.size, args.size)
    C = np.random.rand(args.size, args.size)

    stats = {"Tiled_Numpy": [], "Tiled_Native": []}

    for i in range(args.iterations):
        _, t_np = tiled_gemm_numpy(A, B, C, args.block_size)
        _, t_nat = tiled_gemm_native(A, B, C, args.block_size)
        
        stats["Tiled_Numpy"].append(t_np)
        stats["Tiled_Native"].append(t_nat)
        
        print(f"Iter {i+1} | Numpy: {t_np:.5f}s | Native: {t_nat:.5f}s")

    if args.plot:
        plot_tiled_results(stats, args)
