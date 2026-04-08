# ECE452 - Project #2
# Group Members: Arnav Revankar, Raghav Bharathan, and Daniel Assaf

import os
import sys
import timeit
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Runs A x B + C using numpy, returns result and time elapsed
def naive_gemm_numpy(A, B, C: np.ndarray) -> tuple[np.ndarray, float]:
    try:
        start = timeit.default_timer() #------------------- Start timer

        ret = A @ B + C

        stop = timeit.default_timer() #-------------------- Stop timer

        elapsed = stop - start
    except Exception as e:
        print(f'Error occurred while doing numpy MAC: {e}')
        sys.exit(1)

    return ret, elapsed

# Runs the same using native python, returns result and time elapsed
def naive_gemm_python(A, B, C: np.ndarray) -> tuple[np.ndarray, float]:
    try:
        # make sure they match
        rows_A, cols_A = A.shape
        rows_B, cols_B = B.shape

        if cols_A != rows_B:
            raise ValueError('Columns of A must match rows of B')

        ret = np.zeros((rows_A, cols_B))

        start = timeit.default_timer() #------------------- Start timer

        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    ret[i, j] += A[i, k] * B[k, j]

                ret[i, j] += C[i, j]

        stop = timeit.default_timer() #-------------------- Stop timer

        elapsed = stop - start
    except Exception as e:
        print(f'Error occurred while doing python MAC: {e}')
        sys.exit(1)

    return ret, elapsed

def plot_times(times, label, filename, avg_time):
    dirr = 'plots/CPU_Naive'
    os.makedirs(dirr, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(times, marker='o', linestyle='-', alpha=0.7, label=f'Iteration time')
    plt.axhline(y=avg_time, color='r', linestyle='--', label=f'Average time {avg_time:.6f}s')

    plt.title(f'Performance Analysis: {label}')
    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    plt.savefig(os.path.join(dirr, filename))
    print(f'Plot saved as {filename}')

def plot_both(times, native, filename, avg1, avg2):
    dirr = 'plots/CPU_Naive'
    os.makedirs(dirr, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(times, color='b', marker='o', linestyle='-', alpha=0.7, label=f'Iteration time (numpy)')
    plt.plot(native, color='r', marker='o', linestyle='-', alpha=0.7, label=f'Iteration time (native)')
    plt.axhline(y=avg1, color='b', linestyle='--', label=f'Average time {avg1:.6f}s')
    plt.axhline(y=avg2, color='r', linestyle='--', label=f'Average time {avg2:.6f}s')

    plt.title(f'Performance Analysis of both')
    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    plt.savefig(os.path.join(dirr, filename))
    print(f'Plot saved as {filename}')

def native_thing(args: argparse.Namespace) -> None:
    times = []
    native = []

    A = np.random.rand(args.size, args.size)
    B = np.random.rand(args.size, 1)
    C = np.random.rand(args.size, 1)

    for i in range(args.iterations):
        _, j = naive_gemm_numpy(A, B, C)
        times.append(j)

        _, k = naive_gemm_python(A, B, C)
        native.append(k)
        
        print(f'Iteration {i+1} of {args.iterations}.\tTime numpy: {j:.6f}\tTime native: {k:.6f}')

    avg_times = sum(times) / len(times)
    avg_native = sum(native) / len(times)

    print(f'Average numpy time: {avg_times:.6f}\tAverage native time: {avg_native:.6f}')

    if args.plot:
        plot_both(times, native, f'{args.size}_{args.iterations}_both.png', avg_times, avg_native)
        plot_times(native, 'Native Python', f'{args.size}_{args.iterations}_native.png', avg_native)
        plot_times(times, 'Numpy', f'{args.size}_{args.iterations}_numpy.png', avg_times)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='time some naive matrix operations (A * B + C)')

    parser.add_argument('-s', '--size',         type=int, default=128,  help='Dimension of A (square matrix), and rows of B and C.')
    parser.add_argument('-i', '--iterations',   type=int, default=1,    help='Number of times to iterate and take average.')
    parser.add_argument('-n', '--native',       action='store_true',    help='Include this flag to also run the native python 3 nested loop thing.')
    parser.add_argument('-p', '--plot',         action='store_true',    help='Plot the outputs.')

    args = parser.parse_args()

    if args.iterations < 1:
        print('must have positive iteration count')
        sys.exit(1)
    
    if args.size < 1:
        print('must have positive matrix size')
        sys.exit(1)

    if args.native:
        native_thing(args)
    else:
        times = []

        A = np.random.rand(args.size, args.size)
        B = np.random.rand(args.size, 1)
        C = np.random.rand(args.size, 1)

        for i in range(args.iterations):
            _, j = naive_gemm_numpy(A, B, C)
            times.append(j)

            print(f'Iteration {i+1} of {args.iterations}.\tTime numpy: {j:.6f}')

        avg_times = sum(times) / len(times)

        print(f'Average numpy time: {avg_times:.6f}')

        if args.plot:
            plot_times(times, 'Numpy', f'{args.size}_{args.iterations}_numpy.png', avg_times)
