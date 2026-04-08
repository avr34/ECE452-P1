# ECE452 - Project #2
# Group Members: Arnav Revankar, Raghav Bharathan, and Daniel Assaf

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

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='time some naive matrix operations (A * B + C)')

    parser.add_argument('-s', '--size', type=int, default=128, help='Dimension of A (square matrix), and rows of B and C.')
    parser.add_argument('-i', '--iterations', type=int, default=1, help='Number of times to iterate and take average.')
    parser.add_argument('-n', '--native', action='store_true', help='Include this flag to also run the native python 3 nested loop thing.')

    args = parser.parse_args()
