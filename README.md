# ECE452 Projects

## Project 2

For this portion of the project, we experimented with matrix multiplication algorithms on CPUs and GPUs. We first implemented a naive algorithm, which runs a standard matrix multiplication and vector addition. Mathematically, this runs in O(n^3) time; but in terms of cache utilization, it runs poorly. In the second, we wrote a cache-aware, tiled matrix multiplication algorithm, which attacks the matrices in chunks such that the cache has better utilization.

We used numpy to handle the matrices and vectors, and also used its native matrix multiplication and vector addition functions, so that we'd inherit the full benefit of numpy's underlying [C library][01], compared to python which is sluggish.

Since the CPU was running on our own PCs, we each had different results, shown below:

### Arnav:

Running on an Intel Core i5-6300U @ 2.4GHz, running debian. Relevant output of `lscpu`:

```
Model name:             Intel(R) Core(TM) i5-6300U CPU @ 2.40GHz
CPU family:             6
Model:                  78
Thread(s) per core:     2
Core(s) per socket:     2
Socket(s):              1

...

Virtualization:         VT-x
L1d cache:              64 KiB (2 instances)
L1i cache:              64 KiB (2 instances)
L2 cache:               512 KiB (2 instances)
L3 cache:               3 MiB (1 instance)
```

<!-- Links -->
[01]: https://github.com/numpy/numpy/blob/main/numpy/_core/src/umath/matmul.c.src
