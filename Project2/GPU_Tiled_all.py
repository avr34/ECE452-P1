import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

test_sizes = [512, 1024, 2048, 4096]
block_size = 32
results = {"Standard": [], "Tiled_Slicing": [], "CUDA_Kernel": []}

for size in test_sizes:
    print(f"Running GPU Benchmark for size {size}...")
    cmd = ["python3", "GPU_Tiled.py", "-s", str(size), "-i", "5", "-b", str(block_size)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output using regex
    std_times = [float(x) for x in re.findall(r"Standard: (\d+\.\d+)", res.stdout)]
    tile_times = [float(x) for x in re.findall(r"Tiled: (\d+\.\d+)", res.stdout)]
    ker_times = [float(x) for x in re.findall(r"Kernel: (\d+\.\d+)", res.stdout)]
    
    results["Standard"].append(np.mean(std_times))
    results["Tiled_Slicing"].append(np.mean(tile_times))
    results["CUDA_Kernel"].append(np.mean(ker_times))

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(test_sizes, results["Standard"], 'o-', label="CuPy Standard (@)", linewidth=2)
plt.plot(test_sizes, results["Tiled_Slicing"], 's--', label="CuPy Tiled (Slicing)", alpha=0.8)
plt.plot(test_sizes, results["CUDA_Kernel"], '^:', label="CUDA Kernel (Manual)", alpha=0.8)

plt.yscale('log')
plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Time (seconds) - Log Scale")
plt.title(f"GPU Matrix Multiplication Performance (T4 GPU)\nBlock Size: {block_size}")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.savefig('plots/GPU_Tiled/comprehensive.png')