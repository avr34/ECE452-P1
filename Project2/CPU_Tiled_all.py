import os
import re
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

test_sizes = [128, 256, 512, 1024]
block_size = 32
all_results = []

def run_simulation(size):
    result = subprocess.run(
        [sys.executable, 'CPU_Tiled.py', '-s', str(size), '-i', '5', '-b', str(block_size), '-p'],
        capture_output=True, text=True
    )
    
    np_times = [float(x) for x in re.findall(r'Numpy: (\d+\.\d+)', result.stdout)]
    nat_times = [float(x) for x in re.findall(r'Native: (\d+\.\d+)', result.stdout)]
    return np_times, nat_times

for s in test_sizes:
    print(f"Simulating size {s}...")
    all_results.append(run_simulation(s))

# Plotting
plt.figure(figsize=(10, 6))
colors = {'Numpy': '#3498db', 'Native': '#e67e22'}

for i, size in enumerate(test_sizes):
    np_vals, nat_vals = all_results[i]
    
    # Boxplots for both
    b1 = plt.boxplot(np_vals, positions=[size], widths=8, patch_artist=True)
    b2 = plt.boxplot(nat_vals, positions=[size], widths=8, patch_artist=True)
    
    plt.setp(b1['boxes'], facecolor=colors['Numpy'], alpha=0.5)
    plt.setp(b2['boxes'], facecolor=colors['Native'], alpha=0.5)

plt.yscale('log')
plt.xticks(test_sizes, test_sizes)
plt.title(f'Tiled Comparison (Block Size: {block_size})')
plt.ylabel('Time (s) - Log Scale')
plt.xlabel('Matrix Size')
plt.grid(True, which='both', alpha=0.3)
plt.savefig('plots/CPU_Tiled/comprehensive_tiled.png')
print("Simulation complete. Plot saved.")