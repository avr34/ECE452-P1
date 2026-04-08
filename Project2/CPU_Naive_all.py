import os
import re
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

test_sizes = [128, 256, 512, 1024]

# follows high low avg
all_times = []

def handle_outputs(result):
    lines = result.stdout.strip().split('\n')

    times = []
    native = []
    for line in lines:
        match = re.search(r'numpy: (?P<np>\d+\.\d+).+native: (?P<native>\d+\.\d+)', line)

        if match:
            j = match.groupdict()
            times.append(float(j['np']))
            native.append(float(j['native']))

    return [times, native]

for i in test_sizes:
    # Run the thing for 100 iterations, with native loop, and plot outputs
    result = subprocess.run(
        [
            sys.executable,
            'CPU_Naive.py',
            '-s', f'{i}',
            '-i', '100',
            '-n', '-p'
        ],
        capture_output=True,
        text=True
    )
    
    all_times.append(handle_outputs(result))
    
    print(f'finished {i}')

categories = ['Numpy', 'Native']
colors = {'Numpy': '#3498db', 'Native': '#e67e22'}

plt.figure(figsize=(10, 6))

for i, size_results in enumerate(all_times):
    x_center = test_sizes[i]

    for cat_idx, cat_name in enumerate(categories):
        values = size_results[cat_idx]

        offset = -2 if cat_idx == 0 else 2
        
        x_pos = x_center + offset

        bp = plt.boxplot(values, positions=[x_pos], widths=20, patch_artist=True, showfliers=False)

        plt.setp(bp['boxes'], facecolor=colors[cat_name], alpha=0.4)
        plt.setp(bp['medians'], color='black')

        jitter = np.random.uniform(-0.8, 0.8, size=len(values))
        plt.scatter(x_pos + jitter, values, color=colors[cat_name], s=25, alpha=0.7, label=cat_name if i==0 else "")

plt.xticks(test_sizes)
plt.yscale('log')
plt.xlabel('Matrix Size')
plt.ylabel('Time (seconds - log scale)')
plt.title('Performance Distribution')

plt.legend()
plt.grid(axis='y', which='both', linestyle='--', alpha=0.5)

plt.savefig(os.path.join('plots', 'CPU_Naive', 'compare_all'))
