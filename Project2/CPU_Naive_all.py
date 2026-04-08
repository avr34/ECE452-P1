import os
import re
import sys
import subprocess

test_sizes = [128, 256, 512, 1024]

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

    print(f'finished {i}')
