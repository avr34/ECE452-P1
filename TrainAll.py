import os
import sys
import subprocess

lr = [0.001, 0.01]
neurons = [1024, 512, 128, 64]

lr_filename = ['001', '01']

loss = ['mse','ce']
opt  = ['sgd','adam']

total = len(lr) * len(neurons) * len(loss) * len(opt)
count = 1

try:
    os.makedirs(os.path.join('outputs','pics'))
except (FileExistsError, FileNotFoundError):
    pass
except Exception as e:
    print(f"error making folder: {e}")
    sys.exit(0)

cd = os.path.join(os.getcwd(), 'outputs')

for i in neurons:
    for j in range(2):
        foldername = f'{i}_{lr_filename[j]}'
        try:
            os.makedirs(os.path.join(cd, foldername))
        except (FileExistsError, FileNotFoundError):
            pass
        except Exception as e:
            print(f"error making folder {foldername}: {e}")
            sys.exit(0)
        
        for l in loss:
            for o in opt:
                subprocess.run(['python3', 'Train.py', '--epochs', '2000', '--lr', f'{lr[j]}', '--output', os.path.join(cd, foldername, foldername + f'_{l}_{o}'), '--plot', os.path.join(cd, 'pics', foldername + f'_{l}_{o}')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f'Completed {count} out of {total}')
                count += 1

