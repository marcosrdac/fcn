#!/usr/bin/env python3

from os import listdir, mkdir, makedirs, remove, rename
from os.path import join, dirname, basename, splitext
from datetime import datetime
from train_config import DATA_DIRS, CLASSES
from train_config import PATCH_CONFIG
from train_config import open_data, mask_dtype
from utils.osutils import with_same_basename
from utils.datautils import make_patches, gen_default_patch_cond
from utils.labelutils import gen_dataset
import numpy as np
from time import time

output_dirs = ['output', 'X', 'Y']
for dir_nick in output_dirs:
    print(f'Creating {dir_nick} directory',
          f'at {DATA_DIRS[dir_nick]!r}...',
          end=' ')
    makedirs(DATA_DIRS[dir_nick], exist_ok=True)
    print('Done!')
dirs_with_data = []
for dir_nick in output_dirs:
    if len(listdir(DATA_DIRS[dir_nick])) > 0:
        dirs_with_data.append(dir_nick)

if dirs_with_data:
    print()
    print('WARNING: there is old data under the following directories:',
          ', '.join(dirs_with_data) + '!')

    answered = False
    while not answered:
        procedure = input('How to resolve [Overwrite,Clear,Backup]? ').lower()

        if procedure.startswith('o'):
            answered = True

        elif procedure.startswith('c'):
            answered = True
            print('Removing old output files:')
            for dir_nick in dirs_with_data:
                dir_path = DATA_DIRS[dir_nick]
                print(f'- Removing files in {dir_nick}...', end=' ')
                filepaths = [join(dir_path, f) for f in listdir(dir_path)]
                for path in filepaths:
                    remove(path)
                print('Done!')

        elif procedure.startswith('b'):
            answered = True
            print('Making backups for old output files:')
            prefix = datetime.now().strftime('%Y%m%d%H%M%S%f_bkp_')
            for dir_nick in dirs_with_data:
                dir_path = DATA_DIRS[dir_nick]
                dir_path_bkp = join(dirname(dir_path),
                                    f'{prefix}{basename(dir_path)}')
                print(f'- Making a backup for {dir_nick} directory up',
                      f'at {dir_path_bkp!r}...',
                      end=' ')
                rename(dir_path, dir_path_bkp)
                print('Done!')
                print(f'- Creating a new {dir_nick} directory...', end=' ')
                mkdir(dir_path)
                print('Done!')
        else:
            print('Could not understand answer.', end=' ')

time_start = time()

print('Generating mask outputs (from labels, classes)')
X, Y, names = gen_dataset(DATA_DIRS['input'],
                          DATA_DIRS['label'],
                          CLASSES.ids.get,
                          mask_dtype=mask_dtype,
                          x_as_function=True,
                          return_names=True,
                          data_opener=open_data)

print()
print('Saving mask outputs')
n_outs = 0
for y, name in zip(Y, names):
    output_path = join(DATA_DIRS['output'], f'{name}.npy')
    print(f'- Output associated with {name!r} saved to: {output_path!r}')
    np.save(output_path, y)
    n_outs += 1

print()
print('Patching data (from inputs, outputs)')
cond = gen_default_patch_cond(PATCH_CONFIG['ignore_border'])

input_filenames = listdir(DATA_DIRS['input'])
output_filenames = listdir(DATA_DIRS['output'])

for output_filename in output_filenames:
    name = splitext(output_filename)[0]
    input_filename = with_same_basename(output_filename, input_filenames)
    if not input_filename:
        continue

    x = open_data(join(DATA_DIRS['input'], input_filename))
    y = np.load(join(DATA_DIRS['output'], output_filename))

    x_patches, y_patches = make_patches(x=x,
                                        y=y,
                                        shape=PATCH_CONFIG['shape'],
                                        overlap=PATCH_CONFIG['overlap'],
                                        cond=cond)

    x_path = join(DATA_DIRS['X'], f'{name}.npz')
    y_path = join(DATA_DIRS['Y'], f'{name}.npz')

    np.savez_compressed(x_path, x_patches)
    np.savez_compressed(y_path, y_patches)

    print(f'- Patched {name!r} to:', f'  - inputs: {x_path!r}',
          f'  - outputs: {y_path!r}')

inputs_not_matched = [
    input_filename for input_filename in input_filenames
    if not with_same_basename(input_filename, output_filenames)
]

outputs_not_matched = [
    output_filename for output_filename in output_filenames
    if not with_same_basename(output_filename, input_filenames)
]

if inputs_not_matched or outputs_not_matched:
    print()
    if inputs_not_matched:
        print('WARNING: the following input files have no associated outputs:',
              *(f'- {path}' for path in inputs_not_matched),
              sep='\n')

    if outputs_not_matched:
        print('WARNING: the following output files have no associated inputs:',
              *(f'- {path}' for path in outputs_not_matched),
              sep='\n')

interval = time() - time_start
mean_interval = interval / n_outs
print()
print('Done!')
print(f'Elapsed time: {interval:.0f} s; per image: {mean_interval:.0f} s)')
