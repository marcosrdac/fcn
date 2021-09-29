from os import listdir, makedirs, remove
from os.path import join, splitext
import numpy as np
from utils.osutils import with_same_basename
from train_config import DATA_DIRS, CLASSES, PATCH_CONFIG
from train_config import open_data
from utils.datautils import make_patches, gen_default_patch_cond

makedirs(DATA_DIRS['X'], exist_ok=True)
makedirs(DATA_DIRS['Y'], exist_ok=True)
for folder in (DATA_DIRS['X'], DATA_DIRS['Y']):
    for f in listdir(folder):
        path = join(folder, f)
        remove(path)

input_filenames = listdir(DATA_DIRS['input'])
output_filenames = listdir(DATA_DIRS['output'])

cond = gen_default_patch_cond(PATCH_CONFIG['ignore_border'])

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

    np.savez_compressed(join(DATA_DIRS['X'], f'{name}.npz'), x_patches)
    np.savez_compressed(join(DATA_DIRS['Y'], f'{name}.npz'), y_patches.astype(np.int8))
