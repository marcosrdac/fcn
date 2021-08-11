from os import listdir, makedirs
from os.path import join, splitext
import numpy as np
from utils.osutils import with_same_basename
from config import INPUT_DIR, OUTPUT_DIR, PATCHED_DIR, IMAGE_DIR, CLASSES, X_DIR, Y_DIR
from utils.labelutils import gen_dataset
from utils.datautils import open_image, make_patches, gen_default_patch_cond

CLASS_NAMES = [c.name for c in CLASSES]

makedirs(X_DIR, exist_ok=True)
makedirs(Y_DIR, exist_ok=True)

input_filenames = listdir(INPUT_DIR)
output_filenames = listdir(OUTPUT_DIR)

open_data = open_image
shape = (64, ) * 2
# overlap = (.5, ) * 2
overlap = False
cond = gen_default_patch_cond(.05)

for output_filename in output_filenames:
    name = splitext(output_filename)[0]
    input_filename = with_same_basename(output_filename, input_filenames)

    x = open_data(join(INPUT_DIR, input_filename))
    y = np.load(join(OUTPUT_DIR, output_filename))

    x_patches, y_patches = make_patches(x=x,
                                        y=y,
                                        shape=shape,
                                        overlap=overlap,
                                        cond=cond)

    np.savez_compressed(join(X_DIR, f'{name}.npz'), x_patches)
    np.savez_compressed(join(Y_DIR, f'{name}.npz'), y_patches.astype(np.int8))
