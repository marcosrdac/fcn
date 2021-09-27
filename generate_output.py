from os import listdir, makedirs, remove
from os.path import join
import numpy as np
from config import INPUT_DIR, OUTPUT_DIR, IMAGE_DIR, LABEL_DIR, CLASSES
from utils.labelutils import gen_dataset
from utils.datautils import open_nc_variable, patch_dataset, gen_default_patch_cond

makedirs(OUTPUT_DIR, exist_ok=True)


def open_data(path):
    return open_nc_variable(path, 'Sigma0_VV_db')


CLASS_NAMES = [c.name for c in CLASSES]
X, Y, names = gen_dataset(INPUT_DIR,
                          LABEL_DIR,
                          CLASS_NAMES,
                          x_as_function=True,
                          # cut=200,
                          return_names=True,
                          open_data=open_data)

for f in listdir(OUTPUT_DIR):
    path = join(OUTPUT_DIR, f)
    remove(path)

for name, y in zip(names, Y):
    output_path = join(OUTPUT_DIR, f'{name}.npy')
    np.save(output_path, y)
