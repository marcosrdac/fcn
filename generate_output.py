from os import makedirs
from os.path import join
import numpy as np
from config import INPUT_DIR, OUTPUT_DIR, IMAGE_DIR, LABEL_DIR, CLASSES
from utils.labelutils import gen_dataset
from utils.datautils import patch_dataset, gen_default_patch_cond

makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = [c.name for c in CLASSES]
X, Y, names = gen_dataset(IMAGE_DIR,
                          LABEL_DIR,
                          CLASS_NAMES,
                          x_as_function=True,
                          return_names=True)

for name, y in zip(names, Y):
    output_path = join(OUTPUT_DIR, f'{name}.npy')
    np.save(output_path, y)
