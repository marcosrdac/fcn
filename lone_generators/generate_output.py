from os import listdir, makedirs, remove
from os.path import join
import numpy as np
from train_config import DATA_DIRS, CLASSES
from train_config import open_data
from utils.labelutils import gen_dataset

makedirs(DATA_DIRS['output'], exist_ok=True)

X, Y, names = gen_dataset(DATA_DIRS['input'],
                          DATA_DIRS['label'],
                          CLASSES.ids.get,
                          mask_dtype=np.int8,
                          x_as_function=True,
                          return_names=True,
                          data_opener=open_data)

for f in listdir(DATA_DIRS['output']):
    path = join(DATA_DIRS['output'], f)
    remove(path)

for y, name in zip(Y, names):
    output_path = join(DATA_DIRS['output'], f'{name}.npy')
    np.save(output_path, y)
