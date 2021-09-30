#!/usr/bin/env python3

from os.path import join, isdir
from functools import partial
import numpy as np
import itertools
from utils.labelutils import Classes
from utils.datautils import open_image, open_nc_variable

# Directory settings
# - Base data directory
DATA_DIRS = {'data': '/home/marcosrdac/tmp/los/unet_training'}
# - User managed directories
# -- Input data
DATA_DIRS['input'] = join(DATA_DIRS['data'], 'input')
# -- Image (JPG/PNG) versions of input data
DATA_DIRS['image'] = join(DATA_DIRS['data'], 'image')  # usually 'inputs'
# -- Label files produced by LabelMe from image data
DATA_DIRS['label'] = join(DATA_DIRS['data'], 'label')
# - Automatically managed directories
# -- Produced output masks from inputs and label files
DATA_DIRS['output'] = join(DATA_DIRS['data'], 'output')
# -- Patches of inputs (X) and outputs (Y)
DATA_DIRS['patched'] = join(DATA_DIRS['data'], 'patched')
DATA_DIRS['X'] = join(DATA_DIRS['patched'], 'x')
DATA_DIRS['Y'] = join(DATA_DIRS['patched'], 'y')

# Class settings
# - Class set declarations
# -- General ocean phenomena classifier
OCEAN_CLASSES = Classes()
OCEAN_CLASSES.create('Oil spill', ['oil', 'oil spill'])
OCEAN_CLASSES.create('Biological film', ['biofilm', 'phyto', 'phytoplancton'])
OCEAN_CLASSES.create('Rain cells', ['rain'])
OCEAN_CLASSES.create('Low-wind condition', ['wind'])
OCEAN_CLASSES.create('Ship', ['ship'])
OCEAN_CLASSES.create('Ship wake', ['ship wake'])
OCEAN_CLASSES.create('Ocean', ['ocean'])
OCEAN_CLASSES.create('Land cover', ['land'])
# -- Oil spill detector
OIL_CLASSES = Classes()
non_oil_name_sets = [n for i, n in OCEAN_CLASSES.names.items() if i > 0]
non_oil_names = ['non-oil', *itertools.chain(*non_oil_name_sets)]
OIL_CLASSES.create(OCEAN_CLASSES.descriptions[0], OCEAN_CLASSES.names[0])
OIL_CLASSES.create('Non-oil', non_oil_names)
# -- Test classes
TEST_CLASSES = Classes()
TEST_CLASSES.create('Sea', ['sea'])
TEST_CLASSES.create('Fish', ['fish'])
TEST_CLASSES.create('Plant', ['plant'])
# - Class set definition
CLASSES = OCEAN_CLASSES

# Input data opener
# open_data = open_image  # usual
open_data = partial(open_nc_variable, var='Sigma0_VV_db')

# Train output data settings
mask_dtype = np.int8  # cannot be unsigned!

# Train data patch settings
PATCH_CONFIG = {}
# - Modify these parameters
patch_size = 64
overlap = False  # False or ratio
PATCH_CONFIG['shape'] = (patch_size, patch_size),
PATCH_CONFIG['overlap'] = (overlap, overlap) if overlap else False
PATCH_CONFIG['ignore_border'] = 5 / 100

# Train config
TRAIN_CONFIG = {}
# - Independent labels to also keep track of metrics
TRAIN_CONFIG['keep_label_info'] = [0,]
TRAIN_CONFIG['batch_size'] = 20  # int or None
TRAIN_CONFIG['max_epochs'] = 2  # 50
TRAIN_CONFIG['test_size'] = 1 / 3
# TRAIN_CONFIG['learning_rates'] = 10**np.linspace(-1, 2, 8)
# TRAIN_CONFIG['learning_rates'] = 1e-3, 1e-2, 1e-1,
TRAIN_CONFIG['learning_rates'] = 2e-2,

# - rngkeys
TRAIN_CONFIG['rngkeys'] = {}
TRAIN_CONFIG['rngkeys']['train'] = 0

# Model settings
UNET_CONFIG = {}
# - rngkeys
UNET_CONFIG['rngkeys'] = {}
UNET_CONFIG['rngkeys']['init'] = 1
UNET_CONFIG['rngkeys']['drop_init'] = 2
UNET_CONFIG['rngkeys']['drop_apply'] = 3
# - U-net architectures
UNET_CONFIG['architectures'] = {
    'model_a':
    dict(
        rescale=(-2, -2, 0, 2, 2),
        nfeat=(
            (8, ),
            (16, ),
            (32, ),
            (16, ),
            (8, ),
        ),
        norm=True,
        # drop=(),
        droplast=.3),
    # 'b':
    # dict(
    #    rescale=(-2, -2, 0, 2, 2),
    #    nfeat=(
    #        (8, ),
    #        (16, ),
    #        (32, ),
    #        (16, ),
    #        (8, ),
    #    ),
    #    norm=True,
    #    # drop=(),
    #    droplast=.3),
}

if __name__ == '__main__':
    from os import listdir
    from yaml import dump

    print('Directories defined')
    for dir_nick, dir_path in DATA_DIRS.items():
        dir_exists = 'X' if isdir(dir_path) else ' '
        num_files = len(listdir(dir_path)) if isdir(dir_path) else 0
        print(f'- [{dir_exists}]',
              f'{dir_nick} -> {dir_path!r}',
              f'(with {num_files} files)',
              sep=' ')
    print()
    print(f'Classes defined ({len(CLASSES)}):',
          *dump(CLASSES.descriptions, default_flow_style=False).splitlines(),
          sep='\n- ')
    print()
    print(
        'Patch settings:',
        # *dump(PATCH_CONFIG, default_flow_style=False).splitlines(),
        PATCH_CONFIG,
        sep='\n  ')
    print()
    print('U-net configurations', UNET_CONFIG, sep='\n  ')
