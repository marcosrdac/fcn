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
DATA_DIRS['X'] = join(DATA_DIRS['data'], 'patched', 'x')
DATA_DIRS['Y'] = join(DATA_DIRS['data'], 'patched', 'y')
# -- Training outputs
DATA_DIRS['result'] = join(DATA_DIRS['data'], 'result')

# Class settings
# - Class set declarations
# -- General ocean phenomena classifier
OCEAN_CLASSES = Classes()
OCEAN_CLASSES.create('Oil spill', ['oil', 'oil spill'], color='red')
OCEAN_CLASSES.create('Biological film', ['biofilm', 'phyto', 'phytoplancton'],
                     color='forestgreen')
OCEAN_CLASSES.create('Rain cells', ['rain'], color='lightskyblue')
OCEAN_CLASSES.create('Low-wind condition', ['wind'], color='grey')
OCEAN_CLASSES.create('Ocean', ['ocean'], color='blue')
OCEAN_CLASSES.create('Ship', ['ship'], color='darkmagenta')
OCEAN_CLASSES.create('Ship wake', ['ship wake'], color='pink')
OCEAN_CLASSES.create('Land cover', ['land'], color='brown')
# -- Oil spill detector
OIL_CLASSES = Classes()
non_oil_name_sets = [n for i, n in OCEAN_CLASSES.names.items() if i > 0]
non_oil_names = ['non-oil', *itertools.chain(*non_oil_name_sets)]
OIL_CLASSES.create(OCEAN_CLASSES.descriptions[0], OCEAN_CLASSES.names[0])
OIL_CLASSES.create('Non-oil', non_oil_names, color='blue')
# -- Test classes
TEST_CLASSES = Classes()
TEST_CLASSES.create('Sea', ['sea'], color='blue')
TEST_CLASSES.create('Fish', ['fish'], color='orange')
TEST_CLASSES.create('Plant', ['plant'], color='green')
# - Class set definition (IMPORTANT DEFINITION)
CLASSES = OCEAN_CLASSES

# Input data opener
# open_data = open_image  # usual
open_data = partial(open_nc_variable, var='Sigma0_VV_db')

# Train output data settings
mask_dtype = np.int8  # no unsigned types here!

# Train data patch settings
PATCH_CONFIG = {}
# - Modify these parameters
patch_size = 64
overlap = False  # False or ratio
PATCH_CONFIG['shape'] = (patch_size, patch_size)
PATCH_CONFIG['overlap'] = (overlap, overlap) if overlap else False
PATCH_CONFIG['ignore_border'] = 5 / 100

# Train config
TRAIN_CONFIG = {}
# - Independent labels to also keep track of metrics
TRAIN_CONFIG['keep_label_info'] = [0]
metrics_bsc = ['accuracy', 'precision']
metrics_all = ['accuracy', 'precision', 'recall', 'f1-score']
TRAIN_CONFIG['metrics'] = [
    'loss', *metrics_bsc, *[f'{m}_0' for m in metrics_bsc]
]
TRAIN_CONFIG['batch_size'] = 20  # int or None
TRAIN_CONFIG['max_epochs'] = 10000
TRAIN_CONFIG['test_size'] = 1 / 3
# TRAIN_CONFIG['learning_rates'] = 10 ** np.linspace(-1, 2, 8)
# TRAIN_CONFIG['learning_rates'] = 1e-3, 1e-2, 1e-1,
TRAIN_CONFIG['learning_rates'] = 2e-2,
TRAIN_CONFIG['early_stopping'] = {
    'enable': True,
    'metric': 'accuracy',
    'greater_is_better': True,
    'patience': 20,
    'delta': 0,
}

# Model settings
UNET_CONFIG = {}
# - U-net architectures
UNET_CONFIG['models'] = {
    'a':
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

# Random number generation seeds
# - Model initialization
UNET_CONFIG['rngkeys'] = {}
UNET_CONFIG['rngkeys']['init'] = 0
UNET_CONFIG['rngkeys']['drop_init'] = 1
UNET_CONFIG['rngkeys']['drop_apply'] = 2
# - Stochastic optimization batch permutations
TRAIN_CONFIG['rngkeys'] = {}
TRAIN_CONFIG['rngkeys']['train'] = 3

# Other automatic configs
# WARNING: don't mess here unless you know what you are doing
# - Result directories generators
DATA_DIRS['result_dirs'] = lambda *d: {  # i.e. model_name, train_id
    'checkpoint': join(DATA_DIRS['result'], *d[:2], 'checkpoint'),
    'history_data': join(DATA_DIRS['result'], *d[:2], 'history_data'),
    'history_plot': join(DATA_DIRS['result'], *d[:2], 'plots', 'history'),
    'patch_plot': join(DATA_DIRS['result'], *d[:2], 'plots', 'patch'),
}

if __name__ == '__main__':
    from os import listdir
    from utils.pretty import pprint

    def nprint(*args, **kwargs):
        return print(*['\n' + str(args[0]), *args[1:]], **kwargs)

    print('Directory definitions:')
    for dir_nick, dir_path in DATA_DIRS.items():
        if dir_nick == 'result_dirs':
            continue
        dir_exists = 'X' if isdir(dir_path) else ' '
        num_files = len(listdir(dir_path)) if isdir(dir_path) else 0
        print(f'- [{dir_exists}]',
              f'{dir_nick} -> {dir_path!r}',
              f'(with {num_files} files)',
              sep=' ')

    nprint(f'Class definitions ({len(CLASSES)}):')
    for id, description in CLASSES.descriptions.items():
        print(f'- {description} ({CLASSES.colors[id]})')

    nprint('Patch settings:')
    pprint(PATCH_CONFIG)

    nprint('Train settings:')
    pprint(TRAIN_CONFIG, dontprint=('rngkeys'))

    nprint('U-net settings:')
    pprint(UNET_CONFIG, dontprint='rngkeys')
