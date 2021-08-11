from os.path import join
from utils.labelutils import Label

DATA = '/home/marcosrdac/tmp/los/unet_training'
INPUT_DIR = join(DATA, 'image')  # just in this case!
OUTPUT_DIR = join(DATA, 'output')
IMAGE_DIR = join(DATA, 'image')
LABEL_DIR = join(DATA, 'label')
PATCHED_DIR = join(DATA, 'patched')
X_DIR = join(PATCHED_DIR, 'x')
Y_DIR = join(PATCHED_DIR, 'y')

CLASSES = [
    Label('oil', 'Oil spill'),
    Label('biofilm', 'Biological Film'),
    Label('rain', 'Rain cells'),
    Label('wind', 'Low-wind condition'),
    Label('ship', 'Ship'),
    Label('ship wake', 'Ship wake'),
    Label('ocean', 'Ocean'),
    Label('terrain', 'Land cover'),
]

CLASSES = [
    Label('sea', 'Sea'),
    Label('fish', 'Fish'),
    Label('plant', 'Plant'),
]
