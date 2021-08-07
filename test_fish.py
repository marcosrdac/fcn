from config import INPUT_DIR, OUTPUT_DIR, IMAGE_DIR, LABEL_DIR, CLASSES
from utils.labelutils import gen_dataset
from utils.datautils import patch_dataset, gen_default_patch_cond

CLASS_NAMES = [c.name for c in CLASSES]
X, Y = gen_dataset(IMAGE_DIR, LABEL_DIR, CLASS_NAMES, x_as_function=True)

X, Y = patch_dataset(X,
                     Y,
                     shape=(1000, 1000),
                     overlap=(.5, .5),
                     x_as_function=True,
                     cond=gen_default_patch_cond(.05))

print(X.shape)
print(Y.shape)
