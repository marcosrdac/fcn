from os import listdir
from os.path import join, splitext
import numpy as np
from utils.datautils import open_all_patched
from config import CLASSES, X_DIR, Y_DIR
import matplotlib.pyplot as plt

X_all, Y_all = open_all_patched(X_DIR, Y_DIR, show=True)

print(X_all)
print(len(X_all))
