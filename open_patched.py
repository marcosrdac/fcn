#!/usr/bin/env python3

from train_config import DATA_DIRS
from utils.datautils import open_all_patched

X_all, Y_all = open_all_patched(DATA_DIRS['X'], DATA_DIRS['Y'], show=True)
