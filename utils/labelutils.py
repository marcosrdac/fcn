from typing import Any
from dataclasses import dataclass
from base64 import b64decode
from copy import copy
from io import BytesIO
import json
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from PIL import Image
from os import listdir
from os.path import join, splitext
from .osutils import with_same_basename
from .datautils import open_image


@dataclass(frozen=True)
class LabelFile:
    flags: dict
    shapes: list
    img_shape: tuple
    img_path: str = ''
    img: np.array = np.zeros(0)
    version: str = ''
    cut_idx: Any = None

    def __repr__(self):
        data = ", ".join([
            f"flags={self.flags}", f"shapes={self.shapes}",
            f"img_shape={self.img_shape}", f"img_path={self.img_path}"
        ])
        return f'{self.__class__.__name__}({data})'

    @staticmethod
    def read(path: str):
        with open(path, 'r') as f:
            loaded = json.load(f)
        img_shape = loaded['imageHeight'], loaded['imageWidth']
        img_path = loaded['imagePath']
        version = loaded['version']
        flags = loaded['shapes']
        shapes = [Shape(**shape) for shape in loaded['shapes']]
        img = LabelFile.get_img(loaded['imageData'])
        return LabelFile(flags, shapes, img_shape, img_path, img, version)

    def cut(self, extra=0):
        assert extra >= 0
        h, w = self.img_shape
        x_min = y_min = np.inf
        x_max = y_max = 0

        for shape in self.shapes:
            _x_min, _y_min = shape.points.min(axis=0)
            _x_max, _y_max = shape.points.max(axis=0)
            x_min = int(np.floor(np.min((_x_min, x_min))))
            y_min = int(np.floor(np.min((_y_min, y_min))))
            x_max = int(np.ceil(np.max((_x_max, x_max))))
            y_max = int(np.ceil(np.max((_y_max, y_max))))

        if extra:
            x_min = np.max((x_min - extra, 0))
            y_min = np.max((y_min - extra, 0))
            x_max = np.min((x_max + extra, w))
            y_max = np.min((y_max + extra, h))

        new = copy(self)

        for shape in new.shapes:
            shape.points[:, 0] -= x_min
            shape.points[:, 1] -= y_min

        cut_idx = slice(y_min, y_max), slice(x_min, x_max)

        if self.img is not None:
            object.__setattr__(new, 'img', new.img[cut_idx])
            object.__setattr__(new, 'img_shape', new.img.shape[:2])
            object.__setattr__(new, 'cut_idx', cut_idx)

        return new, cut_idx

    def get_img(imgData):
        try:
            f = BytesIO()
            f.write(b64decode(imgData))
            return np.asarray(Image.open(f))
        except TypeError:
            return None


@dataclass(frozen=True)
class Shape:
    label: str
    points: np.array
    shape_type: str
    flags: dict
    group_id: int

    def __post_init__(self):
        object.__setattr__(self, 'points', np.asarray(self.points))

    def to_pixel_mask(self, shape):
        h, w = shape
        path = mpl.path.Path(self.points)
        y, x = np.mgrid[:h, :w]
        coords = np.stack((x.ravel(), y.ravel()), axis=1)
        mask = path.contains_points(coords).reshape(shape)
        return mask


@dataclass(frozen=True)
class Label:
    name: str
    full_name: str = None


def gen_dataset(data_dir,
                label_dir,
                class_names,
                show=False,
                cut=-1,
                open_data=open_image,
                x_as_function=False,
                return_names=False):
    data_filenames = listdir(data_dir)
    label_filenames = listdir(label_dir)

    X, Y, names = [], [], []

    for label_filename in label_filenames:
        name = splitext(label_filename)[0]
        names.append(name)

        label_file = LabelFile.read(join(label_dir, label_filename))

        if cut > -1:
            label_file, cut_idx = label_file.cut(cut)

        data_filename = with_same_basename(label_filename, data_filenames)

        def data_getter():
            data = open_data(join(data_dir, data_filename))
            if cut > -1:
                data = data[cut_idx]
            return data

        x = data_getter if x_as_function else data_getter()
        y = -np.ones(label_file.img_shape)
        for shape in label_file.shapes:
            num_label = class_names.index(shape.label)
            idx = shape.to_pixel_mask(label_file.img_shape)
            y[idx] = num_label

        X.append(x)
        Y.append(y)

        if show:
            fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
            axes[0].imshow(x() if x_as_function else x)
            axes[1].imshow(y)
            plt.show()

    if return_names:
        return X, Y, names
    else:
        return X, Y


def test_gen_dataset():
    DATA = '/home/marcosrdac/tmp/los/unet_training'
    IMAGE_DIR = join(DATA, 'image')
    LABEL_DIR = join(DATA, 'label')
    CLASS_NAMES = ['sea', 'fish', 'plant']

    X, Y = gen_dataset(IMAGE_DIR,
                       LABEL_DIR,
                       CLASS_NAMES,
                       show=True,
                       cut=0,
                       x_as_function=True)


def test_labelfile():
    label_paths = [
        '/home/marcosrdac/los/img/teste0_S1B_IW_SLC__1SDV_20210120T080527_20210120T080554_025235_030137_BB07_Cal_Orb_deb_ML.json',
    ]

    for label_path in label_paths:
        label_file = LabelFile.read(label_path)
        label_file, idx = label_file.cut()

        plt.imshow(label_file.img)
        for shape in label_file.shapes:
            plt.plot(*shape.points.T, marker='o')
        plt.show()


if __name__ == '__main__':
    test_labelfile()
    test_gen_dataset()
