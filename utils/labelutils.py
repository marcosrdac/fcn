from typing import Any, Type, Callable, Sequence, Dict
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
try:
    from .osutils import with_same_basename
    from .datautils import open_image
except ImportError:
    from osutils import with_same_basename
    from datautils import open_image


@dataclass(frozen=True)
class LabelFile:
    '''Class used to operate LabelMe files.'''
    flags: dict
    shapes: list
    img_shape: tuple
    img_path: str = ''
    img: np.array = None
    version: str = ''
    cut_idx: Any = None

    def __repr__(self):
        data = ", ".join([
            f"flags={self.flags}", f"shapes={self.shapes}",
            f"img_shape={self.img_shape}", f"img_path={self.img_path}"
        ])
        return f'{self.__class__.__name__}({data})'

    @staticmethod
    def read(path: str, load_img=False):
        '''
        Create a LabelFile object from a JSON LabelMe output's path.
        `load_img` tells us if we should open the image inside the JSON.
        '''
        with open(path, 'r') as f:
            loaded = json.load(f)

        img_shape = loaded['imageHeight'], loaded['imageWidth']
        img_path = loaded['imagePath']
        version = loaded['version']
        flags = loaded['shapes']
        shapes = [Shape(**shape) for shape in loaded['shapes']]

        if load_img:
            img = LabelFile.load_img(loaded['imageData'])
        else:
            img = None

        return LabelFile(flags, shapes, img_shape, img_path, img, version)

    def load_img(imgData):
        '''
        Method to load image data from inside JSON LabelMe output.
        '''
        try:
            f = BytesIO()
            f.write(b64decode(imgData))
            return np.asarray(Image.open(f))
        except TypeError:
            return None

    def cut(self, extra=0):
        '''
        Cut data extents to get a new, small object. `extra` pads minimal
        dimensions of the LabelFile.
        '''
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
        new_shape = y_max - y_min, x_max - x_min

        new = copy(self)

        for shape in new.shapes:
            shape.points[:, 0] -= x_min
            shape.points[:, 1] -= y_min

        cut_idx = slice(y_min, y_max), slice(x_min, x_max)

        if self.img is not None:
            object.__setattr__(new, 'img', new.img[cut_idx])
        object.__setattr__(new, 'img_shape', new_shape)
        object.__setattr__(new, 'cut_idx', cut_idx)

        return new, cut_idx


@dataclass(frozen=True)
class Shape:
    '''
    Class to operate on polygons and associated labels inside a LabelMe JSON
    file.
    '''
    label: str
    points: np.array
    shape_type: str
    flags: dict
    group_id: int

    def __post_init__(self):
        object.__setattr__(self, 'points', np.asarray(self.points))

    def indices(self):
        '''
        Get indices of image array that are inside this shape's polygon.
        '''
        # LabelMe coords are (x,y), not (i,j)
        poly = self.points[:, ::-1]

        crd_min = np.floor(np.min(poly, axis=0)).astype(int)
        crd_max = np.ceil(np.max(poly, axis=0)).astype(int)

        win_size = crd_max - crd_min
        win_idx = np.indices(win_size)
        win_crds = np.column_stack([d.flat for d in win_idx])

        crds = crd_min[None, :] + win_crds
        del win_crds

        path = mpl.path.Path(poly)
        mask = path.contains_points(crds)
        idx = tuple(crds[mask].T)
        return idx


class Classes:
    '''
    Class used to manage a set of classes associated with unique ids and sets 
    of labels.
    '''
    ids: Dict[str, int]
    descriptions: Dict[int, str]
    names: Dict[int, list]
    main_names: Dict[int, str]

    def __init__(self):
        self.ids = {}
        self.descriptions = {}
        self.names = {}
        self.main_names = {}

    def __len__(self):
        return len(self.main_names)

    def __repr__(self):
        return str(self.main_names)

    def create(self, description: str, names: Sequence):
        '''
        Create a label for the `Classes` instance. Names can be a string or a sequence of strings.
        '''
        id = len(self.main_names)

        if isinstance(names, str):
            main_name = names
            names = [names]
        else:
            main_name = names[0]
            names = [*names]

        for name in names:
            self.ids[name] = id
        self.main_names[id] = main_name
        self.names[id] = names
        self.descriptions[id] = description

    def remove(self, id, keepids=True):
        '''
        Remove a label from this `Classes` instance. If `keepids` is `false`,
        rearrange ids so that they are uniform again.
        '''
        for name in self.names[id]:
            del self.ids[name]
        del self.names[id]
        del self.main_names[id]
        del self.descriptions[id]
        if not keepids:
            for new_id in range(id, len(self) - 1):
                old_id = new_id + 1

                self.names[new_id] = self.names[old_id]
                self.main_names[new_id] = self.main_names[old_id]
                self.descriptions[new_id] = self.descriptions[old_id]
        self.ids = {n: (i if i < id else i - 1) for n, i in self.ids.items()}


def gen_dataset(data_dir: str,
                label_dir: str,
                label_to_id: Callable,
                show: bool = False,
                cut: int = -1,
                data_opener: Callable = open_image,
                x_as_function: Callable = False,
                mask_dtype: Type = np.int8,
                return_names: bool = False):
    '''
    Generates dataset from original files, LabelMe files and a function that 
    maps label names to class ids. `cut` = -1 means that we do not want to cut
    the to only the useful parts.
    '''

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
            data = data_opener(join(data_dir, data_filename))
            if cut > -1:
                data = data[cut_idx]
            return data

        x = data_getter if x_as_function else data_getter()
        y = -np.ones(label_file.img_shape, dtype=mask_dtype)
        for shape in label_file.shapes:
            num_label = label_to_id(shape.label)
            if num_label is not None:
                y[shape.indices()] = num_label
            else:
                print(f'WARNING: not using undescribed label {shape.label!r}')

        X.append(x)
        Y.append(y)

        if show:
            fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
            axes[0].imshow(x() if x_as_function else x, cmap='gray')
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
    classes = Classes()
    classes.create('Oil spill label', ['oil', 'oil spill'])
    classes.create('Ocean label', ['ocean', 'sea'])

    X, Y = gen_dataset(IMAGE_DIR,
                       LABEL_DIR,
                       classes.ids.get,
                       show=True,
                       cut=0,
                       x_as_function=True)


def test_labelfile(cut=False):
    label_paths = [
        '/home/marcosrdac/tmp/los/unet_training/label/S1A_IW_SLC__1SDV_20170810T024712_20170810T024738_017855_01DEF7_445E.json',
    ]

    for label_path in label_paths:
        label_file = LabelFile.read(label_path, load_img=False)
        if cut:
            label_file, idx = label_file.cut()

        y = np.zeros(label_file.img_shape, dtype=np.int8)
        for s, shape in enumerate(label_file.shapes, start=1):
            plt.plot(*shape.points.T, marker='o')
            idx = shape.indices()
            y[idx] = s
        plt.imshow(y)
        plt.show()


if __name__ == '__main__':
    test_labelfile()
    test_gen_dataset()
