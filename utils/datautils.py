import numpy as np
from os.path import join, splitext, basename
from PIL import Image


def open_image(path):
    name, ext = splitext(basename(path))
    if ext in ['.jpg', '.png']:
        img = Image.open(path)
        img = np.asarray(img)
        return img
    else:
        raise NotImplementedError


def make_patches(x, shape, y=None, overlap=False, cond=None):
    '''
    Creates sequential patches of image such that a condition is satisfied
    for each patch. Overlap can be chosen.
    '''
    nh_arr, nw_arr = x.shape[:2]
    if overlap:
        if isinstance(overlap[0], float):
            overlap = tuple(int(o * s) for o, s in zip(overlap, shape))
    nh_pat, nw_pat = shape
    h_gap, w_gap = overlap or shape
    cond = cond or (lambda x, y: True)

    X, Y = [], []

    h_stop = False
    for h_i in range(0, nh_arr, h_gap):
        w_stop = False
        for w_i in range(0, nw_arr, w_gap):
            w_f = w_i + nw_pat
            if w_f > nw_arr:
                w_f = nw_arr
                w_i = w_f - nw_pat
                if w_i < 0:
                    break
                w_stop = True

            h_f = h_i + nh_pat
            if h_f > nh_arr:
                h_f = nh_arr
                h_i = h_f - nh_pat
                if h_i < 0:
                    break
                h_stop = True

            x_patch = x[h_i:h_f, w_i:w_f]
            y_patch = y[h_i:h_f, w_i:w_f] if y is not None else y
            if cond(x_patch, y_patch):
                X.append(x_patch)
                Y.append(y_patch)

            if w_stop:
                break
        if h_stop:
            break

    if y is not None:
        return np.stack(X), np.stack(Y)
    else:
        return np.stack(X)


def gen_default_patch_cond(border=0.0):
    def default_patch_cond(x_patch, y_patch):
        n_border = int(border * np.mean(x_patch.shape))
        y_inside = y_patch[n_border:x_patch.shape[0] - n_border,
                           n_border:x_patch.shape[1] - n_border]
        return np.any(y_inside > -1)

    return default_patch_cond


def patch_dataset(X, Y, shape, overlap, x_as_function=False, cond=None):
    X_patched, Y_patched = [], []

    for x, y in zip(X, Y):
        if x_as_function:
            x = x()
        x_patches, y_patches = make_patches(x=x,
                                            y=y,
                                            shape=shape,
                                            overlap=overlap,
                                            cond=cond)
        X_patched.append(x_patches)
        Y_patched.append(y_patches)
    X_patched = np.concatenate(X_patched)
    Y_patched = np.concatenate(Y_patched)
    return X_patched, Y_patched


if __name__ == '__main__':
    pass
