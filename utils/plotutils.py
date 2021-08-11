import numpy as np
import matplotlib.pyplot as plt


def show_data(
        X,
        Y,
        Ŷ=None,
        # idx=range(4*2),
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None):
    ymin = ymin or Y.min()
    ymax = ymax or Y.max()
    rows, cols = 6, 3

    data = [X, Y]
    val = [
        (xmin, xmax),
        (ymin, ymax),
    ]
    if Ŷ is not None:
        data.append(Ŷ)
        val.append((ymin, ymax))

    fig, axes = plt.subplots(rows, cols * len(data))

    for row in range(rows):
        for col in range(cols):
            for i, arr in enumerate(data):
                cmap = 'nipy_spectral' if i else 'Greys_r'
                axes[row, len(data) * col + i].imshow(arr[row + col * rows],
                                                      vmin=val[i][0],
                                                      vmax=val[i][1],
                                                      cmap=cmap)
    plt.show()
