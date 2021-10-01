import numpy as np
import matplotlib.pyplot as plt


def plot_patches(rows=6,
                 cols=None,
                 X=None,
                 Y=None,
                 Ŷ=None,
                 xmin=None,
                 xmax=None,
                 ymin=None,
                 ymax=None,
                 ŷmin=None,
                 ŷmax=None,
                 xname='Input',
                 yname='Ground\ntruth',
                 ŷname='Predicted',
                 xcmap='gray',
                 ycmap='nipy_spectral',
                 title=None,
                 show=False,
                 figsize=(8, 5),
                 figname=None):
    data = {}
    if X is not None:
        data['X'] = X
    if Y is not None:
        data['Y'] = Y
    if Ŷ is not None:
        data['Ŷ'] = Ŷ

    ymin = ymin if ymin else (Y.min() if Y is not None else ymin)
    ymax = ymax if ymax else (Y.max() if Y is not None else ymax)

    lims = {
        'X': {
            'min': xmin,
            'max': xmax
        },
        'Y': {
            'min': ymin,
            'max': ymax
        },
        'Ŷ': {
            'min': ŷmin or ymin,
            'max': ŷmax or ymax,
        }
    }
    names = {'X': xname, 'Y': yname, 'Ŷ': ŷname}
    cmaps = {'X': xcmap, 'Y': ycmap, 'Ŷ': ycmap}

    mini_cols = len(data)
    cols = cols or (6 // mini_cols)
    total_cols = mini_cols * cols  # divisible by 2 and 3 # TODO turn into param

    fig, axes = plt.subplots(rows, total_cols, figsize=figsize)

    fig.suptitle(title)

    for big_col in range(cols):
        for col, kind in enumerate(data):
            name = names[kind]
            axes[-1, big_col * mini_cols + col].set_xlabel(f'{name}')
        for row in range(rows):
            case = big_col * rows + row
            axes[row, big_col * mini_cols].set_ylabel(f'Case {case + 1}')

    for i in range(rows * cols):
        big_col = i // (rows)
        row = i % rows
        for col, (kind, D) in enumerate(data.items()):
            ax = axes[row, big_col * mini_cols + col]
            if i < X.shape[0]:
                ax.imshow(D[i],
                          cmap=cmaps[kind],
                          vmin=lims[kind]['min'],
                          vmax=lims[kind]['max'])
            else:
                ax.remove()
            ax.set_xticks([])
            ax.set_yticks([])

    fig.subplots_adjust(top=.90,bottom=.08, left=.05, right=.95, hspace=.5)

    if figname:
        fig.savefig(figname)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':

    def get_digits_data(*masked):
        from sklearn import datasets
        X, labels = datasets.load_digits(return_X_y=True)
        sample_shape = (8, 8, 1)  # (h, w, ch)
        X = X.reshape(-1, *sample_shape)

        X_min, X_max = X.min(), X.max()
        X = (X - X_min) / (X_max - X_min)

        T = .45
        Y = 1 * (X > T)

        classes = {0: -1}
        for n in range(10):
            if n in masked:
                Y[labels == n] = 0
            else:
                Y[labels == n] *= len(classes)
                classes[len(classes)] = n

        # maybe better labels
        return X, Y, labels, classes

    X, Y, labels, classes = get_digits_data()
    until = None
    X, Y = X[:until], Y[:until]
    print(classes)
    # show_patches(X[..., 0], Y[..., 0], Y[..., 0], show=True)
    plot_patches(6,
                 3,
                 X=X[..., 0],
                 Y=Y[..., 0],
                 Ŷ=Y[..., 0],
                 show=True,
                 title='Results')
