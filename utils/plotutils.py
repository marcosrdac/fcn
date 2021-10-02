import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


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
                 xcmap=None,
                 ycmap=None,
                 classes=None,
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

    ymin = ymin if ymin is not None else (Y.min() if Y is not None else ymin)
    ymax = ymax if ymax is not None else (Y.max() if Y is not None else ymax)

    xcmap = xcmap or 'gray'
    if not ycmap:
        if classes:
            created_cmap = True
            unlabeled_color = ['black'] if ymin < 0 else []
            class_colors = [*unlabeled_color, *classes.colors.values()]
            ycmap = mpl.colors.LinearSegmentedColormap.from_list(
                'classes',
                class_colors,
                N=len(class_colors),
            )
            ycmap_colors = ycmap(np.linspace(0, 1, len(class_colors)))
            max_label = max(classes.main_names)
            nclasses = len(classes)
            print(classes.main_names)
            nlabels = max_label + 1 - ymin
            print(max_label)
            print(ymin, 'ymin')
            print(nlabels)
            print(nclasses)
            bounds = [ymin - .5, *range(max_label + 1), max_label + .5]
            ncolors = len(class_colors)
            # colors = class_cmap(np.linspace(0, 1, 20))
            # plt.imshow(colors[None, ...])
            # plt.show()
        else:
            created_cmap = False
            ycmap = 'nippy_spectral'

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
    total_cols = mini_cols * cols

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
                im = ax.imshow(D[i],
                               cmap=cmaps[kind],
                               vmin=lims[kind]['min'],
                               vmax=lims[kind]['max'])
            else:
                ax.remove()
            ax.set_xticks([])
            ax.set_yticks([])

    fig.subplots_adjust(top=.90, bottom=.2, left=.05, right=.95, hspace=.5)

    cbar_ax = fig.add_axes([.05, .05, .9, .05])
    cbar_ax.imshow(ycmap_colors[None, :], aspect='auto')
    cbar_ax.set_yticks([])
    cbar_ax.set_xticks([*range(0, nlabels)])
    unlabeled = ['Unlabeled'] if nlabels == nclasses + 1 else []
    cbar_ax.set_xticklabels([
        *unlabeled,
        *[classes.descriptions.get(i, '') for i in range(nclasses)]
    ])

    # cbar_ax.set_xlim(-1.5,nclasses+.5)
    # cbar_norm = mpl.colors.BoundaryNorm(bounds, ncolors)
    # cbar = plt.colorbar(im,
    # cax=cbar_ax,
    # orientation='horizontal',
    # norm=cbar_norm)
    # ticklabels = ['Unlabeled', *classes.descriptions.values()]
    # cbar.set_ticks([-1.5, *range(-1,len(ticklabels)+1), len(ticklabels)+.5])
    # cbar.set_ticklabels(ticklabels)

    if figname:
        fig.savefig(figname)
    if show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    from labelutils import Classes

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
    Y -= 1
    until = None
    X, Y = X[:until], Y[:until]
    Ŷ = Y.copy()
    Ŷ[Ŷ == -1] = 0
    print(classes)
    # show_patches(X[..., 0], Y[..., 0], Y[..., 0], show=True)

    classes = Classes()
    for d in range(10):
        classes.create(f'{d}th', [f'{1}'])

    nclasses = len(classes)
    class_colors = [*classes.colors.values()]
    class_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'classes', class_colors, N=len(class_colors))

    #colors = class_cmap(np.linspace(0, 1, 20))
    #plt.imshow(colors[None, ...])
    # plt.show()

    plot_patches(6,
                 3,
                 X=X[..., 0],
                 Y=Y[..., 0],
                 Ŷ=Ŷ[..., 0],
                 show=True,
                 title='Results',
                 classes=classes,
                 ymin=-1)
