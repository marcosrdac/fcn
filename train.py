#!/usr/bin/env python3

from os import environ, makedirs
from datetime import datetime
from time import time
import pickle
import numpy as np
import jax
from jax import random, numpy as jnp
from flax import optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model.train import build_batch_fcn
from model.train import make_calc_grad, make_train_epoch, make_eval_epoch
from model.metrics import xentropy_loss
from train_config import DATA_DIRS, CLASSES, UNET_CONFIG, TRAIN_CONFIG
from utils.datautils import open_all_patched
from utils.plotutils import plot_patches
from utils.abcutils import AccumulatingDict
from utils.pretty import pprint

NOW = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
plt.style.use('dark_background')
train_rng = random.PRNGKey(TRAIN_CONFIG['rngkeys']['train'])

# Getting data
until = 60  # change to None
nclasses = len(CLASSES)
X, Y = open_all_patched(DATA_DIRS['X'], DATA_DIRS['Y'])
X, Y = np.concatenate(X), np.concatenate(Y)
X, Y = X[:until], Y[:until]

# Hold-out validation setup
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=TRAIN_CONFIG['test_size'],
)

keep_labels = np.zeros(len(CLASSES), dtype=bool)
keep_labels[TRAIN_CONFIG['keep_label_info']] = True

# Training U-net architectures
for model_name, fcn_params in UNET_CONFIG['models'].items():

    result_dirs = {**DATA_DIRS['result_dirs'](NOW, model_name)}
    for dir_nick, dir_path in result_dirs:
        makedirs(dir_path)

    init_variables, model, predict = build_batch_fcn(
        **fcn_params,
        mode='classifier',
        nout=nclasses,
        X_init=X[:1],
        init_rngkey=UNET_CONFIG['rngkeys']['init'],
        drop_init_rngkey=UNET_CONFIG['rngkeys']['drop_init'],
        drop_apply_rngkey=UNET_CONFIG['rngkeys']['drop_apply'],
    )
    calc_grad = make_calc_grad(predict, xentropy_loss, nclasses, keep_labels)
    eval_epoch = make_eval_epoch(predict, xentropy_loss, nclasses, keep_labels)
    train_epoch = make_train_epoch(calc_grad, nclasses)

    # printing net variables
    print(jax.tree_map(jnp.shape, init_variables))

    # setting up histories
    histories = {}

    # train options
    epochs = TRAIN_CONFIG['max_epochs']
    learning_rates = TRAIN_CONFIG['learning_rates']
    batch_size = TRAIN_CONFIG['batch_size']

    for learning_rate in learning_rates:
        # tracking train and test metrics
        histories[learning_rate] = {
            'train': AccumulatingDict(),
            'test': AccumulatingDict(),
        }

        # defining optimizer
        optimizer_method = optim.Adam(learning_rate=learning_rate)
        variables = {'params': init_variables['params']}
        optimizer = optimizer_method.create(variables)

        # training loop
        time_train = [time()]
        for epoch in range(1, epochs + 1):
            try:
                train_rng, epoch_rng = random.split(train_rng)
                optimizer, train_metrics = train_epoch(optimizer,
                                                       X_train,
                                                       Y_train,
                                                       epoch_rng,
                                                       batch_size,
                                                       accum_grads=True)

                # update params
                variables = optimizer.target

                # eval step
                # TODO IS CUT EVEN NEEDED??
                X_test_cut = X_test[:batch_size]
                Y_test_cut = Y_test[:batch_size]
                Y_masks = [(Y_test_cut == c).nonzero()
                           for c in range(nclasses)]
                Y_joined_masks = (
                    *[jnp.concatenate(Y_c) for Y_c in zip(*Y_masks)], )

                Ŷ_test, test_metrics = eval_epoch(variables, X_test_cut,
                                                  Y_test_cut, Y_masks,
                                                  Y_joined_masks)

                histories[learning_rate]['train'].append(train_metrics)
                histories[learning_rate]['test'].append(test_metrics)

                # print to screen
                print(f'{epoch}', end=' ')
                print(f"train loss: {train_metrics['loss']:.1e}",
                      f"test loss: {test_metrics['loss']:.1e}",
                      f"acc: {test_metrics['accuracy']:.0%}",
                      sep=' | ',
                      end='\n')

            except KeyboardInterrupt:
                break

        time_train.append(time())
        interval = time_train.pop() - time_train.pop()
        print(f'Elapsed time: {interval:.2f}', end=' ')
        print(f'(per epoch: {interval/epoch:.2f})')

        # Plotting results

        Ŷ_train, *mutated_vars = predict(variables, X_train)
        plot_patches(X=X_train,
                     Y=Y_train,
                     Ŷ=Ŷ_train,
                     ymin=-1,
                     ymax=nclasses,
                     classes=CLASSES)
        Ŷ_test, *mutated_vars = predict(variables, X_test)
        plot_patches(X=X_test,
                     Y=Y_test,
                     Ŷ=Ŷ_test,
                     ymin=-1,
                     ymax=nclasses,
                     classes=CLASSES)

    # Plot histories
    metrics = [*histories[learning_rates[0]]['train']]
    if 'loss' in metrics:
        metrics.remove('loss')
        metrics = ['loss', *metrics]
    nmetrics = len(metrics)

    fig, axes = plt.subplots(nmetrics, 2, sharex=True, sharey='row')

    # TODO save plots
    # TODO see if we are using metrics calculated at each gradient step... else optimize it!
    # TODO plot only chosen variables for specific classes (CONFIG)
    for part in ('train', 'test'):
        col = 0 if part == 'train' else 1
        axes[0, col].set_title(part.capitalize())
        for lr, hists in histories.items():
            hist = hists[part]

            for row, m in enumerate(metrics):
                vals = hist[m]

                ax = axes[row, col]
                e = np.arange(1, 1 + len(vals))
                ax.plot(e, vals, label=f'lr={lr}')
                ax.set_ylabel(m.capitalize())

                if m == 'loss':
                    ax.set_yscale('log')

    for ax in axes.ravel():
        ax.legend()
        ax.grid(alpha=.2)
        ax.set_xlabel('Epoch')
    plt.show()
