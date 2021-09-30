#!/usr/bin/env python3

from os import environ
from functools import partial
from time import time
import numpy as np
import jax
from jax import random, numpy as jnp
from jax import jit, value_and_grad
from jax.experimental import optimizers
from flax import optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model.train import build_batch_fcn
from model.train import make_calc_grad, make_train_epoch, make_eval_epoch
from model.metrics import xentropy_loss
from train_config import DATA_DIRS, CLASSES, UNET_CONFIG, TRAIN_CONFIG
from utils.labelutils import gen_dataset
from utils.datautils import open_all_patched
from utils.plotutils import show_data
from utils.abcutils import AccumulatingDict
import pickle

environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
train_rng = random.PRNGKey(UNET_CONFIG['rngkeys']['train'])

# getting data
nclasses = len(CLASSES)
X, Y = open_all_patched(DATA_DIRS['X'], DATA_DIRS['Y'])
# X, Y = np.concatenate(X), np.concatenate(Y)
X, Y = np.concatenate([*X, *X]), np.concatenate([*Y, *Y])
until = 30
X, Y = X[:until], Y[:until]

# hold-out validation setup
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3)

for arch_name, fcn_params in UNET_CONFIG['architectures'].items():
    variables, model, predict = build_batch_fcn(
        **fcn_params,
        mode='classifier',
        nout=nclasses,
        X_init=X[:1],
        init_rngkey=UNET_CONFIG['rngkeys']['init'],
        drop_init_rngkey=UNET_CONFIG['rngkeys']['drop_init'],
        drop_apply_rngkey=UNET_CONFIG['rngkeys']['drop_apply'],
    )
    calc_grad = make_calc_grad(predict, xentropy_loss, nclasses)
    eval_epoch = make_eval_epoch(predict, xentropy_loss, nclasses)
    train_epoch = make_train_epoch(calc_grad, nclasses)

    # setting histories up
    history = {'train': {}, 'test': {}}

    # setting initial weights for every test
    print(jax.tree_map(jnp.shape, variables))
    init_variables = variables

    # training
    epochs = 50
    # learning_rates = 10**np.linspace(-1, 2, 8)
    # learning_rates = 1e-3, 1e-2, 1e-1,
    learning_rates = 2e-2,
    batch_size = 20

    for learning_rate in learning_rates:
        # defining optimizer
        optimizer_method = optim.Adam(learning_rate=learning_rate)
        variables = {'params': init_variables['params']}
        optimizer = optimizer_method.create(variables)

        # tracking train and test metrics
        train_history = []
        test_history = []

        # training loop
        start = time()
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
                Ŷ_test, test_metrics = eval_epoch(
                    variables,
                    X_test_cut,
                    Y_test_cut,
                    Y_masks,
                    Y_joined_masks,
                    keep_labels=TRAIN_CONFIG['keep_labels'])

                train_history.append(train_metrics)
                test_history.append(test_metrics)

                # print to screen
                print(f'{epoch}', end=' ')
                print(f"train loss: {train_metrics['loss']:.1e}",
                      f"test loss: {test_metrics['loss']:.1e}",
                      f"acc: {test_metrics['accuracy']:.0%}",
                      sep=' | ',
                      end='\n')

            except KeyboardInterrupt:
                break

        end = time()
        interval = end - start
        print(f'Elapsed time: {interval:.2f}', end=' ')
        print(f'(per epoch: {interval/epochs:.2f})')

        history['train'][learning_rate] = train_history
        history['test'][learning_rate] = test_history

        print(jax.tree_map(jnp.shape, variables))

        if len(learning_rates) == 1:
            Ŷ_train, *mutated_vars = predict(variables, X_train)
            show_data(X_train,
                      Y_train,
                      Ŷ_train,
                      xmax=1,
                      ymin=-1,
                      ymax=nclasses)
            Ŷ_test, *mutated_vars = predict(variables, X_test)
            show_data(X_test, Y_test, Ŷ_test, xmax=1, ymin=-1, ymax=nclasses)

    # plotting results
    plt.style.use('dark_background')
    metrics = [*history['train'][learning_rates[0]][0]]
    if 'loss' in metrics:
        metrics.remove('loss')
        metrics = ['loss'] + metrics
    nmetrics = len(metrics)
    fig, axes = plt.subplots(nmetrics, 2, sharex=True, sharey='row')

    for part in ('train', 'test'):
        col = 0 if part == 'train' else 1
        axes[0, col].set_title(part.capitalize())
        for lr, hist in history[part].items():
            hist_metrics = {m: [t[m] for t in hist] for m in metrics}

            for row, (m, vals) in enumerate(hist_metrics.items()):
                ax = axes[row, col]
                if m == 'loss':
                    ax.set_yscale('log')

                x = np.arange(1, 1 + len(vals))
                ax.plot(x, vals, label=f'lr={lr}')
                ax.set_ylabel(m.capitalize())

    for ax in axes.ravel():
        ax.legend()
        ax.grid(alpha=.2)
        ax.set_xlabel('Epoch')
    plt.show()
