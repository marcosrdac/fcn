#!/usr/bin/env python3

from os import environ, mkdir, makedirs
from os.path import join
from datetime import datetime
from time import time
import pickle
import numpy as np
import jax
from jax import random, numpy as jnp
from flax import optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from model.train import build_batch_fcn
from model.train import make_calc_grad, make_train_epoch, make_eval_epoch
from model.metrics import xentropy_loss
from train_config import DATA_DIRS, CLASSES, UNET_CONFIG, TRAIN_CONFIG
from utils.datautils import open_all_patched
from utils.plotutils import plot_patches
from utils.abcutils import AccumulatingDict
from utils.pretty import pprint
from utils.jaxutils import EarlyStopping
from flax.training.checkpoints import save_checkpoint, restore_checkpoint

NOW = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
plt.style.use('dark_background')
sns.set_context = 'paper'
train_rng = random.PRNGKey(TRAIN_CONFIG['rngkeys']['train'])

# Getting data
until = None  # change to None
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
    for dir_nick, dir_path in result_dirs.items():
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

    for lr in learning_rates:
        # tracking train and test metrics
        histories[lr] = {
            'train': AccumulatingDict(),
            'test': AccumulatingDict(),
        }

        # saving model
        checkpoints_dir = join(result_dirs['checkpoint'], f'lr={lr}')
        mkdir(checkpoints_dir)

        # defining optimizer
        optimizer_method = optim.Adam(learning_rate=lr)
        variables = {'params': init_variables['params']}
        optimizer = optimizer_method.create(variables)

        # defining early stopping strategy
        es_config = TRAIN_CONFIG['early_stopping']
        if es_config['enable']:
            stop_strategy = EarlyStopping(
                greater_is_better=es_config['greater_is_better'],
                patience=es_config['patience'],
                delta=es_config['delta'],
            )
        else:
            stop_strategy = None

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
                # TODO is cut even needed?? (i guess it isn't...)
                # I cannot remember why I wrote this workaround now, but my
                # code seems to support evaluating the whole test set now 
                #X_test_cut = X_test[:batch_size]
                #Y_test_cut = Y_test[:batch_size]

                X_test_cut = X_test
                Y_test_cut = Y_test

                Y_masks = [(Y_test_cut == c).nonzero()
                           for c in range(nclasses)]
                Y_joined_masks = (
                    *[jnp.concatenate(Y_c) for Y_c in zip(*Y_masks)], )

                Ŷ_test, test_metrics = eval_epoch(variables, X_test_cut,
                                                  Y_test_cut, Y_masks,
                                                  Y_joined_masks)

                histories[lr]['train'].append(train_metrics)
                histories[lr]['test'].append(test_metrics)

                # print to screen
                print(f'{epoch}', end=' ')
                print(f"train loss: {train_metrics['loss']:.1e}",
                      f"test loss: {test_metrics['loss']:.1e}",
                      f"acc: {test_metrics['accuracy']:.0%}",
                      sep=' | ',
                      end='\n')

                checkpoint_path = save_checkpoint(checkpoints_dir,
                                                  variables,
                                                  step=epoch,
                                                  prefix='',
                                                  keep=1000,
                                                  overwrite=False)
                if stop_strategy:
                    stop_strategy.update(test_metrics[es_config['metric']],
                                         epoch)
                    best_checkpoint = stop_strategy.best_checkpoint
                    if stop_strategy.stop:
                        break
                else:
                    best_checkpoint = epoch

            except KeyboardInterrupt:
                break

        variables = restore_checkpoint(checkpoints_dir,
                                       target=variables,
                                       step=best_checkpoint,
                                       prefix='',
                                       parallel=True)

        time_train.append(time())
        interval = time_train.pop() - time_train.pop()
        print(f'Elapsed time: {interval:.2f}', end=' ')
        print(f'(per epoch: {interval/epoch:.2f})')

        name = f'lr={lr}_e={epoch}'

        # Plotting results
        Ŷ_train, *mutated_vars = predict(variables, X_train)
        train_fig_path = join(result_dirs['patch_plot'], f'train_{name}.png')
        plot_patches(X=X_train,
                     Y=Y_train,
                     Ŷ=Ŷ_train,
                     ymin=-1,
                     ymax=nclasses,
                     classes=CLASSES,
                     figname=train_fig_path,
                     dpi=300)
        Ŷ_test, *mutated_vars = predict(variables, X_test)
        test_fig_path = join(result_dirs['patch_plot'], f'test_{name}.png')
        plot_patches(X=X_test,
                     Y=Y_test,
                     Ŷ=Ŷ_test,
                     ymin=-1,
                     ymax=nclasses,
                     classes=CLASSES,
                     figname=test_fig_path,
                     dpi=300)

    history_data_path = join(result_dirs['history_data'], 'history_data.pkl')
    with open(history_data_path, 'wb') as f:
        pickle.dump(histories, f)

    # Plot histories
    metrics = [*histories[learning_rates[0]]['train']]
    if 'loss' in metrics:
        metrics.remove('loss')
        metrics = ['loss', *metrics]

    # remove unwanted info from plots
    metrics = [m for m in TRAIN_CONFIG['metrics'] if m in metrics]

    fig, axes = plt.subplots(len(metrics),
                             2,
                             figsize=(6, 10),
                             dpi=300,
                             sharex=True,
                             sharey='row')

    fig.subplots_adjust(left=.12)

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

    history_fig_path = join(result_dirs['history_plot'], 'history.png')

    fig.savefig(history_fig_path)
