#!/usr/bin/env python3

from functools import partial
from time import time
import numpy as np
import jax
from jax import random, numpy as jnp
from jax import jit, value_and_grad
from jax.flatten_util import ravel_pytree
from flax import optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from fcn.fcn import BatchFCN
from config import INPUT_DIR, OUTPUT_DIR, IMAGE_DIR, LABEL_DIR, CLASSES, X_DIR, Y_DIR
from utils.labelutils import gen_dataset
from utils.datautils import open_all_patched
from utils.plotutils import show_data
from os import environ
import pickle
from jax.experimental import optimizers

environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

CLASS_NAMES = [c.name for c in CLASSES]


def xentropy(logits, labels):
    '''Cross-entropy between logits and labels.'''
    return -jnp.sum(logits * labels, axis=-1)


def onehot(labels, nclasses):
    '''One-hot encoder. Example transform: [2] -> [0, 0, 1].'''
    classes = jnp.arange(nclasses)
    logits = labels[..., None] == classes[None, None, None, :]
    return logits.astype(jnp.float32)


def xentropy_loss(ŷ_logits, y, nclasses, idx=None):
    '''
    Cross-entropy loss function based on logits, labels and a number of
    classes.
    '''
    labels = onehot(y, nclasses)
    return jnp.mean(xentropy(ŷ_logits[idx], labels[idx]))


def accuracy(ŷ, y):
    '''Accuracy metric.'''
    return jnp.mean(ŷ == y)


def eval_metrics(Ŷ, Y, idx=None, **other_metrics):
    '''
    Evaluate metrics of a model.
    '''
    return {
        **other_metrics,
        'accuracy': accuracy(Ŷ[idx], Y[idx]),
    }


# getting data
nclasses = len(CLASS_NAMES)
X, Y = open_all_patched(X_DIR, Y_DIR)
# X, Y = np.concatenate(X), np.concatenate(Y)
X, Y = np.concatenate([*X, *X]), np.concatenate([*Y, *Y])
# X, Y = X[:100], Y[:100]
# print(X.shape)
# X, Y = X[:30], Y[:30]
# print(X.shape)

# hold-out validation setup
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 3)

# initializing model
model = BatchFCN((-2, -2, 0, 2, 2),
                 nfeat=(
                     (8, ),
                     (16, ),
                     (32, ),
                     (16, ),
                     (8, ),
                 ),
                 mode='classifier',
                 norm=True,
                 droplast=.3,
                 nout=nclasses)

# rngs initialization
init_rngs = {'params': random.PRNGKey(0)}
apply_rngs = {}
if model.use_drop():
    init_rngs['dropout'] = random.PRNGKey(1)
    apply_rngs['dropout'] = random.PRNGKey(2)
train_rng = random.PRNGKey(3)

# parameter initialization
variables = model.init(init_rngs, X[:1])

# mutable variables setup
mutable = []
if 'batch_stats' in variables:
    mutable.append('batch_stats')

# getting model prediction function
predict = partial(model.apply, rngs=apply_rngs, mutable=mutable)


@jit
def calc_grad(variables, X, Y, masked_idx):
    '''Train model with batched data.'''
    def loss_fun(variables):
        '''Loss function for weights optimization.'''
        Ŷ_logits, *mutated_vars = predict(variables, X, proba=True)
        loss = xentropy_loss(Ŷ_logits, Y, nclasses, masked_idx)
        return loss, (Ŷ_logits, *mutated_vars)

    lossgrad_fun = value_and_grad(loss_fun, has_aux=True)
    (loss, (Ŷ_logits, *mutated_vars)), lossgrad = lossgrad_fun(variables)

    Ŷ = jnp.argmax(Ŷ_logits, -1)[..., None]
    metrics = eval_metrics(Ŷ, Y, masked_idx, loss=loss)

    return lossgrad, metrics


@jit
def add_pytrees(tree_a, tree_b):
    tree_a_flat, unravel = ravel_pytree(tree_a)
    tree_b_flat, unravel = ravel_pytree(tree_b)
    return unravel(tree_a_flat + tree_b_flat)


@jit
def multiply_pytree(tree, coeficient):
    tree_flat, unravel = ravel_pytree(tree)
    return unravel(coeficient * tree_flat)


def train_epoch(optimizer, X, Y, rng, batch_size=None, accum_grads=False):
    '''Trains model for an epoch.'''
    batch_size = batch_size or X.shape[0]
    epoch_steps = X.shape[0] // batch_size

    perms = random.permutation(rng, X.shape[0])
    perms = perms[:epoch_steps * batch_size]
    perms = perms.reshape((epoch_steps, batch_size))
    batch_metrics = []

    if accum_grads:
        lossgrad_mean, unravel_vars = ravel_pytree(optimizer.target)
        lossgrad_mean = lossgrad_mean.at[...].set(0)

    # maybe pmap over batches
    for perm in perms:
        X_batch, Y_batch = X[perm], Y[perm]
        masked_idx = jnp.where(Y_batch > -1)
        lossgrad, metrics = calc_grad(variables, X_batch, Y_batch, masked_idx)
        batch_metrics.append(metrics)

        if accum_grads:
            lossgrad_mean = add_pytrees(lossgrad_mean, lossgrad)
        else:
            optimizer = optimizer.apply_gradient(lossgrad)

    if accum_grads:
        lossgrad_mean = multiply_pytree(lossgrad_mean, 1 / perms.shape[0])
        optimizer = optimizer.apply_gradient(lossgrad_mean)

    # maybe pmap here
    epoch_metrics = {
        metric: np.mean([metrics[metric] for metrics in batch_metrics])
        for metric in batch_metrics[0]
    }
    return optimizer, epoch_metrics


@jit
def eval_epoch(variables, X, Y, idx):
    Ŷ_logits, *_ = predict(variables, X, proba=True)
    loss = xentropy_loss(Ŷ_logits, Y, nclasses, idx)

    Ŷ = jnp.argmax(Ŷ_logits, -1)
    return Ŷ, eval_metrics(Ŷ, Y, idx, loss=loss)


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
            masked_idx = jnp.where(Y_test > -1)
            Ŷ_test, test_metrics = eval_epoch(variables, X_test[:batch_size],
                                              Y_test[:batch_size], masked_idx)

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
        show_data(X_train, Y_train, Ŷ_train, xmax=1, ymin=-1, ymax=nclasses)
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
