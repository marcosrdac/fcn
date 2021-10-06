from functools import partial
import jax
from jax import random, jit, numpy as jnp, value_and_grad
from jax.flatten_util import ravel_pytree
import numpy as np
try:
    from .fcn import BatchFCN
    from .metrics import eval_metrics
except ImportError:
    from fcn import BatchFCN
    from metrics import eval_metrics


def build_batch_fcn(*args,
                    X_init,
                    init_rngkey=0,
                    drop_init_rngkey=1,
                    drop_apply_rngkey=2,
                    **kwargs):
    '''Wrapper for easily creating an FCN.'''
    model = BatchFCN(*args, **kwargs)

    # rngs initialization
    init_rngs = {'params': random.PRNGKey(init_rngkey)}
    apply_rngs = {}
    if model.use_drop():
        init_rngs['dropout'] = random.PRNGKey(drop_init_rngkey)
        apply_rngs['dropout'] = random.PRNGKey(drop_apply_rngkey)

    # parameter initialization
    variables = model.init(init_rngs, X_init)

    # mutable variables setup
    mutable = []
    if 'batch_stats' in variables:
        mutable.append('batch_stats')

    # getting model prediction function
    predict = partial(model.apply, rngs=apply_rngs, mutable=mutable)

    return variables, model, predict


@jit
def add_pytrees(tree_a, tree_b):
    '''Adds leaves of two pytrees.'''
    tree_a_flat, unravel = ravel_pytree(tree_a)
    tree_b_flat, unravel = ravel_pytree(tree_b)
    return unravel(tree_a_flat + tree_b_flat)


@jit
def multiply_pytree(tree, coefficient):
    '''Multiplies leaves of a pytree by a coefficient.'''
    tree_flat, unravel = ravel_pytree(tree)
    return unravel(coefficient * tree_flat)


def make_calc_grad(predict, loss_function, nclasses, keep_labels):
    '''Makes a gradient calculator.'''
    @jit
    def calc_grad(variables, X, Y, Y_masks, Y_joined_masks):
        '''Train model with batched data.'''
        def loss_fun(variables):
            '''Loss function for weights optimization.'''
            Ŷ_logits, *mutated_vars = predict(variables, X, proba=True)
            loss = loss_function(Ŷ_logits, Y, nclasses, Y_joined_masks)
            return loss, (Ŷ_logits, *mutated_vars)

        lossgrad_fun = value_and_grad(loss_fun, has_aux=True)
        (loss, (Ŷ_logits, *mutated_vars)), lossgrad = lossgrad_fun(variables)

        Ŷ = jnp.argmax(Ŷ_logits, -1)[..., None]
        metrics = eval_metrics(Ŷ, Y, Y_masks, keep_labels, loss=loss)

        return lossgrad, metrics

    return calc_grad


def make_train_epoch(calc_grad, nclasses):
    '''Makes an epoch trainer for the FCN model.'''
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
            Y_masks = [(Y_batch == c).nonzero() for c in range(nclasses)]
            Y_joined_masks = (*[jnp.concatenate(Y_c)
                                for Y_c in zip(*Y_masks)], )

            # add ntrue here!!!
            lossgrad, metrics = calc_grad(optimizer.target, X_batch, Y_batch,
                                          Y_masks, Y_joined_masks)
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
            metric: np.mean(
                [jax.device_get(metrics[metric]) for metrics in batch_metrics])
            for metric in batch_metrics[0]
        }
        return optimizer, epoch_metrics

    return train_epoch


def make_eval_epoch(predict, loss_function, nclasses, keep_labels):
    '''Creates a metric evaluator for an epoch.'''
    @jit
    def eval_epoch(variables, X, Y, Y_masks, Y_joined_masks):
        '''Evaluates an epoch.'''
        Ŷ_logits, *_ = predict(variables, X, proba=True)
        loss = loss_function(Ŷ_logits, Y, nclasses, Y_joined_masks)

        Ŷ = jnp.argmax(Ŷ_logits, -1)
        metrics = eval_metrics(Ŷ, Y, Y_masks, keep_labels, loss=loss)
        return Ŷ, metrics

    return eval_epoch
