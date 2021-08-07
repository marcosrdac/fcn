#!/usr/bin/env python3

from typing import Any, Iterable, Callable
import dataclasses
import jax
from jax import numpy as jnp, image as jim
from flax import linen as nn


def downscale(x, ratio=2):
    '''Scales input down by a ratio.'''
    x = nn.max_pool(x, window_shape=(ratio, ratio), strides=(ratio, ratio))
    return x


def upscale(x, shape=None, resizer='nearest'):
    '''Resizes input to a new shape.'''
    shape = (*shape[:2], x.shape[-1])
    x = jim.resize(x, shape, resizer)
    return x


class FCN(nn.Module):
    '''Fully convolutional neural network module.'''
    rescale: Iterable[int]
    nfeat: Any = 1
    activation: Callable = nn.relu
    kernsize: Any = 3
    norm: bool = False
    drop: float = 0.0
    droplast: float = 0.0
    final_kernsize: int = 1
    mode: str = None
    nout: int = None

    def use_drop(self):
        '''Checks if dropout is ever used in the network.'''
        use_drop = False
        if isinstance(self.drop, Iterable):
            if any(d != 0 for d in self.drop):
                use_drop = True
        elif self.drop:
            use_drop = True
        return use_drop

    def genpvtattr(self, attr, iterable_elements=False):
        '''Reformats module attributes.'''
        if isinstance(attr, Iterable):
            if iterable_elements:
                #  Iterable[Any] --> Iterable[Iterable[Any]]
                _attr = []
                for step, attr_i in enumerate(attr):
                    if isinstance(attr_i, Iterable):
                        _attr.append(attr_i)
                    else:
                        try:
                            _attr.append(
                                tuple((attr_i for lay in self._nfeats[step])))
                        except AttributeError:
                            # first run: generate self._nfeats for lazy people
                            nfeats = tuple(self.nfeat
                                           for step in self._rescales)
                            _attr = nfeats
                            break
            else:
                #  Iterable[Iterable[Any]] --> Iterable[Iterable[Any]]
                _attr = attr
        else:
            if iterable_elements:
                # Any --> Iterable[Iterable[Any]]
                try:
                    nfeats = self._nfeats

                    _attr = tuple(
                        tuple(attr for lay in nfeats)
                        for step, ratio in enumerate(self._rescales))

                except AttributeError:
                    # first run: generate self._nfeats for lazy people
                    nfeats = tuple((self.nfeat, ) for step in self._rescales)
                    _attr = nfeats

            else:
                _attr = tuple(attr for step in self._rescales)
        return tuple(_attr)

    def genpvtnout(self):
        '''
        Reformats nout attribute or generates default values based on FCN
        output kind.
        '''
        if self.nout:
            nout = self.nout
        elif self.mode:
            if self.mode.startswith('c'):  # classifier
                nout = 2
            else:
                nout = 1
        else:
            nout = 1
        return nout

    def setup(self):
        if self.use_drop():
            self.make_rng('dropout')
        self._rescales = self.rescale
        self._nfeats = self.genpvtattr(self.nfeat, True)
        self._kernsizes = self.genpvtattr(self.kernsize, True)
        self._activations = self.genpvtattr(self.activation, True)
        self._norms = self.genpvtattr(self.norm)
        drops = [*self.genpvtattr(self.drop)]
        if self.droplast:
            drops[-1] = self.droplast
        self._drops = tuple(drops)
        self._nout = self.genpvtnout()

    @nn.compact
    def __call__(self, x, proba=False, train=False):
        if x.ndim < 3:
            x = x[..., None]
        old = []
        for step, ratio in enumerate(self._rescales):
            if ratio > 0:
                # negative indexing instead of pop can generalize this model
                x_old = old.pop()
                x = upscale(x, x_old.shape)
                x = jnp.concatenate((x_old, x), axis=-1)
            if self._norms[step]:
                x = nn.BatchNorm(use_running_average=False)(x)
            if self._drops[step]:
                x = nn.Dropout(self._drops[step], deterministic=~train)(x)
            for lay, nfeat in enumerate(self._nfeats[step]):
                x = nn.Conv(features=nfeat,
                            kernel_size=(self._kernsizes[step][lay],
                                         self._kernsizes[step][lay]))(x)
                x = self._activations[step][lay](x)
            if ratio < 0:
                old.append(x)
                x = downscale(x, -ratio)

        # classifier
        if self.mode.startswith('c'):
            x = nn.Conv(features=self._nout,
                        kernel_size=(self.final_kernsize,
                                     self.final_kernsize))(x)
            x = nn.log_softmax(x, axis=-1)
            if not proba:
                x = jnp.argmax(x, axis=-1)

        # regressor
        elif self.mode.startswith('r'):
            x = nn.Conv(features=self._nout,
                        kernel_size=(self.final_kernsize,
                                     self.final_kernsize))(x)
        return x


class BatchFCN(FCN):
    '''Batched version of the fully convolutional neural network module.'''
    def setup(self):
        super().setup()
        variable_axes = {'params': None}
        split_rngs = {'params': False}
        if self.use_drop():
            variable_axes['dropout'] = None
            split_rngs['dropout'] = False
        if self.norm:
            variable_axes['batch_stats'] = 0

        Model = nn.vmap(
            self.__class__.__bases__[-1],
            in_axes=(0, None),
            variable_axes=variable_axes,
            split_rngs=split_rngs,
        )

        attr = dataclasses.asdict(self)
        self.model = Model(**attr)

    @nn.compact
    def __call__(self, X, proba=False):
        return self.model(X, proba)


if __name__ == '__main__':
    from jax import random
    from sklearn import datasets

    # getting data
    X, labels = datasets.load_digits(return_X_y=True)
    X = X.reshape(-1, 8, 8)

    # model definitions

    # nfeat = ((2, 3), (8, 8), (2, 3))
    # model = FCN(
    #     rescale=(-2, 0, 2),
    #     nfeat=nfeat,
    #     activation=[[nn.relu for lay in layers] for layers in nfeat],
    #     norm=[True for lay in nfeat],
    #     drop=[.2 for lay in nfeat],
    # )

    # nfeat = ((2, 3), (8, 8), (2, 3))
    # model = FCN(
    #     rescale=(-2, 0, 2),
    #     nfeat=nfeat,  # special
    #     activation=[[nn.relu for lay in layers]
    #                 for layers in nfeat],  # special bh
    #     norm=True,
    #     drop=.2,
    # )

    # example w-net definition
    model = BatchFCN(rescale=(-2, 0, 2, 0, -2, 0, 2),
                     nfeat=(8, ),
                     activation=nn.relu,
                     norm=True,
                     drop=.2,
                     mode='classifier')

    variables = model.init(
        {
            'params': random.PRNGKey(0),
            'dropout': random.PRNGKey(1),
        },
        X,
    )

    Ŷ, *mutated_vars = model.apply(variables,
                                   X,
                                   rngs={'dropout': random.PRNGKey(2)},
                                   mutable=['batch_stats'])

    print('Model parameters:')
    print(jax.tree_map(jnp.shape, variables))
    print('Input shape:', X.shape)
    print('Output shape:', Ŷ.shape)
