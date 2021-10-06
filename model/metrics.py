#!/usr/bin/env python3

import numpy as np
from jax import numpy as jnp, lax


def onehot(labels, nclasses):
    '''One-hot encoder. Example transform: [2] -> [0, 0, 1].'''
    classes = jnp.arange(nclasses)
    logits = labels[..., None] == classes[None, None, None, :]
    return logits.astype(jnp.float32)


def xentropy(logits, labels):
    '''Cross-entropy between logits and labels.'''
    return -jnp.sum(logits * labels, axis=-1)


def xentropy_loss(ŷ_logits, y, nclasses, idx=None):
    '''
    Cross-entropy loss function based on logits, labels and a number of
    classes.
    '''
    labels = onehot(y, nclasses)
    return jnp.mean(xentropy(ŷ_logits[idx], labels[idx]))


def get_conf_matrix(ŷ, y, y_masks):
    '''
    Takes predicted and true labels and compares them to produce a confusion
    matrix. `y_masks` is a list of `nclasses` masks, one for each label used,
    where masked values of element `i` in such are pixels of belonging to class
    `i`.
    '''
    nclasses = len(y_masks)
    conf_matrix = jnp.zeros((nclasses, nclasses), dtype=jnp.int32)
    for true_label in range(nclasses):
        mask = y_masks[true_label]
        for pred_label in range(nclasses):
            n_pred = (ŷ[mask] == pred_label).sum()
            conf_matrix = conf_matrix.at[true_label, pred_label].set(n_pred)
    return conf_matrix


def true_positives(conf_matrix, label):
    '''Calculates the number of true positives in the confusion matrix.'''
    tp = conf_matrix[label, label]
    return tp


def false_positives(conf_matrix, label):
    '''Calculates the number of false positives in the confusion matrix.'''
    def accum_false_positives(true_label, s):
        return s + lax.cond(
            true_label != label,
            lambda _: conf_matrix[true_label, label],
            lambda _: 0,
            None,
        )

    fp = lax.fori_loop(0, conf_matrix.shape[0], accum_false_positives, 0)
    return fp


def false_negatives(conf_matrix, label):
    '''Calculates the number of false negatives in the confusion matrix.'''
    def accum_false_negatives(pred_label, s):
        return s + lax.cond(
            label != pred_label,
            lambda _: conf_matrix[label, pred_label],
            lambda _: 0,
            None,
        )

    fn = lax.fori_loop(0, conf_matrix.shape[0], accum_false_negatives, 0)
    return fn


def get_num_cases(conf_matrix, label):
    '''
    Derives the number of the four cases (true/false positives, true/false
    negatives)from a confusion matrix
    '''
    total = conf_matrix.sum()
    tp = true_positives(conf_matrix, label)
    fp = false_positives(conf_matrix, label)
    fn = false_negatives(conf_matrix, label)
    tn = (total - tp - fp - fn)
    return {'total': total, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def accuracy(cases):
    '''Accuracy metric calculated from a dict with number of cases.'''
    return (cases['tp'] + cases['tn']) / cases['total']


def precision(cases):
    '''Precision metric calculated from a dict with number of cases.'''
    return cases['tp'] / (cases['tp'] + cases['fp'])


def recall(cases):
    '''Recall metric calculated from a dict with number of cases.'''
    return cases['tp'] / (cases['tp'] + cases['fn'])


def f1_score(cases):
    '''F1-score metric calculated from a dict with number of cases.'''
    p = precision(cases)
    r = recall(cases)
    return 2 * p * r / (p + r)


def get_main_metrics_for_label(conf_matrix, label):
    '''Calculates main metrics for a label from a confusion matrix.'''
    cases = get_num_cases(conf_matrix, label)
    metrics = {}
    for metric in (accuracy, precision, recall, f1_score):
        m = metric.__name__
        metrics[m] = metric(cases)
    return metrics


def get_main_metrics(conf_matrix, keep_labels):
    '''
    Calculates main metrics for all labels from a confusion matrix, then
    compute averages metrics as default output. `keep_labels` tells which
    specific labels are also meant to have its metrics kept.
    '''
    metrics = {}
    nclasses = conf_matrix.shape[0]
    for label in range(nclasses):
        label_metrics = get_main_metrics_for_label(conf_matrix, label)
        for m, val in label_metrics.items():
            metrics[m] = val / nclasses + metrics.get(m, 0.)
            if keep_labels[label]:
                label_m = f"{m}_{label}"
                metrics[label_m] = val
    return metrics


def eval_metrics(Ŷ, Y, Y_masks, keep_labels, **other_metrics):
    '''
    Evaluate metrics of a model.
    '''
    conf_matrix = get_conf_matrix(Ŷ, Y, Y_masks)
    metrics = get_main_metrics(conf_matrix, keep_labels)
    return {**other_metrics, **metrics}


def overall_accuracy(ŷ, y):
    '''
    Overall accuracy metric. This is highly affected by dataset unbalance, as
    oposed to the mean accuracy. They are equal if only two classes are used.
    '''
    return jnp.mean(ŷ == y)


if __name__ == '__main__':
    mask_dtype = jnp.int32  # actually get from data
    nclasses = 3  # 0, 1, 2

    y = np.zeros((3, 4), dtype=mask_dtype)
    ŷ = np.zeros((3, 4), dtype=mask_dtype)

    y[0, :] = 1
    ŷ[0, ::2] = 1
    y[1, :] = 2
    ŷ[1, ::3] = 2
    y[2, ::3] = 2
    ŷ[2, ::3] = 2
    y[:, 2] = -1

    print(y)
    print(ŷ)

    y, ŷ = jnp.asarray(y), jnp.asarray(ŷ)
    y_masks = [(y == c).nonzero() for c in range(nclasses)]
    y_joined_masks = tuple(jnp.concatenate(i) for i in zip(*y_masks))

    # print(y_masks)
    # print(len(y_masks))
    # print(len(y_masks[0]))
    print(y.at[y_joined_masks].set(9))
    # print(y_joined_masks)

    conf_matrix = get_conf_matrix(ŷ, y, y_masks)
    # keep_labels = [False, False, True]
    keep_labels = [i == c for c in range(nclasses) for i in [1]]
    metrics = eval_metrics(ŷ, y, y_masks, keep_labels=keep_labels, loss=3)

    print(conf_matrix)
    print(metrics)
