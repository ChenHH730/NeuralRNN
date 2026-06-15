"""Population structure analysis for low-rank and similar RNN models.

Tools for extracting connectivity vectors, fitting Gaussian mixture models
on neuron feature spaces, and analyzing functional populations.  Ported from
Dubreuil et al. (2022) Nature Neuroscience.

Design: make_vecs() uses duck-typing on model attributes (m, n, wi, wo,
rank, input_size, output_size).  It works with any model that exposes these
— primarily LowrankRNNModel, but also SupportLowRankRNN and similar.
"""
from __future__ import annotations

import numpy as np


def make_vecs(net):
    """Extract connectivity vectors from a low-rank-like network.

    Returns a list of vectors characterising each neuron's connectivity
    profile.  For a rank-R network with K inputs and O outputs, returns
    2R + K + O vectors, each of length N (hidden_size):

        - R vectors from columns of ``m``  (recurrent output directions)
        - R vectors from columns of ``n``  (recurrent input directions)
        - K vectors from rows of ``wi``    (input weights)
        - O vectors from columns of ``wo`` (output weights)

    Args:
        net: A model with attributes ``m`` (N, R), ``n`` (N, R),
             ``wi`` (input_dim, N), ``wo`` (N, output_dim), ``rank``,
             ``input_size``, ``output_size``.  Works with
             ``LowrankRNNModel`` and duck-type-compatible objects.

    Returns:
        list of (N,) numpy arrays — one per connectivity vector
    """
    vecs = []
    for i in range(net.rank):
        vecs.append(net.m[:, i].detach().cpu().numpy())
    for i in range(net.rank):
        vecs.append(net.n[:, i].detach().cpu().numpy())
    for i in range(net.input_size):
        vecs.append(net.wi[i].detach().cpu().numpy())
    for i in range(net.output_size):
        vecs.append(net.wo[:, i].cpu().detach().numpy())
    return vecs


def gmm_fit(neurons_fs, n_components, algo='bayes', n_init=50,
            random_state=None):
    """Fit a Gaussian mixture model to neuron feature vectors.

    Used for identifying functional populations of neurons based on
    their connectivity profiles (e.g. m, n, wi, wo vectors).

    Args:
        neurons_fs: list of (N,) arrays or (d, N) array — feature matrix.
                    If list, stacked horizontally and transposed to (N, d).
        n_components: int — number of GMM components (populations)
        algo: ``'em'`` (GaussianMixture) or ``'bayes'``
              (BayesianGaussianMixture)
        n_init: int — number of initializations for EM
        random_state: int or None — seed for reproducibility

    Returns:
        labels: (N,) int numpy array — cluster assignments
        model: fitted sklearn mixture model
    """
    from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

    if isinstance(neurons_fs, list):
        X = np.vstack(neurons_fs).transpose()
    else:
        X = neurons_fs

    if algo == "em":
        model = GaussianMixture(n_components=n_components, n_init=n_init,
                                random_state=random_state)
    else:
        model = BayesianGaussianMixture(
            n_components=n_components, n_init=n_init,
            random_state=random_state, init_params='random')

    model.fit(X)
    z = model.predict(X)
    return z, model


def compute_population_means(X, labels):
    """Compute per-population mean feature vectors.

    Args:
        X: (N, d) numpy array — feature matrix
        labels: (N,) int array — cluster assignments (0 .. n_pops-1)

    Returns:
        (n_pops, d) numpy array — mean feature vector per population
    """
    n_pops = labels.max() + 1
    means = np.vstack([X[labels == i].mean(axis=0) for i in range(n_pops)])
    return means


def compute_population_covariances(X, labels):
    """Compute per-population covariance matrices.

    Args:
        X: (N, d) numpy array — feature matrix
        labels: (N,) int array — cluster assignments (0 .. n_pops-1)

    Returns:
        list of (d, d) numpy arrays — covariance matrix per population
    """
    n_pops = labels.max() + 1
    covs = [np.cov(X[labels == i].transpose()) for i in range(n_pops)]
    return covs
