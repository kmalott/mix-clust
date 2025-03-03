import numpy as np

import warnings

from .base_mixture import BaseMixture, _check_shape

###############################################################################
# GPC mixture shape checkers used by the GPCMixture class

def _check_weights(weights, n_components):
    """Check the user provided 'weights'.

    Parameters
    ----------
    weights : array-like of shape (n_components,)
        The proportions of components of each mixture.

    n_components : int
        Number of components.

    Returns
    -------
    weights : array, shape (n_components,)
    """
    # TODO: implemeent check_array
    # weights = check_array(weights, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(weights, (n_components,), "weights")

    # check range
    if any(np.less(weights, 0.0)) or any(np.greater(weights, 1.0)):
        raise ValueError(
            "The parameter 'weights' should be in the range "
            "[0, 1], but got max value %.5f, min value %.5f"
            % (np.min(weights), np.max(weights))
        )

    # check normalization
    if not np.allclose(np.abs(1.0 - np.sum(weights)), 0.0):
        raise ValueError(
            "The parameter 'weights' should be normalized, but got sum(weights) = %.5f"
            % np.sum(weights)
        )
    return weights


def _check_means(means, n_components, n_features):
    """Validate the provided 'means'.

    Parameters
    ----------
    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    n_components : int
        Number of components.

    n_features : int
        Number of features.

    Returns
    -------
    means : array, (n_components, n_features)
    """
    # TODO: implemeent check_array
    # means = check_array(means, dtype=[np.float64, np.float32], ensure_2d=False)
    _check_shape(means, (n_components, n_features), "means")
    return means

###############################################################################
# GPC mixture parameters estimators (used by the M-Step)

def _estimate_gpc_parameters(X, resp, reg_covar, covariance_type):
    """Estimate the Gaussian distribution parameters.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.

    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.

    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.

    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components.

    means : array-like of shape (n_components, n_features)
        The centers of the current components.

    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    # covariances = _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar)
    covariances = ...
    return nk, means, covariances

def _estimate_log_gaussian_prob(X, means, covariances):
    """Estimate the log Gaussian probability.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)

    means : array-like of shape (n_components, n_features)

    covariances : array-like shape of (n_components, n_features, n_features)

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape

    log_prob = np.zeros((n_samples, n_components))
    for k in range(n_components):
        diff = X - means[k:]
        num = np.exp(0.5*(diff.T @ np.linalg.inv(covariances[k,:,:]) @ diff))
        denom = np.sqrt((2*np.pi)**(n_features) * np.linalg.det(covariances[k,:,:]))
        log_prob[:,k] = np.log(num / denom)

    return log_prob