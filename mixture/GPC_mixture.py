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

def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """Estimate the full guassian covariance matrices per group (w_g).

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff)
        covariances[k].flat[:: n_features + 1] += reg_covar
    return covariances

def _estimate_gaussian_covariances(resp, X, nk, means, reg_covar):
    """Estimate the full gaussian covariance matrices (w).

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff)
        covariances[k].flat[:: n_features + 1] += reg_covar
    covariances = covariances.sum(axis=0)
    return covariances

def _estimate_covariances(resp, X, nk, means, reg_covar, covariance_type):
    """Estimate the covariance matrices.

    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)

    X : array-like of shape (n_samples, n_features)

    nk : array-like of shape (n_components,)

    means : array-like of shape (n_components, n_features)

    reg_covar : float

    covariance_type : string 

    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    covariances = None
    W_g = _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar)
    if covariance_type == "VEE":
        covariances = _estimate_VEE_covariances(X, W_g, nk)
    elif covariance_type == "EVE":
        covariances = _estimate_EVE_covariances()
    elif covariance_type == "VVE":
        covariances = _estimate_VVE_covariances()
    elif covariance_type == "EEV":
        covariances = _estimate_EEV_covariances(X, W_g)
    elif covariance_type == "VEV":
        covariances = _estimate_VEV_covariances(X, W_g, nk)
    elif covariance_type == "EVV":
        covariances = _estimate_EVV_covariances(X, W_g)
    elif covariance_type == "VVV":
        covariances = _estimate_VVV_covariances(W_g, nk)
    elif covariance_type == "VEI":
        covariances = _estimate_VEI_covariances(W_g, nk)
    elif covariance_type == "EVI":
        covariances = _estimate_EVI_covariances(X, W_g)
    elif covariance_type == "VVI":
        covariances = _estimate_VVI_covariances(W_g, nk)
    elif covariance_type == "VII":
        covariances = _estimate_VII_covariances(W_g, nk)
    W = _estimate_gaussian_covariances(resp, X, nk, means, reg_covar)
    if covariance_type == "EEE":
        covariances = _estimate_EEE_covariances(X, W)
    elif covariance_type == "EEI":
        covariances = _estimate_EEI_covariances(X, W)
    elif covariance_type == "EII":
        covariances = _estimate_EII_covariances(X, W)

    # TODO: duplicate covariances along group dimension
    return covariances

def _estimate_EEE_covariances(X,W):
    # TODO: add description 
    n_samples, _ = X.shape
    covariances = W / n_samples
    return covariances

def _estimate_VEE_covariances(X, W_g, nk):
    n_samples, n_features = X.shape
    n_components = nk.shape
    # TODO: add initial estimate of C
    C = np.zeros((n_features, n_features))
    lambda_g = np.zeros((n_components))
    # TODO: add iterative loop
    for k in range(0, n_components):
        lambda_g[k] = np.linalg.trace(W_g[k,:,:] @ np.linalg.inv(C)) / (n_features*nk[k])
        num += W_g[k] / lambda_g[k]
    denom = np.linalg.det(num) ** (1/n_features)
    C = num / denom
    # After loop
    # covariances = np.zeros((n_components, n_features, n_features))
    covariances = lambda_g * np.tile(C, (n_components, 1, 1))
    return covariances

def _estimate_EVE_covariances():
    pass

def _estimate_VVE_covariances():
    pass

def _estimate_EEV_covariances(X, W_g):
    n_components, _, _ = W_g.shape
    n_samples, n_features = X.shape
    omega = np.zeros((n_components, n_features, n_features))
    L = np.zeros((n_components, n_features, n_features))
    covariances = np.zeros((n_components, n_features, n_features))
    omega_sum = np.zeros((n_features, n_features))
    for k in range(0, n_components):
        eigenval, L[k,:,:] = np.linalg.eigh(W_g[k,:,:])
        omega[k,:,:] = np.diag(eigenval)
        omega_sum += omega[k,:,:]
    A = omega_sum / np.linalg.det(omega_sum)**(1/n_features)
    Lambda = np.linalg.det(omega_sum)**(1/n_features) / n_samples
    for k in range(0, n_components):
        covariances[k,:,:] = Lambda * (L[k,:,:] @ A @ L[k,:,:].T)
    return covariances

def _estimate_VEV_covariances(X, W_g, nk):
    n_components, _, _ = W_g.shape
    n_samples, n_features = X.shape
    omega = np.zeros((n_components, n_features, n_features))
    L = np.zeros((n_components, n_features, n_features))
    covariances = np.zeros((n_components, n_features, n_features))
    omega_sum = np.zeros((n_features, n_features))
    lambda_g = np.zeros((n_components))
    # TODO: better initialization for A
    A = np.zeros((n_components, n_features, n_features))
    # TODO: create iterative loop
    for k in range(0, n_components):
        eigenval, L[k,:,:] = np.linalg.eigh(W_g[k,:,:])
        omega[k,:,:] = np.diag(eigenval)
        lambda_g[k] = np.trace(W_g[k,:,:] @ L[k,:,:] @ np.linalg.inv(A) @ L[k,:,:].T) / (n_features*nk[k])
        omega_sum += (1/lambda_g[k])*omega[k,:,:]
    A = omega_sum / np.linalg.det(omega_sum)**(1/n_features)
    # After loop
    for k in range(0, n_components):
        covariances[k,:,:] = lambda_g[k] * (L[k,:,:] @ A @ L[k,:,:].T)
    return covariances

def _estimate_EVV_covariances(X, W_g):
    n_samples, n_features = X.shape
    n_components, _, _ = W_g.shape
    C = np.zeros((n_components, n_features, n_features))
    covariances = np.zeros((n_components, n_features, n_features))
    Lambda = 0
    for k in range(0, n_components):
        C[k,:,:] = W_g[k,:,:] / np.linalg.det(W_g[k,:,:]) ** (1/n_features)
        Lambda += np.linalg.det(W_g[k,:,:]) ** (1/n_features)
    Lambda /= n_samples
    for k in range(0, n_components):
        covariances[k,:,:] = Lambda * C[k,:,:]
    return covariances

def _estimate_VVV_covariances(W_g, nk):
    n_components, _, _ = W_g.shape
    for k in range(0,n_components):
        W_g[k,:,:] = (1/nk[k])*W_g[k,:,:]
    return W_g

def _estimate_EEI_covariances(X, W):
    n_samples, n_features = X.shape
    W_diag = np.diag(np.linalg.diagonal(W))
    A = W_diag / (np.linalg.det(W_diag) ** (1/n_features))
    Lambda = (np.linalg.det(W_diag) ** (1/n_features)) / n_samples
    return Lambda * A

def _estimate_VEI_covariances(W_g, nk):
    n_components, n_features, _ = W_g.shape
    # TODO: initial estimate of A
    A = np.zeros((n_features, n_features))
    lambda_g = np.zeros((n_components))
    covariances = np.zeros((n_components, n_features, n_features))
    # TODO: make iterative loop
    W_sum = np.zeros((n_features, n_features))
    for k in range(0, n_components):
        lambda_g[k] = np.linalg.trace(W_g[k,:,:] @ np.linalg.inv(A)) / (n_features * nk[k])
        W_sum += (1/lambda_g[k]) * W_g[k,:,:]
    W_sum = np.diag(np.linalg.diagonal(W_sum))
    A = W_sum / np.linalg.det(W_sum) ** (1/n_features)
    # After loop
    for k in range(0, n_components):
        covariances[k,:,:] = lambda_g[k] * A
    return covariances

def _estimate_EVI_covariances(X, W_g):
    n_samples, _ = X.shape
    n_components, n_features, _ = W_g.shape
    A_g = np.zeros((n_components, n_features, n_features))
    W_sum = np.zeros((n_features, n_features))
    covariances = np.zeros((n_components, n_features, n_features))
    for k in range(0, n_components):
        W_diag = np.diag(np.linalg.diagonal(W_g[k,:,:]))
        denom = np.linalg.det(W_diag) ** (1/n_features)
        A_g[k,:,:] = W_diag / denom
        W_sum += denom
    Lambda = W_sum / n_samples
    for k in range(0, n_components):
        covariances[k,:,:] = Lambda * A_g[k,:,:]
    return covariances

def _estimate_VVI_covariances(W_g, nk):
    n_components, n_features, _ = W_g.shape
    covariances = np.zeros((n_components, n_features, n_features))
    for k in range(0, n_components):
        W_diag = np.diag(np.linalg.diagonal(W_g[k,:,:]))
        denom = np.linalg.det(W_diag) ** (1/n_features)
        covariances[k,:,:] = (denom / nk[k]) * (W_diag / denom)
    return covariances[k,:,:]

def _estimate_EII_covariances(X, W):
    n_samples, n_features = X.shape
    return np.diag(np.linalg.trace(W) / (n_samples*n_features))

def _estimate_VII_covariances(W_g, nk):
    n_components, n_features, _ = W_g.shape
    covariances = np.zeros((n_components, n_features, n_features))
    for k in range(0, n_components):
        covariances[k,:,:] = np.linalg.trace(W_g[k,:,:]) / (n_features*nk[k])
    return covariances

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

    covariance_type : string
        Type of regularization placed on the covariance matrix. 

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
    covariances = _estimate_covariances(resp, X, nk, means, reg_covar, covariance_type)
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

###############################################################################
# GPC Mixture Class

class GPCMixture(BaseMixture):
    def __init__(
        self,
        n_components=1,
        *,
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        covariance_type="VVV",
        weights_init=None,
        means_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        super().__init__(
            n_components=n_components,
            tol=tol,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )
        self.covariance_type = covariance_type,
        self.reg_covar=reg_covar,
        self.weights_init = weights_init
        self.means_init = means_init

    def _check_parameters(self, X):
        """Check the GPC mixture parameters are well defined."""
        _, n_features = X.shape

        if self.weights_init is not None:
            self.weights_init = _check_weights(self.weights_init, self.n_components)

        if self.means_init is not None:
            self.means_init = _check_means(
                self.means_init, self.n_components, n_features
            )

    def _initialize(self, X, resp):
        """Initialize the model parameters of the derived class.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = X.shape
        weights, means, covariances = None, None, None
        if resp is not None:
            weights, means, covariances = _estimate_gpc_parameters(
                X, resp, self.reg_covar, self.covariance_type
            )
            if self.weights_init is None:
                weights /= n_samples

        self.weights_ = weights if self.weights_init is None else self.weights_init
        self.means_ = means if self.means_init is None else self.means_init
        self.covariances_ = covariances

    def fit_predict(self, X, y=None):
        """Estimate model parameters using X and predict the labels for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        # TODO: implement validate data
        # X = validate_data(self, X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False
        
        # TODO: implement check random state
        # random_state = check_random_state(self.random_state)
        random_state = self.random_state

        n_samples, _ = X.shape
        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)

            lower_bound = -np.inf if do_init else self.lower_bound_

            if self.max_iter == 0:
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                converged = False
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound

                    log_prob_norm, log_resp = self._e_step(X)
                    self._m_step(X, log_resp)
                    lower_bound = log_prob_norm

                    change = lower_bound - prev_lower_bound
                    self._print_verbose_msg_iter_end(n_iter, change)

                    if abs(change) < self.tol:
                        converged = True
                        break

            self._print_verbose_msg_init_end(lower_bound, converged)

            if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                max_lower_bound = lower_bound
                best_params = self._get_parameters()
                best_n_iter = n_iter
                self.converged_ = converged
        
        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
        if not self.converged_ and self.max_iter > 0:
            warnings.warn(
                (
                    "Best performing initialization did not converge. "
                    "Try different init parameters, or increase max_iter, "
                    "tol, or check for degenerate data."
                ),
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X)

        return log_resp.argmax(axis=1)

    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        self.weights_, self.means_, self.covariances_ = _estimate_gpc_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= self.weights_.sum()

    def _get_parameters(self):
        return (
            self.weights_,
            self.means_,
            self.covariances_,
        )

    def _set_parameters(self, params):
        (
            self.weights_,
            self.means_,
            self.covariances_,
        ) = params

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.means_.shape
        cov_params = self.n_components * n_features * (n_features + 1) / 2.0
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

    def sample(self, n_samples=1):
        """Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample.

        y : array, shape (nsamples,)
            Component labels.
        """
        # TODO: implement check_is_fitted
        # check_is_fitted(self)

        if n_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample." % (self.n_components)
            )

        _, n_features = self.means_.shape
        # TODO: implement check_random_state
        # rng = check_random_state(self.random_state)
        rng = self.random_state
        n_samples_comp = rng.multinomial(n_samples, self.weights_)

        X = np.vstack(
            [
                rng.multivariate_normal(mean, covariance, int(sample))
                for (mean, covariance, sample) in zip(
                    self.means_, self.covariances_, n_samples_comp
                )
            ]
        )

        y = np.concatenate(
            [np.full(sample, j, dtype=int) for j, sample in enumerate(n_samples_comp)]
        )

        return (X, y)

    def _estimate_log_weights(self):
        """Estimate log-weights in EM algorithm, E[ log pi ] in VB algorithm.

        Returns
        -------
        log_weight : array, shape (n_components, )
        """
        return np.log(self.weights_)

    def _estimate_log_prob(self, X):
        """Estimate the log-probabilities log P(X | Z).

        Compute the log-probabilities per each component for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob : array, shape (n_samples, n_component)
        """
        return _estimate_log_gaussian_prob(
            X, self.means_, self.covariances_
        )

    def bic(self, X):
        """Bayesian information criterion for the current model on the input X.

        You can refer to this :ref:`mathematical section <aic_bic>` for more
        details regarding the formulation of the BIC used.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.

        Returns
        -------
        bic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + self._n_parameters() * np.log(
            X.shape[0]
        )

    def aic(self, X):
        """Akaike information criterion for the current model on the input X.

        You can refer to this :ref:`mathematical section <aic_bic>` for more
        details regarding the formulation of the AIC used.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)
            The input samples.

        Returns
        -------
        aic : float
            The lower the better.
        """
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()