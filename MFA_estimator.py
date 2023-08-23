import numpy as np
import numpy.typing as npt
from MFA_cplx import MFA_cplx

class MfaEstimator(MFA_cplx.MFA_cplx):
    def estimate(self,
                 y: npt.ArrayLike,
                 Cn: npt.NDArray,
                 A: npt.NDArray = None,
                 n_summands_or_proba: float = 1
    ):
        """
        Use the noise covariance matrix and the matrix A to update the
        covariance matrices of the Gaussian mixture model. This GMM is then
        used for channel estimation from y.

        Args:
            y: A 2D complex numpy array of shape (n_samples, n_dims) representing the observations.
            Cn: A 2D numpy array representing the noise covariance matrix.
            A: A 2D complex numpy array representing the observation matrix.
            n_summands_or_proba:
                If this is an integer, compute the sum of the top (highest
                component probabilities) n_components_or_probability LMMSE
                estimates.
                If this is a float, compute the sum of as many LMMSE estimates
                as are necessary to reach at least a cumulative component
                probability of n_components_or_probability.
        """

        if A is None:
            A = np.eye(self.D, dtype=y.dtype)
        Cy_invs = self._prepare_for_prediction(A, Cn)
        h_est = np.zeros([y.shape[0], A.shape[-1]], dtype=y.dtype)
        if isinstance(n_summands_or_proba, int):
            # n_summands_or_proba represents a number of summands
            if n_summands_or_proba == 1:
                # use predicted label to choose the channel covariance matrix
                labels = self.predict_proba_max(y)
                for yi in range(y.shape[0]):
                    h_est[yi] = self._lmmse(y[yi], A, labels[yi], Cy_invs[labels[yi]])
            else:
                # use predicted probabilites to compute weighted sum of estimators
                proba = self.predict_proba(y)
                for yi in range(y.shape[0]):
                    # indices for probabilites in descending order
                    idx_sort = np.argsort(proba[yi, :])[::-1]
                    for argproba in idx_sort[:n_summands_or_proba]:
                        h_est[yi, :] += proba[yi, argproba] * self._lmmse(y[yi], A, argproba, Cy_invs[argproba])
                    h_est[yi, :] /= np.sum(proba[yi, idx_sort[:n_summands_or_proba]])
        elif n_summands_or_proba == 1.0:
            # use all predicted probabilities to compute weighted sum of estimators
            proba = self.predict_proba(y)
            for yi in range(y.shape[0]):
                for argproba in range(proba.shape[1]):
                    h_est[yi, :] += proba[yi, argproba] * self._lmmse(y[yi], A, argproba, Cy_invs[argproba])
        else:
            # n_summands_or_proba represents a probability
            # use predicted probabilites to compute weighted sum of estimators
            proba = self.predict_proba(y)
            for yi in range(y.shape[0]):
                # probabilities and corresponding indices in descending order
                idx_sort = np.argsort(proba[yi, :])[::-1]
                nr_proba = np.searchsorted(np.cumsum(proba[yi, idx_sort]), n_summands_or_proba) + 1
                for argproba in idx_sort[:nr_proba]:
                    h_est[yi, :] += proba[yi, argproba] * self._lmmse(y[yi], A, argproba, Cy_invs[argproba])
                h_est[yi, :] /= np.sum(proba[yi, idx_sort[:nr_proba]])
        return h_est


    def _prepare_for_prediction(self, A, Cn):
        self._means = np.squeeze(A @ np.expand_dims(self.means, 2))
        self._covs = A @ self.covs @ A.conj().T
        for k in range(self.n_components):
            self._covs[k] += Cn
        self._inv_covs = np.linalg.pinv(self._covs, hermitian=True)
        return self._inv_covs


    def _lmmse(self, y, A, k, Cy_inv):
        return self.means[k] + self.covs[k] @ A.conj().T @ (Cy_inv @ (y - A @ self.means[k]))
