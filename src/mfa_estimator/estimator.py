"""MFA-based estimator for complex-valued linear inverse problems."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from cplx_mfa import ComplexMFA


class MfaEstimator(ComplexMFA):
    """MFA-based estimator for linear inverse problems.

    The estimator uses a fitted complex-valued mixture of factor analyzers as
    a prior and applies component-wise LMMSE estimation for observations from a
    linear measurement model.
    """

    def estimate(
        self,
        y: npt.ArrayLike,
        Cn: npt.NDArray,
        A: npt.NDArray | None = None,
        n_summands_or_proba: float = 1,
    ) -> npt.NDArray:
        """Estimate latent vectors from noisy linear observations.

        Parameters
        ----------
        y : array-like of shape (n_samples, n_observations)
            Complex-valued observations.
        Cn : ndarray of shape (n_observations, n_observations)
            Observation noise covariance matrix.
        A : ndarray of shape (n_observations, n_features), optional
            Linear observation matrix. If omitted, the identity matrix is used.
        n_summands_or_proba : int or float, default=1
            If an integer is provided, use the top component probabilities up
            to the specified number of summands. If a float is provided, use as
            many components as required to reach the cumulative probability.

        Returns
        -------
        h_est : ndarray of shape (n_samples, n_features)
            Estimated latent vectors.
        """
        y = np.asarray(y)

        if A is None:
            A = np.eye(self.means_.shape[1], dtype=y.dtype)

        Cy_invs = self._prepare_for_prediction(A, Cn)
        h_est = np.zeros((y.shape[0], A.shape[-1]), dtype=y.dtype)

        if isinstance(n_summands_or_proba, int):
            if n_summands_or_proba == 1:
                labels = self.predict(y)
                for yi in range(y.shape[0]):
                    h_est[yi] = self._lmmse(
                        y[yi],
                        A,
                        labels[yi],
                        Cy_invs[labels[yi]],
                    )
            else:
                proba = self.predict_proba(y)
                for yi in range(y.shape[0]):
                    idx_sort = np.argsort(proba[yi, :])[::-1]
                    for argproba in idx_sort[:n_summands_or_proba]:
                        h_est[yi, :] += proba[yi, argproba] * self._lmmse(
                            y[yi],
                            A,
                            argproba,
                            Cy_invs[argproba],
                        )
                    h_est[yi, :] /= np.sum(
                        proba[yi, idx_sort[:n_summands_or_proba]]
                    )
        elif n_summands_or_proba == 1.0:
            proba = self.predict_proba(y)
            for yi in range(y.shape[0]):
                for argproba in range(proba.shape[1]):
                    h_est[yi, :] += proba[yi, argproba] * self._lmmse(
                        y[yi],
                        A,
                        argproba,
                        Cy_invs[argproba],
                    )
        else:
            proba = self.predict_proba(y)
            for yi in range(y.shape[0]):
                idx_sort = np.argsort(proba[yi, :])[::-1]
                nr_proba = (
                    np.searchsorted(
                        np.cumsum(proba[yi, idx_sort]),
                        n_summands_or_proba,
                    )
                    + 1
                )
                for argproba in idx_sort[:nr_proba]:
                    h_est[yi, :] += proba[yi, argproba] * self._lmmse(
                        y[yi],
                        A,
                        argproba,
                        Cy_invs[argproba],
                    )
                h_est[yi, :] /= np.sum(proba[yi, idx_sort[:nr_proba]])

        return h_est

    def _prepare_for_prediction(
        self,
        A: npt.NDArray,
        Cn: npt.NDArray,
    ) -> npt.NDArray:
        self._means = np.squeeze(A @ np.expand_dims(self.means_, 2))
        self._covs = A @ self.covariances_ @ A.conj().T

        for k in range(self.n_components):
            self._covs[k] += Cn

        self._inv_covs = np.linalg.pinv(self._covs, hermitian=True)
        return self._inv_covs

    def _lmmse(
        self,
        y: npt.NDArray,
        A: npt.NDArray,
        k: int,
        Cy_inv: npt.NDArray,
    ) -> npt.NDArray:
        return self.means_[k] + self.covariances_[k] @ A.conj().T @ (
            Cy_inv @ (y - A @ self.means_[k])
        )