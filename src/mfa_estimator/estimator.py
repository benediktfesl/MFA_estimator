"""MFA-based estimator for complex-valued linear inverse problems."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
from cplx_mfa import ComplexMFA


@dataclass(frozen=True)
class _ObservationModel:
    """Component-wise Gaussian model in the observation domain."""

    means: np.ndarray
    precisions: np.ndarray
    log_determinants: np.ndarray


class MfaEstimator(ComplexMFA):
    """MFA-based estimator for linear inverse problems.

    The estimator uses a fitted complex-valued mixture of factor analyzers as a
    prior for the unknown vector h and applies component-wise LMMSE estimation
    for observations from a linear model

        y = A h + n,

    where n is zero-mean complex Gaussian noise with covariance Cn.

    The MFA prior parameters are fitted by the inherited
    :meth:`cplx_mfa.ComplexMFA.fit` method. This class only adds the estimation
    step for noisy linear observations.
    """

    def estimate(
        self,
        y: npt.ArrayLike,
        Cn: npt.ArrayLike,
        A: npt.ArrayLike | None = None,
        n_summands_or_proba: int | float = 1,
    ) -> np.ndarray:
        """Estimate latent vectors from noisy linear observations.

        Parameters
        ----------
        y : array-like of shape (n_samples, n_observations)
            Complex-valued observations.
        Cn : array-like of shape (n_observations, n_observations)
            Observation noise covariance matrix.
        A : array-like of shape (n_observations, n_features), optional
            Linear observation matrix. If omitted, the identity matrix is used.
        n_summands_or_proba : int or float, default=1
            If an integer is provided, use the corresponding number of most
            likely mixture components. If a float in (0, 1] is provided, use as
            many components as required to reach the cumulative posterior
            probability. The value 1.0 uses all components.

        Returns
        -------
        h_est : ndarray of shape (n_samples, n_features)
            Estimated latent vectors.
        """
        y, Cn, A = self._validate_estimation_inputs(y=y, Cn=Cn, A=A)
        self._validate_component_selection(n_summands_or_proba)

        observation_model = self._prepare_observation_model(A=A, Cn=Cn)
        probabilities = self._predict_observation_proba(
            y=y,
            observation_model=observation_model,
        )

        h_est = np.zeros((y.shape[0], A.shape[1]), dtype=complex)

        for sample_idx in range(y.shape[0]):
            component_indices = self._select_components(
                probabilities=probabilities[sample_idx],
                n_summands_or_proba=n_summands_or_proba,
            )
            selected_probability_sum = np.sum(
                probabilities[sample_idx, component_indices]
            )

            for component in component_indices:
                h_est[sample_idx] += probabilities[sample_idx, component] * self._lmmse(
                    y=y[sample_idx],
                    A=A,
                    component=component,
                    observation_precision=observation_model.precisions[component],
                )

            h_est[sample_idx] /= selected_probability_sum

        return h_est

    def _validate_estimation_inputs(
        self,
        y: npt.ArrayLike,
        Cn: npt.ArrayLike,
        A: npt.ArrayLike | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate and normalize inputs for estimation."""
        self._check_is_fitted()

        y_array = np.asarray(y)
        noise_covariance = np.asarray(Cn)

        if y_array.ndim != 2:
            raise ValueError(
                "y must be a 2D array of shape (n_samples, n_observations)."
            )
        if y_array.shape[0] < 1:
            raise ValueError("y must contain at least one sample.")
        if y_array.shape[1] < 1:
            raise ValueError("y must contain at least one observation.")

        if not np.iscomplexobj(y_array):
            y_array = y_array.astype(complex)

        if not np.all(np.isfinite(y_array)):
            raise ValueError("y must not contain NaN or infinite values.")

        if noise_covariance.ndim != 2:
            raise ValueError(
                "Cn must be a 2D array of shape (n_observations, n_observations)."
            )

        expected_noise_shape = (y_array.shape[1], y_array.shape[1])
        if noise_covariance.shape != expected_noise_shape:
            raise ValueError(
                "Cn has incompatible shape. "
                f"Expected {expected_noise_shape}, got {noise_covariance.shape}."
            )

        if not np.iscomplexobj(noise_covariance):
            noise_covariance = noise_covariance.astype(complex)

        if not np.all(np.isfinite(noise_covariance)):
            raise ValueError("Cn must not contain NaN or infinite values.")

        if A is None:
            observation_matrix = np.eye(self.means_.shape[1], dtype=complex)
        else:
            observation_matrix = np.asarray(A)

            if observation_matrix.ndim != 2:
                raise ValueError(
                    "A must be a 2D array of shape (n_observations, n_features)."
                )

            if not np.iscomplexobj(observation_matrix):
                observation_matrix = observation_matrix.astype(complex)

            if not np.all(np.isfinite(observation_matrix)):
                raise ValueError("A must not contain NaN or infinite values.")

        expected_observation_matrix_shape = (y_array.shape[1], self.means_.shape[1])
        if observation_matrix.shape != expected_observation_matrix_shape:
            raise ValueError(
                "A has incompatible shape. "
                f"Expected {expected_observation_matrix_shape}, "
                f"got {observation_matrix.shape}."
            )

        return y_array, noise_covariance, observation_matrix

    def _validate_component_selection(
        self,
        n_summands_or_proba: int | float,
    ) -> None:
        """Validate the component-selection parameter."""
        if isinstance(n_summands_or_proba, bool):
            raise TypeError("n_summands_or_proba must be an int or float, not bool.")

        if isinstance(n_summands_or_proba, Integral):
            n_summands = int(n_summands_or_proba)

            if n_summands < 1:
                raise ValueError("n_summands_or_proba must be at least 1.")
            if n_summands > self.n_components:
                raise ValueError(
                    "n_summands_or_proba cannot exceed the number of components. "
                    f"Expected at most {self.n_components}, got {n_summands}."
                )
            return

        if isinstance(n_summands_or_proba, Real):
            probability_threshold = float(n_summands_or_proba)

            if not 0.0 < probability_threshold <= 1.0:
                raise ValueError(
                    "If n_summands_or_proba is a float, it must be in (0, 1]."
                )
            return

        raise TypeError("n_summands_or_proba must be an int or float.")

    def _prepare_observation_model(
        self,
        A: np.ndarray,
        Cn: np.ndarray,
    ) -> _ObservationModel:
        """Build the component-wise Gaussian model for y."""
        observation_means = self.means_ @ A.T
        observation_covariances = (
            A[None, :, :] @ self.covariances_ @ A.conj().T[None, :, :]
        )
        observation_covariances = observation_covariances + Cn[None, :, :]

        observation_precisions = np.linalg.pinv(
            observation_covariances,
            hermitian=True,
        )

        signs, log_determinants = np.linalg.slogdet(observation_covariances)
        if not np.all(np.real(signs) > 0):
            raise np.linalg.LinAlgError(
                "Observation covariance matrices must have positive determinants."
            )

        return _ObservationModel(
            means=observation_means,
            precisions=observation_precisions,
            log_determinants=np.real(log_determinants),
        )

    def _predict_observation_proba(
        self,
        y: np.ndarray,
        observation_model: _ObservationModel,
    ) -> np.ndarray:
        """Calculate posterior component probabilities p(k | y)."""
        log_responsibilities = np.zeros((self.n_components, y.shape[0]))

        safe_weights = np.maximum(self.weights_, np.finfo(float).eps)
        safe_weights = safe_weights / np.sum(safe_weights)

        for component in range(self.n_components):
            log_responsibilities[component] = np.log(
                safe_weights[component]
            ) + self._observation_log_complex_normal(
                y=y,
                mean=observation_model.means[component],
                precision=observation_model.precisions[component],
                log_determinant=observation_model.log_determinants[component],
            )

        log_likelihoods = self._log_sum_exp(log_responsibilities)
        log_responsibilities -= log_likelihoods[None, :]

        return np.exp(log_responsibilities).T

    @staticmethod
    def _observation_log_complex_normal(
        y: np.ndarray,
        mean: np.ndarray,
        precision: np.ndarray,
        log_determinant: float,
    ) -> np.ndarray:
        """Calculate complex Gaussian log likelihoods for observations."""
        centered_y = (y - mean).T
        transformed_y = precision @ centered_y
        quadratic_form = np.sum(centered_y.conj() * transformed_y, axis=0)

        return np.real(-np.log(np.pi) * y.shape[1] - log_determinant - quadratic_form)

    def _select_components(
        self,
        probabilities: np.ndarray,
        n_summands_or_proba: int | float,
    ) -> np.ndarray:
        """Select posterior components for one observation."""
        sorted_indices = np.argsort(probabilities)[::-1]

        if isinstance(n_summands_or_proba, Integral):
            return sorted_indices[: int(n_summands_or_proba)]

        probability_threshold = float(n_summands_or_proba)

        if probability_threshold == 1.0:
            return sorted_indices

        cumulative_probabilities = np.cumsum(probabilities[sorted_indices])
        n_summands = (
            np.searchsorted(cumulative_probabilities, probability_threshold) + 1
        )

        return sorted_indices[:n_summands]

    def _lmmse(
        self,
        y: np.ndarray,
        A: np.ndarray,
        component: int,
        observation_precision: np.ndarray,
    ) -> np.ndarray:
        """Calculate the component-wise LMMSE estimate."""
        mean = self.means_[component]
        covariance = self.covariances_[component]

        return mean + covariance @ A.conj().T @ (observation_precision @ (y - A @ mean))
