"""Tests for MFA-based linear inverse problem estimation."""

from __future__ import annotations

import numpy as np
import pytest

from mfa_estimator import MfaEstimator


def _make_fitted_two_component_estimator() -> MfaEstimator:
    """Create a deterministic fitted estimator for functional tests."""
    estimator = MfaEstimator(
        n_components=2,
        latent_dim=1,
        ppca=False,
        lock_psis=False,
        max_iter=1,
        verbose=False,
    )

    estimator.means_ = np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j],
            [3.0 + 0.0j, 3.0 + 0.0j],
        ]
    )
    estimator.loadings_ = np.zeros((2, 2, 1), dtype=complex)
    estimator.covariances_ = np.array(
        [
            [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
            [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
        ]
    )
    estimator.precisions_ = np.array(
        [
            [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
            [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
        ]
    )
    estimator.noise_variances_ = np.ones((2, 2))
    estimator.weights_ = np.array([0.5, 0.5])

    return estimator


def test_estimate_with_identity_observation_matrix() -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.array([[0.1 + 0.0j, -0.1 + 0.0j]])
    Cn = 0.1 * np.eye(2)

    h_est = estimator.estimate(y=y, Cn=Cn, A=None, n_summands_or_proba=1.0)

    assert h_est.shape == (1, 2)
    assert np.iscomplexobj(h_est)
    assert np.all(np.isfinite(h_est))


def test_estimate_with_rectangular_observation_matrix() -> None:
    estimator = _make_fitted_two_component_estimator()

    A = np.array([[1.0 + 0.0j, 0.0 + 0.0j]])
    y = np.array([[0.2 + 0.0j]])
    Cn = np.array([[0.1 + 0.0j]])

    h_est = estimator.estimate(y=y, Cn=Cn, A=A, n_summands_or_proba=1.0)

    assert h_est.shape == (1, 2)
    assert np.iscomplexobj(h_est)
    assert np.all(np.isfinite(h_est))


def test_estimate_with_complex_observation_matrix() -> None:
    estimator = _make_fitted_two_component_estimator()

    A = np.array([[1.0 + 1.0j, 0.0 + 0.0j]])
    y = np.array([[0.2 + 0.1j]])
    Cn = np.array([[0.1 + 0.0j]])

    h_est = estimator.estimate(y=y, Cn=Cn, A=A, n_summands_or_proba=1.0)

    assert h_est.shape == (1, 2)
    assert np.iscomplexobj(h_est)
    assert np.all(np.isfinite(h_est))


def test_estimate_accepts_real_inputs() -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.array([[0.1, -0.1]])
    Cn = np.eye(2)

    h_est = estimator.estimate(y=y, Cn=Cn, n_summands_or_proba=1.0)

    assert h_est.shape == (1, 2)
    assert np.iscomplexobj(h_est)
    assert np.all(np.isfinite(h_est))


def test_estimate_supports_component_count_selection() -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.array([[0.1 + 0.0j, -0.1 + 0.0j]])
    Cn = 0.1 * np.eye(2)

    h_est_one = estimator.estimate(y=y, Cn=Cn, n_summands_or_proba=1)
    h_est_two = estimator.estimate(y=y, Cn=Cn, n_summands_or_proba=2)

    assert h_est_one.shape == (1, 2)
    assert h_est_two.shape == (1, 2)
    assert np.all(np.isfinite(h_est_one))
    assert np.all(np.isfinite(h_est_two))


def test_estimate_supports_cumulative_probability_selection() -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.array([[0.1 + 0.0j, -0.1 + 0.0j]])
    Cn = 0.1 * np.eye(2)

    h_est_threshold = estimator.estimate(y=y, Cn=Cn, n_summands_or_proba=0.9)
    h_est_all = estimator.estimate(y=y, Cn=Cn, n_summands_or_proba=1.0)

    assert h_est_threshold.shape == (1, 2)
    assert h_est_all.shape == (1, 2)
    assert np.all(np.isfinite(h_est_threshold))
    assert np.all(np.isfinite(h_est_all))


def test_higher_cumulative_probability_can_improve_estimate() -> None:
    estimator = _make_fitted_two_component_estimator()

    true_h = np.array([[1.5 + 0.0j, 1.5 + 0.0j]])
    y = true_h.copy()
    Cn = 0.5 * np.eye(2)

    h_est_top_component = estimator.estimate(
        y=y,
        Cn=Cn,
        n_summands_or_proba=1,
    )
    h_est_all_components = estimator.estimate(
        y=y,
        Cn=Cn,
        n_summands_or_proba=1.0,
    )

    mse_top_component = np.mean(np.abs(h_est_top_component - true_h) ** 2)
    mse_all_components = np.mean(np.abs(h_est_all_components - true_h) ** 2)

    assert mse_all_components < mse_top_component


def test_observation_probabilities_sum_to_one() -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.array(
        [
            [0.1 + 0.0j, -0.1 + 0.0j],
            [2.9 + 0.0j, 3.1 + 0.0j],
        ]
    )
    Cn = 0.1 * np.eye(2)
    A = np.eye(2, dtype=complex)

    observation_model = estimator._prepare_observation_model(A=A, Cn=Cn)
    probabilities = estimator._predict_observation_proba(
        y=y,
        observation_model=observation_model,
    )

    np.testing.assert_allclose(
        np.sum(probabilities, axis=1),
        np.ones(y.shape[0]),
    )


def test_estimate_does_not_mutate_fitted_prior_parameters() -> None:
    estimator = _make_fitted_two_component_estimator()

    means_before = estimator.means_.copy()
    covariances_before = estimator.covariances_.copy()
    weights_before = estimator.weights_.copy()

    y = np.array([[0.1 + 0.0j, -0.1 + 0.0j]])
    Cn = 0.1 * np.eye(2)

    estimator.estimate(y=y, Cn=Cn, n_summands_or_proba=1.0)

    np.testing.assert_allclose(estimator.means_, means_before)
    np.testing.assert_allclose(estimator.covariances_, covariances_before)
    np.testing.assert_allclose(estimator.weights_, weights_before)


def test_estimate_requires_fitted_model() -> None:
    estimator = MfaEstimator(
        n_components=2,
        latent_dim=1,
        max_iter=1,
        verbose=False,
    )

    y = np.array([[0.0 + 0.0j, 0.0 + 0.0j]])
    Cn = np.eye(2)

    with pytest.raises(RuntimeError, match="must be fitted"):
        estimator.estimate(y=y, Cn=Cn)


def test_estimate_rejects_invalid_y_shape() -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    Cn = np.eye(2)

    with pytest.raises(ValueError, match="y must be a 2D array"):
        estimator.estimate(y=y, Cn=Cn)


def test_estimate_rejects_empty_y() -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.empty((0, 2), dtype=complex)
    Cn = np.eye(2)

    with pytest.raises(ValueError, match="y must contain at least one sample"):
        estimator.estimate(y=y, Cn=Cn)


def test_estimate_rejects_invalid_noise_covariance_shape() -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.array([[0.0 + 0.0j, 0.0 + 0.0j]])
    Cn = np.eye(3)

    with pytest.raises(ValueError, match="Cn has incompatible shape"):
        estimator.estimate(y=y, Cn=Cn)


def test_estimate_rejects_invalid_observation_matrix_shape() -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.array([[0.0 + 0.0j]])
    Cn = np.eye(1)
    A = np.eye(2)

    with pytest.raises(ValueError, match="A has incompatible shape"):
        estimator.estimate(y=y, Cn=Cn, A=A)


def test_estimate_rejects_nan_observations() -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.array([[np.nan + 0.0j, 0.0 + 0.0j]])
    Cn = np.eye(2)

    with pytest.raises(ValueError, match="y must not contain"):
        estimator.estimate(y=y, Cn=Cn)


def test_estimate_rejects_nan_noise_covariance() -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.array([[0.0 + 0.0j, 0.0 + 0.0j]])
    Cn = np.array([[1.0, np.nan], [0.0, 1.0]])

    with pytest.raises(ValueError, match="Cn must not contain"):
        estimator.estimate(y=y, Cn=Cn)


def test_estimate_rejects_nan_observation_matrix() -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.array([[0.0 + 0.0j]])
    Cn = np.eye(1)
    A = np.array([[np.nan, 0.0]])

    with pytest.raises(ValueError, match="A must not contain"):
        estimator.estimate(y=y, Cn=Cn, A=A)


@pytest.mark.parametrize("value", [0, -1, 3])
def test_estimate_rejects_invalid_number_of_summands(value: int) -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.array([[0.0 + 0.0j, 0.0 + 0.0j]])
    Cn = np.eye(2)

    with pytest.raises(ValueError):
        estimator.estimate(y=y, Cn=Cn, n_summands_or_proba=value)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1])
def test_estimate_rejects_invalid_probability_threshold(value: float) -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.array([[0.0 + 0.0j, 0.0 + 0.0j]])
    Cn = np.eye(2)

    with pytest.raises(ValueError):
        estimator.estimate(y=y, Cn=Cn, n_summands_or_proba=value)


def test_estimate_rejects_bool_component_selection() -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.array([[0.0 + 0.0j, 0.0 + 0.0j]])
    Cn = np.eye(2)

    with pytest.raises(TypeError, match="not bool"):
        estimator.estimate(y=y, Cn=Cn, n_summands_or_proba=True)


def test_fit_from_cplx_mfa_package_integrates_with_estimate() -> None:
    rng = np.random.default_rng(1234)

    h_train = (
        rng.standard_normal((40, 2)) + 1j * rng.standard_normal((40, 2))
    ) / np.sqrt(2)
    y = h_train[:5]
    Cn = 0.1 * np.eye(2)

    estimator = MfaEstimator(
        n_components=2,
        latent_dim=1,
        max_iter=2,
        random_state=1234,
        verbose=False,
    )

    with pytest.warns(RuntimeWarning, match="EM did not converge"):
        estimator.fit(h_train)

    h_est = estimator.estimate(y=y, Cn=Cn, n_summands_or_proba=1.0)

    assert h_est.shape == y.shape
    assert np.iscomplexobj(h_est)
    assert np.all(np.isfinite(h_est))


def test_estimate_multiple_samples() -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.array(
        [
            [0.1 + 0.0j, -0.1 + 0.0j],
            [2.9 + 0.0j, 3.1 + 0.0j],
            [1.5 + 0.0j, 1.5 + 0.0j],
        ]
    )
    Cn = 0.1 * np.eye(2)

    h_est = estimator.estimate(y=y, Cn=Cn, n_summands_or_proba=1.0)

    assert h_est.shape == y.shape
    assert np.iscomplexobj(h_est)
    assert np.all(np.isfinite(h_est))


def test_estimate_all_components_by_count_matches_probability_one() -> None:
    estimator = _make_fitted_two_component_estimator()

    y = np.array([[1.5 + 0.0j, 1.5 + 0.0j]])
    Cn = 0.5 * np.eye(2)

    h_est_by_count = estimator.estimate(
        y=y,
        Cn=Cn,
        n_summands_or_proba=2,
    )
    h_est_by_probability = estimator.estimate(
        y=y,
        Cn=Cn,
        n_summands_or_proba=1.0,
    )

    np.testing.assert_allclose(h_est_by_count, h_est_by_probability)


def test_zero_mean_fit_from_cplx_mfa_integrates_with_estimate() -> None:
    rng = np.random.default_rng(1234)

    h_train = (
        rng.standard_normal((40, 2)) + 1j * rng.standard_normal((40, 2))
    ) / np.sqrt(2)
    y = h_train[:5]
    Cn = 0.1 * np.eye(2)

    estimator = MfaEstimator(
        n_components=2,
        latent_dim=1,
        zero_mean=True,
        max_iter=2,
        random_state=1234,
        verbose=False,
    )

    with pytest.warns(RuntimeWarning, match="EM did not converge"):
        estimator.fit(h_train)

    np.testing.assert_allclose(estimator.means_, 0.0)

    h_est = estimator.estimate(y=y, Cn=Cn, n_summands_or_proba=1.0)

    assert h_est.shape == y.shape
    assert np.iscomplexobj(h_est)
    assert np.all(np.isfinite(h_est))
