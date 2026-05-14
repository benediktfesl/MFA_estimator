"""Examples for MFA-based estimation of complex-valued linear inverse problems."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from mfa_estimator import MfaEstimator

SEED = 1235428719812346


@dataclass(frozen=True)
class ExampleConfig:
    """Configuration for one synthetic MFA estimation example."""

    name: str
    n_train: int = 1_000
    n_val: int = 100
    n_dim: int = 10
    n_dim_obs: int | None = None
    n_components: int = 16
    latent_dim: int = 12
    ppca: bool = False
    lock_psis: bool = False
    rs_clip: float = 1e-6
    max_condition_number: float = 1e6
    max_iter: int = 400


def mse(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the mean square error between x and y."""
    return float(np.sum(np.abs(x - y) ** 2) / x.size)


def standard_normal_cplx(
    n_samples: int,
    n_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate standard complex normal random numbers."""
    return (
        rng.standard_normal((n_samples, n_dim))
        + 1j * rng.standard_normal((n_samples, n_dim))
    ) / np.sqrt(2)


def make_selection_matrix(
    n_dim_obs: int,
    n_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a random row-selection matrix."""
    selected_indices = np.sort(rng.choice(n_dim, size=n_dim_obs, replace=False))

    observation_matrix = np.zeros((n_dim_obs, n_dim), dtype=complex)
    observation_matrix[np.arange(n_dim_obs), selected_indices] = 1.0

    return observation_matrix


def make_estimator(config: ExampleConfig) -> MfaEstimator:
    """Create an MFA estimator from an example configuration."""
    return MfaEstimator(
        n_components=config.n_components,
        latent_dim=config.latent_dim,
        ppca=config.ppca,
        lock_psis=config.lock_psis,
        rs_clip=config.rs_clip,
        max_condition_number=config.max_condition_number,
        max_iter=config.max_iter,
        verbose=False,
    )


def run_example(config: ExampleConfig) -> None:
    """Run one synthetic MFA estimation example."""
    rng = np.random.default_rng(SEED)

    n_dim_obs = config.n_dim if config.n_dim_obs is None else config.n_dim_obs

    h_train = standard_normal_cplx(config.n_train, config.n_dim, rng)
    h_val = standard_normal_cplx(config.n_val, config.n_dim, rng)
    noise_val = standard_normal_cplx(config.n_val, n_dim_obs, rng)

    if config.n_dim_obs is None:
        observation_matrix = None
        y_val = h_val + noise_val
    else:
        observation_matrix = make_selection_matrix(
            n_dim_obs=config.n_dim_obs,
            n_dim=config.n_dim,
            rng=rng,
        )
        y_val = h_val @ observation_matrix.T + noise_val

    # The SNR is 0 dB.
    noise_covariance = np.eye(n_dim_obs, dtype=complex)

    tic = time.time()
    estimator = make_estimator(config)
    estimator.fit(h_train)
    toc = time.time()
    print(f"{config.name} training done: {toc - tic:.3f} sec.")

    tic = time.time()
    h_est = estimator.estimate(
        y=y_val,
        Cn=noise_covariance,
        A=observation_matrix,
        n_summands_or_proba=1.0,
    )
    print("NMSE of n_summands_or_proba=1.0 (all):", mse(h_est, h_val))

    h_est = estimator.estimate(
        y=y_val,
        Cn=noise_covariance,
        A=observation_matrix,
        n_summands_or_proba=5,
    )
    print("NMSE of n_summands_or_proba=5:", mse(h_est, h_val))

    toc = time.time()
    print(f"{config.name} estimation done: {toc - tic:.3f} sec.")


EXAMPLES = {
    1: ExampleConfig(
        name="example 1: identity observation, component-wise diagonal Psi",
        ppca=False,
        lock_psis=False,
        rs_clip=1e-6,
    ),
    2: ExampleConfig(
        name="example 2: selection observation, component-wise diagonal Psi",
        n_dim_obs=5,
        ppca=False,
        lock_psis=False,
        rs_clip=1e-6,
    ),
    3: ExampleConfig(
        name="example 3: identity observation, shared diagonal Psi",
        ppca=False,
        lock_psis=True,
        rs_clip=0.0,
    ),
    4: ExampleConfig(
        name="example 4: identity observation, shared scaled identity Psi",
        ppca=True,
        lock_psis=True,
        rs_clip=0.0,
    ),
}


def main() -> None:
    """Run one of the available examples."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nr",
        type=int,
        default=0,
        choices=[0, *EXAMPLES.keys()],
        help="Example number to run. Use 0 to only validate imports.",
    )
    args = parser.parse_args()

    if args.nr == 0:
        return

    run_example(EXAMPLES[args.nr])


if __name__ == "__main__":
    main()
