import argparse
import random
import time

import numpy as np

from mfa_estimator import MfaEstimator


def mse(x, y):
    """Compute the mean square error between x and y."""
    return np.sum(np.abs(x - y) ** 2) / x.size


def standard_normal_cplx(n_samples, n_dim, rng=np.random.default_rng()):
    """Standard complex normal random numbers of shape (n_samples, n_dim)."""
    return (
        rng.standard_normal((n_samples, n_dim))
        + 1j * rng.standard_normal((n_samples, n_dim))
    ) / np.sqrt(2)


def example1():
    """No observation matrix, different diagonal Psi matrices."""
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_val = 100
    n_dim = 10

    h_train = standard_normal_cplx(n_train, n_dim, rng)
    h_val = standard_normal_cplx(n_val, n_dim, rng)
    noise_val = standard_normal_cplx(n_val, n_dim, rng)

    # The SNR is 0 dB.
    Cn = np.eye(n_dim)
    y_val = h_val + noise_val

    tic = time.time()
    mfa_est = MfaEstimator(
        n_components=16,
        latent_dim=12,
        ppca=False,
        lock_psis=False,
        rs_clip=1e-6,
        max_condition_number=1e6,
        max_iter=400,
        verbose=False,
    )
    mfa_est.fit(h_train)
    toc = time.time()
    print(f"training done: {toc - tic} sec.")

    tic = time.time()
    h_est = mfa_est.estimate(y=y_val, Cn=Cn, A=None, n_summands_or_proba=1.0)
    print("NMSE of n_summands_or_proba=1.0 (all):", mse(h_est, h_val))

    h_est = mfa_est.estimate(y=y_val, Cn=Cn, A=None, n_summands_or_proba=5)
    print("NMSE of n_summands_or_proba=5:", mse(h_est, h_val))

    toc = time.time()
    print(f"example 1 estimation done: {toc - tic} sec.")


def example2():
    """Selection matrix A, different diagonal Psi matrices."""
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_val = 100
    n_dim = 10
    n_dim_obs = 5

    # Create random selection matrix.
    A = np.zeros((n_dim_obs, n_dim))
    pattern_vec = random.sample(range(n_dim), n_dim_obs)
    pattern_vec.sort()

    for i, val in enumerate(pattern_vec):
        A[i, val] = 1

    h_train = standard_normal_cplx(n_train, n_dim, rng)
    h_val = standard_normal_cplx(n_val, n_dim, rng)
    noise_val = standard_normal_cplx(n_val, n_dim_obs, rng)

    # The SNR is 0 dB.
    Cn = np.eye(n_dim_obs)
    y_val = np.squeeze(np.matmul(A, np.expand_dims(h_val, 2))) + noise_val

    tic = time.time()
    mfa_est = MfaEstimator(
        n_components=16,
        latent_dim=12,
        ppca=False,
        lock_psis=False,
        rs_clip=1e-6,
        max_condition_number=1e6,
        max_iter=400,
        verbose=False,
    )
    mfa_est.fit(h_train)
    toc = time.time()
    print(f"training done: {toc - tic} sec.")

    tic = time.time()
    h_est = mfa_est.estimate(y=y_val, Cn=Cn, A=A, n_summands_or_proba=1.0)
    print("NMSE of n_summands_or_proba=1.0 (all):", mse(h_est, h_val))

    h_est = mfa_est.estimate(y=y_val, Cn=Cn, A=A, n_summands_or_proba=5)
    print("NMSE of n_summands_or_proba=5:", mse(h_est, h_val))

    toc = time.time()
    print(f"example 2 estimation done: {toc - tic} sec.")


def example3():
    """No observation matrix, single diagonal Psi matrix for all components."""
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_val = 100
    n_dim = 10

    h_train = standard_normal_cplx(n_train, n_dim, rng)
    h_val = standard_normal_cplx(n_val, n_dim, rng)
    noise_val = standard_normal_cplx(n_val, n_dim, rng)

    # The SNR is 0 dB.
    Cn = np.eye(n_dim)
    y_val = h_val + noise_val

    tic = time.time()
    mfa_est = MfaEstimator(
        n_components=16,
        latent_dim=12,
        ppca=False,
        lock_psis=True,
        rs_clip=0.0,
        max_condition_number=1e6,
        max_iter=400,
        verbose=False,
    )
    mfa_est.fit(h_train)
    toc = time.time()
    print(f"training done: {toc - tic} sec.")

    tic = time.time()
    h_est = mfa_est.estimate(y=y_val, Cn=Cn, A=None, n_summands_or_proba=1.0)
    print("NMSE of n_summands_or_proba=1.0 (all):", mse(h_est, h_val))

    h_est = mfa_est.estimate(y=y_val, Cn=Cn, A=None, n_summands_or_proba=5)
    print("NMSE of n_summands_or_proba=5:", mse(h_est, h_val))

    toc = time.time()
    print(f"example 3 estimation done: {toc - tic} sec.")


def example4():
    """No observation matrix, single scaled identity Psi matrix for all components."""
    rng = np.random.default_rng(1235428719812346)

    n_train = 1_000
    n_val = 100
    n_dim = 10

    h_train = standard_normal_cplx(n_train, n_dim, rng)
    h_val = standard_normal_cplx(n_val, n_dim, rng)
    noise_val = standard_normal_cplx(n_val, n_dim, rng)

    # The SNR is 0 dB.
    Cn = np.eye(n_dim)
    y_val = h_val + noise_val

    tic = time.time()
    mfa_est = MfaEstimator(
        n_components=16,
        latent_dim=12,
        ppca=True,
        lock_psis=True,
        rs_clip=0.0,
        max_condition_number=1e6,
        max_iter=400,
        verbose=False,
    )
    mfa_est.fit(h_train)
    toc = time.time()
    print(f"training done: {toc - tic} sec.")

    tic = time.time()
    h_est = mfa_est.estimate(y=y_val, Cn=Cn, A=None, n_summands_or_proba=1.0)
    print("NMSE of n_summands_or_proba=1.0 (all):", mse(h_est, h_val))

    h_est = mfa_est.estimate(y=y_val, Cn=Cn, A=None, n_summands_or_proba=5)
    print("NMSE of n_summands_or_proba=5:", mse(h_est, h_val))

    toc = time.time()
    print(f"example 4 estimation done: {toc - tic} sec.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nr", type=int, default=0)
    parargs = parser.parse_args()

    if parargs.nr == 1:
        example1()
    elif parargs.nr == 2:
        example2()
    elif parargs.nr == 3:
        example3()
    elif parargs.nr == 4:
        example4()