# mfa-estimator

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![Package](https://img.shields.io/badge/package-PyPI-informational.svg)](https://pypi.org/project/mfa-estimator/)

MFA-based estimator for complex-valued linear inverse problems.

`mfa-estimator` provides an estimator for noisy linear observation models using a complex-valued mixture of factor analyzers (MFA) prior. The package builds on [cplx-mfa](https://pypi.org/project/cplx-mfa/) for fitting the complex-valued MFA prior and adds an estimation layer for inverse problems of the form

```text
y = A h + n
```

where `h` is the unknown complex-valued vector, `A` is a known linear observation matrix, and `n` is complex Gaussian observation noise.

The estimator is domain-independent and can be used for linear inverse problems in signal processing, communications, and related applications. Channel estimation is one motivating application and is discussed in the research background section.

## ✨ Highlights

- MFA-based estimator for complex-valued linear inverse problems
- Supports general linear observation models of the form `y = A h + n`
- Uses complex-valued MFA priors fitted with [`cplx-mfa`](https://pypi.org/project/cplx-mfa/)
- Component-wise LMMSE estimation under the fitted mixture prior
- Posterior component weighting in the observation domain
- Supports identity and rectangular observation matrices
- Supports full posterior mixture estimates or truncated component sums
- scikit-learn-like workflow via inherited `fit(...)` and added `estimate(...)`
- Modern Python packaging with `pyproject.toml`, `uv`, `pytest`, and `ruff`

## 📌 Citation

If you use `mfa-estimator` in academic work, please cite the package directly:

```bibtex
@software{fesl_mfa_estimator,
  author = {Fesl, Benedikt},
  title = {{mfa-estimator}: MFA-based estimator for complex-valued linear inverse problems},
  year = {2026},
  url = {https://github.com/benediktfesl/MFA_estimator},
  version = {0.1.0}
}
```

Plain-text citation:

> B. Fesl, `mfa-estimator`: MFA-based estimator for complex-valued linear inverse problems, version 0.1.0. Available: https://github.com/benediktfesl/MFA_estimator

If you use the estimator in the context of channel estimation, please also consider citing the related papers listed in the research background section.

## 📦 Installation

Install from PyPI:

```bash
pip install mfa-estimator
```

or with `uv`:

```bash
uv add mfa-estimator
```

For development, clone the repository and install the development environment:

```bash
git clone https://github.com/benediktfesl/MFA_estimator.git
cd MFA_estimator
uv sync --group dev
```

## 🚀 Quick Start

```python
import numpy as np

from mfa_estimator import MfaEstimator

rng = np.random.default_rng(0)

h_train = (
    rng.normal(size=(1_000, 8))
    + 1j * rng.normal(size=(1_000, 8))
) / np.sqrt(2.0)

h_val = (
    rng.normal(size=(100, 8))
    + 1j * rng.normal(size=(100, 8))
) / np.sqrt(2.0)

noise = (
    rng.normal(size=(100, 8))
    + 1j * rng.normal(size=(100, 8))
) / np.sqrt(2.0)

# Identity observation model: y = h + n
noise_covariance = np.eye(8)
y = h_val + noise

estimator = MfaEstimator(
    n_components=4,
    latent_dim=2,
    random_state=0,
    max_iter=100,
    verbose=False,
)

# Fit the complex-valued MFA prior p(h).
estimator.fit(h_train)

# Estimate h from noisy observations y.
h_est = estimator.estimate(
    y=y,
    Cn=noise_covariance,
    A=None,
    n_summands_or_proba=1.0,
)
```

The estimator follows a two-step pattern:

1. `fit(h_train)` fits the complex-valued MFA prior using the inherited `cplx-mfa` implementation.
2. `estimate(y, Cn, A)` estimates the unknown vector from noisy linear observations.

## 🧩 Estimation Model

The package assumes a linear observation model

```text
y = A h + n
```

where:

| Symbol | Description |
|---|---|
| `h` | Unknown complex-valued vector to be estimated. |
| `y` | Noisy complex-valued observation. |
| `A` | Known linear observation matrix. |
| `n` | Zero-mean complex Gaussian noise with covariance `Cn`. |

The unknown vector `h` is modeled with a fitted complex-valued MFA prior:

```text
p(h) = Σ_k π_k CN(h; μ_k, C_k)
```

For each mixture component, the observation-domain model is

```text
y | k ~ CN(A μ_k, A C_k Aᴴ + Cn)
```

The estimator computes posterior component probabilities in the observation domain and combines component-wise LMMSE estimates.

## 🧠 Estimator API

The main class is:

```python
from mfa_estimator import MfaEstimator
```

`MfaEstimator` inherits from [`cplx_mfa.ComplexMFA`](https://pypi.org/project/cplx-mfa/) and therefore supports the same prior-fitting API.

Core methods:

| Method | Description |
|---|---|
| `fit(X)` | Fit the complex-valued MFA prior. Inherited from `cplx-mfa`. |
| `predict(X)` | Predict the most likely MFA prior component. Inherited from `cplx-mfa`. |
| `predict_proba(X)` | Return prior-domain component probabilities. Inherited from `cplx-mfa`. |
| `sample(n_samples=1, rng=None)` | Draw samples from the fitted MFA prior. Inherited from `cplx-mfa`. |
| `estimate(y, Cn, A=None, n_summands_or_proba=1)` | Estimate unknown vectors from noisy linear observations. |

Constructor parameters are inherited from `ComplexMFA`:

| Parameter | Description |
|---|---|
| `n_components` | Number of mixture components. |
| `latent_dim` | Latent dimensionality of each factor analyzer. |
| `ppca` | If `True`, use one isotropic noise variance per component. |
| `lock_psis` | If `True`, use shared diagonal noise variances across components. |
| `rs_clip` | Lower clipping value for responsibilities during EM. |
| `max_condition_number` | Scaling factor used for random loading initialization. |
| `max_iter` | Maximum number of EM iterations. |
| `tol` | Relative convergence tolerance. |
| `random_state` | Integer seed or NumPy random generator used for initialization. |
| `verbose` | If `True`, print EM progress. |

The fitted prior parameters are exposed using the trailing-underscore attributes from `cplx-mfa`:

| Attribute | Description |
|---|---|
| `weights_` | Mixture weights of shape `(n_components,)`. |
| `means_` | Component means of shape `(n_components, n_features)`. |
| `loadings_` | Factor loading matrices of shape `(n_components, n_features, latent_dim)`. |
| `covariances_` | Full implied covariance matrices of shape `(n_components, n_features, n_features)`. |
| `precisions_` | Inverse covariance matrices of shape `(n_components, n_features, n_features)`. |
| `noise_variances_` | Diagonal noise variances of shape `(n_components, n_features)`. |
| `lower_bound_history_` | EM lower-bound values collected during fitting. |

## 🔎 Estimation

The main method added by this package is `estimate(...)`:

```python
h_est = estimator.estimate(
    y=y,
    Cn=noise_covariance,
    A=observation_matrix,
    n_summands_or_proba=1.0,
)
```

Arguments:

| Argument | Description |
|---|---|
| `y` | Observations of shape `(n_samples, n_observations)`. |
| `Cn` | Observation noise covariance of shape `(n_observations, n_observations)`. |
| `A` | Observation matrix of shape `(n_observations, n_features)`. If `None`, the identity matrix is used. |
| `n_summands_or_proba` | Component-selection rule for the posterior mixture estimate. |

The component-selection parameter can be used in two ways:

| Value | Behavior |
|---|---|
| Integer, e.g. `1` or `5` | Use the corresponding number of most likely posterior components. |
| Float in `(0, 1]`, e.g. `0.9` | Use the fewest most likely components whose cumulative posterior probability reaches the threshold. |
| `1.0` | Use all components. |

## 🧪 Examples

Run the example script:

```bash
uv run python examples/mfa_estimator_example.py --nr 1
```

Available examples:

| Number | Description |
|---|---|
| `1` | Identity observation model with component-wise diagonal noise variances. |
| `2` | Selection observation model with component-wise diagonal noise variances. |
| `3` | Identity observation model with shared diagonal noise variances. |
| `4` | Identity observation model with shared isotropic PPCA-style noise variances. |

Run all examples back to back:

```bash
for nr in 1 2 3 4; do
  uv run python examples/mfa_estimator_example.py --nr "$nr"
done
```

Use `--help` to inspect the script interface:

```bash
uv run python examples/mfa_estimator_example.py --help
```

## 📚 Research Background

This package was originally developed in the context of low-rank structured MMSE estimation with mixture models. The implementation is domain-independent, but the motivating application is channel estimation in communication systems.

Related publications:

- B. Fesl, N. Turan, and W. Utschick, “Low-Rank Structured MMSE Channel Estimation with Mixtures of Factor Analyzers,” *57th Asilomar Conference on Signals, Systems, and Computers*, 2023.  
  [[IEEE](https://ieeexplore.ieee.org/document/10477088)] [[arXiv](https://arxiv.org/abs/2304.14809)]

- B. Fesl, N. Turan, B. Böck, and W. Utschick, “Channel Estimation for Quantized Systems based on Conditionally Gaussian Latent Models,” *IEEE Transactions on Signal Processing*, 2024.  
  [[IEEE](https://ieeexplore.ieee.org/document/10454252)] [[arXiv](https://arxiv.org/abs/2309.04014)]

- B. Fesl, “Generative Model-Aided Channel Estimation Design and Optimality Analysis,” *Ph.D. dissertation, Technical University of Munich*, 2025. 
  [[Link](https://mediatum.ub.tum.de/?id=1748775)]

## 🧪 Development

Install the development environment with `uv`:

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

Format code:

```bash
uv run ruff format .
```

Run an example:

```bash
uv run python examples/mfa_estimator_example.py --nr 1
```

Build the package:

```bash
uv run python -m build
```

Check the package distribution:

```bash
uv run twine check dist/*
```

## ✅ Test Coverage

The test suite covers:

- package imports
- integration with the `cplx-mfa` package
- estimation with identity observation matrices
- estimation with rectangular observation matrices
- estimation with complex-valued observation matrices
- real-valued input handling
- posterior probability normalization
- component-count selection
- cumulative-probability selection
- unfitted-estimator behavior
- invalid input validation
- non-finite input validation
- preservation of fitted prior parameters during estimation
- example script execution

## 📄 License

This project is licensed under the [BSD 3-Clause License](LICENSE).
