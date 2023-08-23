# MFA_estimator
Implementation of the channel estimator based on the Mixture of Factor Analyzers (MFA) from the paper 
- B. Fesl, N. Turan, and W. Utschick, “Low-Rank Structured MMSE Channel Estimation with Mixtures of Factor Analyzers,” in *57th Asilomar Conf. Signals, Syst., Comput.*, 2023.
  https://arxiv.org/abs/2304.14809

The estimator is based on the complex-valued implementation of the expectation-maximization (EM) algorithm for MFA from https://github.com/benediktfesl/MFA_cplx.

## Clone with Submodule
To clone the repository, including the complex-valued EM algorithm as a submodule, use the following command:
```
git clone --recurse-submodules https://github.com/benediktfesl/MFA_estimator.git
```

## Acknowledgement
The implementation is in parts based on the Gaussian Mixture Model (GMM) estimator from https://github.com/michael-koller-91/gmm-estimator.

## License
The implementation of the MFA-based channel estimator is covered by the BSD 3-Clause License:

> BSD 3-Clause License
>
> Copyright (c) 2023 Benedikt Fesl.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
>modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>  list of conditions and the following disclaimer.
>
> * Redistributions in binary form must reproduce the above copyright notice,
>  this list of conditions and the following disclaimer in the documentation
>  and/or other materials provided with the distribution.
>
> * Neither the name of the copyright holder nor the names of its
>  contributors may be used to endorse or promote products derived from
>  this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
