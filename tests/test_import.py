from cplx_mfa import ComplexMFA
from mfa_estimator import MfaEstimator


def test_import_mfa_estimator() -> None:
    assert issubclass(MfaEstimator, ComplexMFA)


def test_import_version() -> None:
    import mfa_estimator

    assert isinstance(mfa_estimator.__version__, str)