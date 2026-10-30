import numpy as np
import pytest

from qrisk.risk import kupiec_pof, rolling_var_backtest


@pytest.mark.parametrize("n_violations", [0, 100])
def test_kupiec_boundary_violation_counts_are_finite(n_violations):
    lr, p_value = kupiec_pof(n_violations, 100, alpha=0.95)
    assert np.isfinite(lr)
    assert np.isfinite(p_value)
    assert lr >= 0
    assert 0 <= p_value <= 1


@pytest.mark.parametrize(
    ("n_violations", "n_obs"),
    [(0, 0), (-1, 10), (11, 10), (1.5, 10), (True, 10)],
)
def test_kupiec_rejects_invalid_counts(n_violations, n_obs):
    with pytest.raises(ValueError):
        kupiec_pof(n_violations, n_obs)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, np.nan, np.inf])
def test_kupiec_rejects_alpha_outside_open_unit_interval(alpha):
    with pytest.raises(ValueError, match="alpha"):
        kupiec_pof(1, 10, alpha)


def test_rolling_backtest_rejects_nonpositive_window():
    with pytest.raises(ValueError, match="window"):
        rolling_var_backtest([0.01, -0.02, 0.03], window=0)


def test_rolling_backtest_rejects_history_without_out_of_sample_return():
    with pytest.raises(ValueError, match="minimal 4 return"):
        rolling_var_backtest([0.01, -0.02, 0.03], window=3)


def test_rolling_backtest_handles_exact_minimum_history():
    result = rolling_var_backtest([0.01, -0.02, 0.03, -0.01], window=3)
    assert result["n_obs"] == 1
    assert result["n_violations"] in (0, 1)
    assert np.isnan(result["var_series"][:3]).all()
    assert np.isfinite(result["var_series"][3])
    assert np.isfinite(result["kupiec_lr"])
    assert np.isfinite(result["kupiec_p"])


def test_rolling_parametric_window_has_two_observation_minimum():
    with pytest.raises(ValueError, match="minimal 2"):
        rolling_var_backtest([0.01, -0.02, 0.03], window=1, method="parametric")


def test_rolling_backtest_rejects_unknown_method():
    with pytest.raises(ValueError, match="tidak dikenal"):
        rolling_var_backtest([0.01, -0.02, 0.03], window=2, method="unknown")
