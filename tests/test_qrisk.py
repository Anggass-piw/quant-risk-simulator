import numpy as np
from scipy import stats

from qrisk import (estimate_gbm_params, expected_shortfall, kupiec_pof, simulate_gbm,
                   var_cornish_fisher, var_historical, var_parametric)


def test_gbm_expected_terminal_price():
    # E[S_T] = S0 * exp(mu * T)
    s0, mu, sigma, days = 100.0, 0.10, 0.30, 252
    paths = simulate_gbm([s0], [mu], [sigma], horizon_days=days, n_paths=200_000, seed=1)
    expected = s0 * np.exp(mu * days / 252)
    assert abs(paths[:, -1, 0].mean() - expected) / expected < 0.01


def test_gbm_log_return_std():
    paths = simulate_gbm([100], [0.05], [0.25], horizon_days=252, n_paths=100_000, seed=2)
    log_ret = np.log(paths[:, -1, 0] / 100)
    assert abs(log_ret.std() - 0.25) < 0.01


def test_gbm_correlation_preserved():
    corr = np.array([[1, 0.7], [0.7, 1]])
    paths = simulate_gbm([100, 100], [0, 0], [0.2, 0.2], corr, horizon_days=1,
                         n_paths=100_000, seed=3)
    r = np.log(paths[:, -1, :] / 100)
    assert abs(np.corrcoef(r.T)[0, 1] - 0.7) < 0.02


def test_estimate_params_roundtrip():
    paths = simulate_gbm([100], [0.12], [0.20], horizon_days=252 * 40, n_paths=1, seed=4,
                         antithetic=False)[0, :, 0]
    mu, sigma, _ = estimate_gbm_params(np.diff(np.log(paths)))
    assert abs(sigma[0] - 0.20) < 0.01


def test_parametric_var_matches_closed_form():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 500_000)
    expected = -0.0005 + 0.01 * stats.norm.ppf(0.99)
    assert abs(var_parametric(r, 0.99) - expected) < 5e-4


def test_normal_data_methods_agree():
    r = np.random.default_rng(1).normal(0, 0.01, 500_000)
    h, p, cf = var_historical(r), var_parametric(r), var_cornish_fisher(r)
    assert abs(h - p) < 3e-4 and abs(cf - p) < 3e-4


def test_cvar_exceeds_var():
    r = np.random.default_rng(2).standard_t(4, 100_000) * 0.01
    assert expected_shortfall(r, 0.95) > var_historical(r, 0.95) > 0


def test_cornish_fisher_widens_var_on_fat_tails():
    # t(8): ekor lebih tebal dari normal => VaR 99% sebenarnya lebih besar dari VaR normal
    r = np.random.default_rng(3).standard_t(8, 500_000) * 0.01
    assert var_historical(r, 0.99) > var_parametric(r, 0.99)
    assert var_cornish_fisher(r, 0.99) > var_parametric(r, 0.99)


def test_kupiec():
    lr, p = kupiec_pof(5, 100, 0.95)  # tepat sesuai ekspektasi
    assert lr < 1e-9 and p > 0.99
    lr, p = kupiec_pof(20, 100, 0.95)  # terlalu banyak pelanggaran
    assert p < 0.001
