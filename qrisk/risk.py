"""Ukuran risiko: VaR (historis, parametrik, Cornish-Fisher, Monte Carlo), CVaR/ES, backtest Kupiec.

Konvensi: loss = -return. VaR dan CVaR dilaporkan sebagai angka positif
(fraksi dari nilai portofolio). VaR_alpha = kuantil ke-alpha dari distribusi loss.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def portfolio_returns(asset_returns: np.ndarray, weights) -> np.ndarray:
    return np.asarray(asset_returns, dtype=float) @ np.asarray(weights, dtype=float)


def var_historical(returns: np.ndarray, alpha: float = 0.95) -> float:
    """VaR historis: kuantil empiris dari loss."""
    return float(np.quantile(-np.asarray(returns, dtype=float), alpha))


def expected_shortfall(returns: np.ndarray, alpha: float = 0.95) -> float:
    """CVaR / Expected Shortfall: rata-rata loss yang melampaui VaR."""
    losses = -np.asarray(returns, dtype=float)
    var = np.quantile(losses, alpha)
    return float(losses[losses >= var].mean())


def var_parametric(returns: np.ndarray, alpha: float = 0.95) -> float:
    """VaR varians-kovarians (asumsi normal): VaR = -mu + sigma * z_alpha."""
    r = np.asarray(returns, dtype=float)
    return float(-r.mean() + r.std(ddof=1) * stats.norm.ppf(alpha))


def var_cornish_fisher(returns: np.ndarray, alpha: float = 0.95) -> float:
    """VaR parametrik dengan koreksi skewness & kurtosis (ekspansi Cornish-Fisher)."""
    r = np.asarray(returns, dtype=float)
    z = stats.norm.ppf(1 - alpha)
    s = stats.skew(r)
    k = stats.kurtosis(r)  # excess kurtosis
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * k / 24
        - (2 * z**3 - 5 * z) * s**2 / 36
    )
    return float(-(r.mean() + r.std(ddof=1) * z_cf))


def var_monte_carlo(terminal_returns: np.ndarray, alpha: float = 0.95):
    """VaR & CVaR dari return terminal hasil simulasi Monte Carlo."""
    return var_historical(terminal_returns, alpha), expected_shortfall(terminal_returns, alpha)


def kupiec_pof(n_violations: int, n_obs: int, alpha: float = 0.95):
    """Uji Proportion-of-Failures Kupiec.

    H0: frekuensi pelanggaran VaR = 1 - alpha. LR ~ chi-square(1) di bawah H0.
    p-value kecil => model VaR tidak terkalibrasi dengan baik.
    """
    p = 1 - alpha
    x, n = n_violations, n_obs
    phat = x / n

    def loglik(prob: float) -> float:
        a = (n - x) * np.log(1 - prob) if (n - x) > 0 else 0.0
        b = x * np.log(prob) if x > 0 else 0.0
        return a + b

    lr = max(-2.0 * (loglik(p) - loglik(phat)), 0.0)
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def rolling_var_backtest(returns, alpha: float = 0.95, window: int = 250, method: str = "historical"):
    """Backtest VaR 1-hari dengan jendela bergulir + uji Kupiec."""
    r = np.asarray(returns, dtype=float)
    fn = {"historical": var_historical, "parametric": var_parametric,
          "cornish-fisher": var_cornish_fisher}[method]
    var_series = np.full(r.size, np.nan)
    for t in range(window, r.size):
        var_series[t] = fn(r[t - window : t], alpha)
    realized_loss = -r
    mask = ~np.isnan(var_series)
    violations = np.zeros(r.size, dtype=bool)
    violations[mask] = realized_loss[mask] > var_series[mask]
    n_obs, n_viol = int(mask.sum()), int(violations.sum())
    lr, pval = kupiec_pof(n_viol, n_obs, alpha)
    return {"var_series": var_series, "realized_loss": realized_loss, "violations": violations,
            "n_violations": n_viol, "n_obs": n_obs, "kupiec_lr": lr, "kupiec_p": pval}
