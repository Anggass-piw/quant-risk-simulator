"""Ukuran risiko: VaR (historis, parametrik, Cornish-Fisher, Monte Carlo), CVaR/ES, backtest Kupiec.

Konvensi: loss = -return. VaR dan CVaR dilaporkan sebagai angka positif
(fraksi dari nilai portofolio). VaR_alpha = kuantil ke-alpha dari distribusi loss.
"""

from __future__ import annotations

import operator

import numpy as np
from scipy import stats


def _validate_alpha(alpha: float) -> float:
    try:
        alpha = float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("alpha harus berupa angka di antara 0 dan 1.") from exc
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha harus berupa angka di antara 0 dan 1 (eksklusif).")
    return alpha


def _as_returns(returns, minimum: int = 1) -> np.ndarray:
    try:
        r = np.asarray(returns, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Return harus berupa array numerik satu dimensi.") from exc
    if r.ndim != 1:
        raise ValueError("Return harus berupa array satu dimensi.")
    if r.size < minimum:
        raise ValueError(f"Diperlukan minimal {minimum} observasi return; tersedia {r.size}.")
    if not np.isfinite(r).all():
        raise ValueError("Return tidak boleh mengandung NaN atau nilai tak hingga.")
    return r


def portfolio_returns(asset_returns: np.ndarray, weights) -> np.ndarray:
    return np.asarray(asset_returns, dtype=float) @ np.asarray(weights, dtype=float)


def var_historical(returns: np.ndarray, alpha: float = 0.95) -> float:
    """VaR historis: kuantil empiris dari loss."""
    r = _as_returns(returns)
    alpha = _validate_alpha(alpha)
    return float(np.quantile(-r, alpha))


def expected_shortfall(returns: np.ndarray, alpha: float = 0.95) -> float:
    """CVaR / Expected Shortfall: rata-rata loss yang melampaui VaR."""
    r = _as_returns(returns)
    alpha = _validate_alpha(alpha)
    losses = -r
    var = np.quantile(losses, alpha)
    return float(losses[losses >= var].mean())


def var_parametric(returns: np.ndarray, alpha: float = 0.95) -> float:
    """VaR varians-kovarians (asumsi normal): VaR = -mu + sigma * z_alpha."""
    r = _as_returns(returns, minimum=2)
    alpha = _validate_alpha(alpha)
    return float(-r.mean() + r.std(ddof=1) * stats.norm.ppf(alpha))


def var_cornish_fisher(returns: np.ndarray, alpha: float = 0.95) -> float:
    """VaR parametrik dengan koreksi skewness & kurtosis (ekspansi Cornish-Fisher)."""
    r = _as_returns(returns, minimum=3)
    alpha = _validate_alpha(alpha)
    if np.ptp(r) == 0:
        raise ValueError("VaR Cornish-Fisher memerlukan return yang bervariasi.")
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
    alpha = _validate_alpha(alpha)
    for value, name in ((n_violations, "n_violations"), (n_obs, "n_obs")):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} harus bilangan bulat.")
        try:
            if name == "n_violations":
                x = operator.index(value)
            else:
                n = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} harus bilangan bulat.") from exc

    if n <= 0:
        raise ValueError("n_obs harus lebih besar dari nol untuk uji Kupiec.")
    if x < 0 or x > n:
        raise ValueError("n_violations harus berada di antara 0 dan n_obs.")

    p = 1 - alpha
    phat = x / n

    def loglik(prob: float) -> float:
        a = (n - x) * np.log(1 - prob) if (n - x) > 0 else 0.0
        b = x * np.log(prob) if x > 0 else 0.0
        return a + b

    lr = max(-2.0 * (loglik(p) - loglik(phat)), 0.0)
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def rolling_var_backtest(returns, alpha: float = 0.95, window: int = 250, method: str = "historical"):
    """Backtest VaR 1-hari dengan jendela bergulir + uji Kupiec.

    Diperlukan setidaknya satu observasi out-of-sample, sehingga jumlah return
    harus lebih besar dari ``window``. Window parametrik perlu >= 2 observasi;
    Cornish-Fisher perlu >= 3.
    """
    r = _as_returns(returns)
    alpha = _validate_alpha(alpha)
    methods = {
        "historical": (var_historical, 1),
        "parametric": (var_parametric, 2),
        "cornish-fisher": (var_cornish_fisher, 3),
    }
    if method not in methods:
        raise ValueError(f"Metode backtest tidak dikenal: {method!r}.")
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)):
        raise ValueError("window harus bilangan bulat positif.")
    fn, minimum_window = methods[method]
    if window < minimum_window:
        raise ValueError(f"window untuk metode {method} harus minimal {minimum_window}.")
    if r.size <= window:
        raise ValueError(
            f"Backtest memerlukan lebih banyak return daripada window: tersedia {r.size}, "
            f"window {window} (minimal {window + 1} return)."
        )

    var_series = np.full(r.size, np.nan)
    for t in range(window, r.size):
        estimate = fn(r[t - window : t], alpha)
        if not np.isfinite(estimate):
            raise ValueError(f"Estimasi VaR tidak valid pada indeks {t}; periksa data dan window.")
        var_series[t] = estimate
    realized_loss = -r
    mask = ~np.isnan(var_series)
    violations = np.zeros(r.size, dtype=bool)
    violations[mask] = realized_loss[mask] > var_series[mask]
    n_obs, n_viol = int(mask.sum()), int(violations.sum())
    lr, pval = kupiec_pof(n_viol, n_obs, alpha)
    return {"var_series": var_series, "realized_loss": realized_loss, "violations": violations,
            "n_violations": n_viol, "n_obs": n_obs, "kupiec_lr": lr, "kupiec_p": pval}
