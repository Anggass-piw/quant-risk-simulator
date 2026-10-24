"""Geometric Brownian Motion (GBM) multi-aset dengan korelasi.

Model:      dS_i = mu_i S_i dt + sigma_i S_i dW_i,   corr(dW_i, dW_j) = rho_ij
Solusi:     S_i(t+dt) = S_i(t) * exp[(mu_i - sigma_i^2 / 2) dt + sigma_i sqrt(dt) Z_i]
dengan Z = L @ eps, eps ~ N(0, I), dan L adalah faktor Cholesky dari matriks korelasi.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def estimate_gbm_params(log_returns: np.ndarray, periods_per_year: int = TRADING_DAYS):
    """Estimasi parameter GBM tahunan dari log-return historis.

    Returns
    -------
    mu : (N,) drift tahunan. Dari E[r] = (mu - sigma^2/2) dt  =>  mu = E[r]/dt + sigma^2/2
    sigma : (N,) volatilitas tahunan = std(r) / sqrt(dt)
    corr : (N, N) matriks korelasi
    """
    r = np.asarray(log_returns, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    dt = 1.0 / periods_per_year
    sigma = r.std(axis=0, ddof=1) / np.sqrt(dt)
    mu = r.mean(axis=0) / dt + 0.5 * sigma**2
    corr = np.ones((1, 1)) if r.shape[1] == 1 else np.corrcoef(r, rowvar=False)
    return mu, sigma, corr


def simulate_gbm(
    s0,
    mu,
    sigma,
    corr=None,
    horizon_days: int = 252,
    n_paths: int = 10_000,
    periods_per_year: int = TRADING_DAYS,
    antithetic: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Simulasi Monte Carlo lintasan harga GBM.

    Returns
    -------
    paths : array (n_paths, horizon_days + 1, N); indeks waktu 0 = harga awal.
    """
    s0 = np.atleast_1d(np.asarray(s0, dtype=float))
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    n_assets = s0.size
    corr = np.eye(n_assets) if corr is None else np.asarray(corr, dtype=float)

    dt = 1.0 / periods_per_year
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(corr)

    if antithetic:
        half = (n_paths + 1) // 2
        eps = rng.standard_normal((half, horizon_days, n_assets))
        eps = np.concatenate([eps, -eps], axis=0)[:n_paths]
    else:
        eps = rng.standard_normal((n_paths, horizon_days, n_assets))

    z = eps @ chol.T  # korelasi antar aset
    log_inc = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(log_inc, axis=1)
    log_paths = np.concatenate([np.zeros((n_paths, 1, n_assets)), log_paths], axis=1)
    return s0 * np.exp(log_paths)
