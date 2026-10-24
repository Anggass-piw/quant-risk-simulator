"""Sumber data: Yahoo Finance (opsional), CSV, atau data sintetis agar proyek jalan offline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .gbm import simulate_gbm


def load_yahoo(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Unduh harga penutupan via yfinance. Contoh ticker: BTC-USD, BBCA.JK, AAPL."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install yfinance dulu: pip install yfinance") from exc
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    return df.dropna()


def load_csv(path: str) -> pd.DataFrame:
    """CSV: kolom pertama tanggal (index), satu kolom harga per aset."""
    return pd.read_csv(path, index_col=0, parse_dates=True).dropna()


def synthetic_prices(n_days: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Data sintetis 3 aset berkorelasi. Return kripto dibuat fat-tailed (Student-t)
    supaya beda VaR normal vs historis terlihat."""
    rng = np.random.default_rng(seed)
    corr = np.array([[1.0, 0.45, -0.10], [0.45, 1.0, 0.0], [-0.10, 0.0, 1.0]])
    paths = simulate_gbm([100, 100, 100], [0.10, 0.35, 0.04], [0.20, 0.70, 0.06], corr,
                         horizon_days=n_days, n_paths=1, antithetic=False, seed=seed)[0]
    log_r = np.diff(np.log(paths), axis=0)
    shocks = rng.standard_t(df=3, size=n_days) / np.sqrt(3.0)  # varians = 1
    log_r[:, 1] = 0.5 * log_r[:, 1] + 0.5 * (0.70 / np.sqrt(252)) * shocks
    prices = 100 * np.exp(np.vstack([np.zeros(3), np.cumsum(log_r, axis=0)]))
    idx = pd.bdate_range(start="2020-01-01", periods=len(prices))
    return pd.DataFrame(prices, index=idx, columns=["SAHAM", "KRIPTO", "OBLIGASI"])


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()
