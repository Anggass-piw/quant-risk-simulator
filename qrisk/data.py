"""Sumber data: Yahoo Finance (opsional), CSV, atau data sintetis agar proyek jalan offline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .gbm import simulate_gbm


def _validate_prices(prices: pd.DataFrame, source: str) -> pd.DataFrame:
    """Pastikan data harga terurut, numerik, lengkap, dan valid untuk log-return."""
    if not isinstance(prices, pd.DataFrame):
        raise ValueError(f"{source}: data harga harus berupa DataFrame.")
    if prices.empty or prices.shape[1] == 0:
        raise ValueError(f"{source}: tidak ada data harga yang tersedia.")
    if not prices.columns.is_unique:
        raise ValueError(f"{source}: nama kolom aset harus unik.")
    if not prices.index.is_unique:
        raise ValueError(f"{source}: tanggal observasi duplikat.")
    if not prices.index.is_monotonic_increasing:
        raise ValueError(f"{source}: tanggal observasi harus berurutan naik.")
    if prices.index.hasnans:
        raise ValueError(f"{source}: tanggal observasi tidak boleh kosong.")

    try:
        numeric = prices.apply(pd.to_numeric, errors="raise")
        values = numeric.to_numpy(dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{source}: semua harga harus berupa angka.") from exc

    if not np.isfinite(values).all():
        raise ValueError(f"{source}: harga tidak boleh kosong, NaN, atau tak hingga.")
    if (values <= 0).any():
        raise ValueError(f"{source}: semua harga harus lebih besar dari nol.")
    return numeric


def load_yahoo(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Unduh harga penutupan via yfinance. Contoh ticker: BTC-USD, BBCA.JK, AAPL."""
    if not tickers:
        raise ValueError("Yahoo Finance: setidaknya satu ticker harus diberikan.")
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install yfinance dulu: pip install yfinance") from exc

    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    return _validate_prices(df.dropna(), "Yahoo Finance")


def load_csv(path: str) -> pd.DataFrame:
    """CSV: kolom pertama tanggal (index), satu kolom harga per aset."""
    if not path:
        raise ValueError("Path CSV wajib diisi untuk sumber data CSV.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("CSV: kolom pertama harus berisi tanggal yang valid.")
    return _validate_prices(df.dropna(), "CSV")


def synthetic_prices(n_days: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Data sintetis 3 aset berkorelasi. Return kripto dibuat fat-tailed (Student-t)
    supaya beda VaR normal vs historis terlihat."""
    if isinstance(n_days, bool) or not isinstance(n_days, (int, np.integer)) or n_days < 1:
        raise ValueError("n_days harus bilangan bulat positif.")

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
    """Ubah harga valid menjadi log-return; perlu minimal dua observasi harga."""
    validated = _validate_prices(prices, "Data harga")
    if len(validated) < 2:
        raise ValueError("Data harga perlu minimal 2 observasi untuk menghitung return.")
    return np.log(validated).diff().iloc[1:]
