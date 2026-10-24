"""CLI Quant Risk & Volatility Simulator.

Contoh:
    python main.py                                   # data sintetis, offline
    python main.py --source yahoo --tickers BTC-USD BBCA.JK AAPL --start 2021-01-01
    python main.py --source csv --csv harga.csv --weights 0.5 0.3 0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qrisk import (estimate_gbm_params, expected_shortfall, rolling_var_backtest,
                   simulate_gbm, var_cornish_fisher, var_historical, var_monte_carlo,
                   var_parametric)
from qrisk.data import load_csv, load_yahoo, synthetic_prices, to_log_returns
from qrisk.risk import portfolio_returns


def parse_args():
    p = argparse.ArgumentParser(description="Simulator risiko portofolio: GBM + Monte Carlo + VaR/CVaR")
    p.add_argument("--source", choices=["synthetic", "yahoo", "csv"], default="synthetic")
    p.add_argument("--tickers", nargs="+", default=["BTC-USD", "AAPL"])
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--csv", default=None)
    p.add_argument("--weights", nargs="+", type=float, default=None, help="default: bobot sama rata")
    p.add_argument("--value", type=float, default=100_000_000, help="nilai portofolio awal (mis. Rp)")
    p.add_argument("--horizon", type=int, default=21, help="horizon simulasi (hari perdagangan)")
    p.add_argument("--paths", type=int, default=20_000)
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--window", type=int, default=250, help="jendela backtest")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="reports")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.source == "yahoo":
        prices = load_yahoo(a.tickers, a.start)
    elif a.source == "csv":
        prices = load_csv(a.csv)
    else:
        prices = synthetic_prices(seed=a.seed)

    n = prices.shape[1]
    w = np.full(n, 1 / n) if a.weights is None else np.asarray(a.weights)
    if len(w) != n or not np.isclose(w.sum(), 1.0):
        raise SystemExit(f"Bobot harus {n} angka dan berjumlah 1 (sekarang {w.sum():.3f}).")

    log_r = to_log_returns(prices)
    simple_r = np.expm1(log_r.values)
    port_r = portfolio_returns(simple_r, w)  # return harian portofolio (rebalancing harian)

    # --- VaR 1 hari dari data historis, 3 metode ---
    one_day = {
        "Historis": var_historical(port_r, a.alpha),
        "Parametrik (Normal)": var_parametric(port_r, a.alpha),
        "Cornish-Fisher": var_cornish_fisher(port_r, a.alpha),
    }

    # --- Monte Carlo GBM multi-aset ---
    mu, sigma, corr = estimate_gbm_params(log_r.values)
    paths = simulate_gbm(prices.iloc[-1].values, mu, sigma, corr, horizon_days=a.horizon,
                         n_paths=a.paths, seed=a.seed)  # (P, T+1, N)
    units = a.value * w / prices.iloc[-1].values  # buy-and-hold selama horizon
    pv = (paths * units).sum(axis=2)  # (P, T+1)
    term_ret = pv[:, -1] / a.value - 1
    mc_var, mc_cvar = var_monte_carlo(term_ret, a.alpha)

    # --- Backtest ---
    bt = rolling_var_backtest(port_r, a.alpha, a.window, "historical")

    # --- Laporan konsol ---
    print(f"\n=== Quant Risk Simulator | {n} aset | {len(port_r)} hari data ===")
    print(pd.DataFrame({"mu (thn)": mu, "sigma (thn)": sigma, "bobot": w}, index=prices.columns).round(4))
    print("\nKorelasi:\n", pd.DataFrame(corr, index=prices.columns, columns=prices.columns).round(2))
    print(f"\nVaR {a.alpha:.0%} 1 hari (nilai {a.value:,.0f}):")
    for k, v in one_day.items():
        print(f"  {k:<22}{v:8.2%}  ->  {v * a.value:>16,.0f}")
    print(f"\nMonte Carlo ({a.paths:,} lintasan, horizon {a.horizon} hari):")
    print(f"  VaR {a.alpha:.0%}  : {mc_var:8.2%}  ->  {mc_var * a.value:>16,.0f}")
    print(f"  CVaR {a.alpha:.0%} : {mc_cvar:8.2%}  ->  {mc_cvar * a.value:>16,.0f}")
    print(f"\nBacktest VaR historis: {bt['n_violations']} pelanggaran dari {bt['n_obs']} hari "
          f"(ekspektasi {(1 - a.alpha) * bt['n_obs']:.1f}); Kupiec LR={bt['kupiec_lr']:.2f}, "
          f"p={bt['kupiec_p']:.3f} -> {'MODEL DITOLAK' if bt['kupiec_p'] < 0.05 else 'model dapat diterima'}")

    if not a.no_plots:
        from qrisk import plots
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        plots.fan_chart(pv).write_html(out / "fan_chart.html")
        plots.loss_distribution(term_ret, mc_var, mc_cvar, a.alpha).write_html(out / "loss_dist.html")
        plots.backtest_chart(bt, dates=log_r.index.values).write_html(out / "backtest.html")
        plots.compare_methods_bar(one_day, a.alpha).write_html(out / "compare_var.html")
        print(f"\nGrafik interaktif disimpan di: {out}/")


if __name__ == "__main__":
    main()
