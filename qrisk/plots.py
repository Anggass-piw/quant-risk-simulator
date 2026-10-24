"""Visualisasi interaktif (Plotly), disimpan sebagai HTML."""

from __future__ import annotations

import numpy as np


def _go():
    import plotly.graph_objects as go
    return go


def fan_chart(portfolio_paths: np.ndarray, title: str = "Simulasi Nilai Portofolio (Monte Carlo GBM)"):
    """portfolio_paths: (n_paths, steps+1). Sampel lintasan + persentil 5/25/50/75/95."""
    go = _go()
    x = np.arange(portfolio_paths.shape[1])
    fig = go.Figure()
    for p in portfolio_paths[: min(60, len(portfolio_paths))]:
        fig.add_trace(go.Scatter(x=x, y=p, mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(width=0.6, color="rgba(100,100,100,0.25)")))
    q = np.percentile(portfolio_paths, [5, 25, 50, 75, 95], axis=0)
    for name, y, dash in zip(["P5", "P25", "Median", "P75", "P95"], q,
                             ["dot", "dash", "solid", "dash", "dot"]):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(width=2, dash=dash)))
    fig.update_layout(title=title, xaxis_title="Hari perdagangan", yaxis_title="Nilai portofolio",
                      template="plotly_white")
    return fig


def loss_distribution(terminal_returns, var: float, cvar: float, alpha: float,
                      title: str = "Distribusi Loss Terminal"):
    go = _go()
    losses = -np.asarray(terminal_returns) * 100
    fig = go.Figure(go.Histogram(x=losses, nbinsx=80, marker_color="steelblue", opacity=0.8))
    fig.add_vline(x=var * 100, line_dash="dash", line_color="orange",
                  annotation_text=f"VaR {alpha:.0%}: {var:.2%}")
    fig.add_vline(x=cvar * 100, line_dash="dash", line_color="red",
                  annotation_text=f"CVaR: {cvar:.2%}", annotation_position="top left")
    fig.update_layout(title=title, xaxis_title="Loss (% dari nilai awal)", yaxis_title="Frekuensi",
                      template="plotly_white")
    return fig


def backtest_chart(bt: dict, dates=None, title: str = "Backtest VaR 1 Hari"):
    go = _go()
    x = np.asarray(dates) if dates is not None else np.arange(len(bt["realized_loss"]))
    v = bt["violations"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=bt["realized_loss"] * 100, mode="lines", name="Loss realisasi (%)",
                             line=dict(width=1, color="gray")))
    fig.add_trace(go.Scatter(x=x, y=bt["var_series"] * 100, mode="lines", name="VaR (%)",
                             line=dict(width=2, color="orange")))
    fig.add_trace(go.Scatter(x=x[v], y=bt["realized_loss"][v] * 100, mode="markers", name="Pelanggaran",
                             marker=dict(color="red", size=6)))
    fig.update_layout(title=f"{title} (Kupiec p={bt['kupiec_p']:.3f})", yaxis_title="% loss",
                      template="plotly_white")
    return fig


def compare_methods_bar(results: dict, alpha: float):
    go = _go()
    fig = go.Figure(go.Bar(x=list(results.keys()), y=[v * 100 for v in results.values()],
                           marker_color="teal"))
    fig.update_layout(title=f"Perbandingan VaR {alpha:.0%} (1 hari)", yaxis_title="VaR (%)",
                      template="plotly_white")
    return fig
