from types import SimpleNamespace

import pandas as pd
import pytest

import main as cli


def test_cli_requires_three_price_observations_for_volatility(monkeypatch):
    prices = pd.DataFrame(
        {"ASSET": [100.0, 101.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="D"),
    )
    monkeypatch.setattr(cli, "parse_args", lambda: SimpleNamespace(source="csv", csv="unused", weights=None))
    monkeypatch.setattr(cli, "load_csv", lambda _path: prices)

    with pytest.raises(SystemExit, match="3 observasi harga"):
        cli.main()


def test_cli_reports_insufficient_returns_for_backtest_window(monkeypatch):
    prices = pd.DataFrame(
        {"ASSET": [100.0, 101.0, 99.0, 102.0]},
        index=pd.date_range("2025-01-01", periods=4, freq="D"),
    )
    args = SimpleNamespace(source="csv", csv="unused", weights=None, window=3)
    monkeypatch.setattr(cli, "parse_args", lambda: args)
    monkeypatch.setattr(cli, "load_csv", lambda _path: prices)

    with pytest.raises(SystemExit, match="minimal 4 return"):
        cli.main()
