import numpy as np
import pandas as pd
import pytest

from qrisk.data import load_csv, to_log_returns


def _prices(values=(100.0, 101.0, 102.0)):
    return pd.DataFrame(
        {"ASSET": values},
        index=pd.date_range("2025-01-01", periods=len(values), freq="D"),
    )


def test_to_log_returns_computes_returns_for_valid_prices():
    result = to_log_returns(_prices())
    assert len(result) == 2
    assert np.all(np.isfinite(result.to_numpy()))
    assert result.iloc[0, 0] == pytest.approx(np.log(101 / 100))


@pytest.mark.parametrize("bad_price", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_to_log_returns_rejects_nonpositive_or_nonfinite_prices(bad_price):
    prices = _prices((100.0, bad_price, 102.0))
    with pytest.raises(ValueError, match="lebih besar dari nol|NaN|tak hingga"):
        to_log_returns(prices)


def test_to_log_returns_rejects_nonnumeric_prices():
    prices = _prices((100.0, 101.0, 102.0)).astype(object)
    prices.iloc[1, 0] = "bukan-harga"
    with pytest.raises(ValueError, match="berupa angka"):
        to_log_returns(prices)


def test_to_log_returns_requires_two_price_observations():
    with pytest.raises(ValueError, match="minimal 2 observasi"):
        to_log_returns(_prices((100.0,)))


def test_to_log_returns_rejects_duplicate_or_unsorted_dates():
    duplicate = _prices()
    duplicate.index = pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-03"])
    with pytest.raises(ValueError, match="duplikat"):
        to_log_returns(duplicate)

    unsorted = _prices().iloc[[1, 0, 2]]
    with pytest.raises(ValueError, match="berurutan naik"):
        to_log_returns(unsorted)


def test_load_csv_rejects_zero_and_negative_prices(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("date,ASSET\n2025-01-01,100\n2025-01-02,0\n")
    with pytest.raises(ValueError, match="lebih besar dari nol"):
        load_csv(str(path))


def test_load_csv_requires_a_path():
    with pytest.raises(ValueError, match="Path CSV wajib diisi"):
        load_csv("")
