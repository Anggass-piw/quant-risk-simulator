# Quant Risk & Volatility Simulator

Simulator risiko portofolio (saham / kripto) berbasis **Geometric Brownian Motion**, **Monte Carlo**, dan **Value at Risk (VaR / CVaR)**, lengkap dengan backtest statistik (uji Kupiec) dan grafik interaktif Plotly.

## Fitur

- **GBM multi-aset berkorelasi** (dekomposisi Cholesky) + *antithetic variates* untuk reduksi varians
- **Estimasi parameter** drift (μ), volatilitas (σ), dan korelasi dari data historis
- **4 metode VaR**: Historis, Parametrik (Normal), Cornish-Fisher (koreksi skewness/kurtosis), Monte Carlo
- **CVaR / Expected Shortfall**
- **Backtest rolling-window** + **uji Kupiec POF** (chi-kuadrat) untuk menguji kalibrasi model
- Sumber data: **Yahoo Finance**, **CSV**, atau **data sintetis** (jalan offline)
- Grafik interaktif: fan chart, distribusi loss, backtest, perbandingan metode

## Landasan Matematika

**Model harga (SDE):**  dS = μ S dt + σ S dW

**Solusi eksak (Itô):**  S(t+Δt) = S(t) · exp[(μ − σ²/2)Δt + σ√Δt · Z],  Z ~ N(0,1)

Koreksi −σ²/2 muncul dari lemma Itô; sehingga E[S_T] = S₀ e^{μT}.

**Multi-aset:** Z = L·ε, dengan ε ~ N(0, I) dan LLᵀ = matriks korelasi ρ.

**Estimasi dari log-return harian r (Δt = 1/252):**
σ̂ = std(r)/√Δt,  μ̂ = mean(r)/Δt + σ̂²/2

**VaR dan CVaR** (loss = −return):
- VaR_α = F⁻¹_L(α)
- CVaR_α = E[L | L ≥ VaR_α]
- Parametrik: VaR = −μ + σ·z_α
- Cornish-Fisher: z_cf = z + (z²−1)S/6 + (z³−3z)K/24 − (2z³−5z)S²/36

**Uji Kupiec (POF):** H₀: laju pelanggaran = 1−α.
LR = −2 ln[ (1−p)^(n−x) p^x / ((1−x/n)^(n−x) (x/n)^x) ] ~ χ²(1)

## Instalasi

```bash
git clone https://github.com/<username>/quant-risk-simulator.git
cd quant-risk-simulator
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Penggunaan

```bash
# Data sintetis (offline)
python main.py

# Data nyata dari Yahoo Finance
python main.py --source yahoo --tickers BTC-USD BBCA.JK AAPL --start 2021-01-01 \
               --weights 0.3 0.4 0.3 --value 100000000 --horizon 21 --alpha 0.99

# Dari CSV (kolom pertama tanggal, kolom lain harga per aset)
python main.py --source csv --csv harga.csv
```

Output: ringkasan di terminal + file HTML interaktif di folder `reports/`.

## Contoh Output (data sintetis)

```
VaR 95% 1 hari (nilai 100,000,000):
  Historis                 1.98%
  Parametrik (Normal)      2.05%
  Cornish-Fisher           1.97%

Monte Carlo (20,000 lintasan, horizon 21 hari):
  VaR 95%  :    8.45%
  CVaR 95% :   10.48%

Backtest VaR historis: 64 pelanggaran dari 1250 hari (ekspektasi 62.5); Kupiec p=0.846
```

## Struktur Proyek

```
qrisk/
  gbm.py     # simulasi GBM + estimasi parameter
  risk.py    # VaR, CVaR, Cornish-Fisher, Kupiec, backtest
  data.py    # Yahoo / CSV / data sintetis
  plots.py   # grafik Plotly
main.py      # CLI
tests/       # tes unit (pytest)
```

## Tes

```bash
pytest -q
```

Tes memverifikasi, antara lain: E[S_T] = S₀e^{μT}, std log-return = σ, korelasi antar aset terjaga, VaR parametrik = rumus tertutup, CVaR > VaR, dan uji Kupiec.

## Keterbatasan

- GBM mengasumsikan volatilitas konstan dan return log-normal; pasar nyata punya *fat tails* dan *volatility clustering*
- Portofolio disimulasikan *buy-and-hold* selama horizon
- Bukan saran investasi

## Rencana Pengembangan

- [ ] GARCH(1,1) untuk volatilitas stokastik
- [ ] Jump-diffusion (Merton) untuk lonjakan harga kripto
- [ ] Simulasi dengan copula t / korelasi dinamis
- [ ] Dashboard Streamlit

## Lisensi

MIT
