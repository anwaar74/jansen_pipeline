"""
========================================================================
Shariah ML Pipeline v4.9 (reverted from v4.11) — Jansen MFAT 2nd Ed. Aligned
========================================================================
Markets:        US (NYSE/NASDAQ) · Malaysia (KLSE) · Hong Kong (HKEX)
Alpha engine:   LightGBM ensemble (21d target × 3 seeds)
Target:         Cross-sectional RANK of forward return     (Jansen Ch.4/12)
Features:       Cross-sectional RANK per date              (Jansen Ch.4)
Selection:      Tree gain importance — no Lasso pre-screen (Jansen Ch.6)
Backtest:       Long-short EW top-quintile, EMA-smoothed,
                vol-targeted (de-lever only)                (Jansen Ch.12)
Liquidity:      Per-market ADV thresholds
Survivorship:   Includes EODHD delisted stocks

v2 → v3 (Sharpe -0.20 → +0.10):
  Target → cross-sectional rank per date  (was raw 21d return)
  Features → rank-transformed per date    (was raw, regime-biased)
  Dropped LassoCV pre-selection (was killing 16/17 features)
  Backtest → long-short market-neutral    (was long-only, all-US)
  Added Jansen Ch.4 alphas: 12-1 momentum, short-term reversal,
   idiosyncratic vol, volume shock, drawdown depth, Amihud illiquidity

v3 → v4 attempt (Sharpe +0.10 → -0.65 — REGRESSED):
  Cell 11: 5d×21d horizon ensemble       → 5d disagreed with 21d on
                                            cross-section levels, INVERTED
                                            the long-short (cum_top 1.46 <
                                            cum_bottom 2.99 vs v3 spread +0.20)
  Cell 13: 30% basket + signal-weighted  → pulled in middle-quantile noise
  Cell 13: MAX_LEVERAGE 1.5×             → amplified bad-signal periods

v4 → v4.1 (rollback to conservative ground, keep wins):
  Cell 11: Drop 5d horizon. Single 21d × 3 seeds.
  Cell 13: Revert to 20% basket, equal-weight positions, MAX_LEVERAGE 1.0.
  KEEP   : EMA smoothing α=0.5 (turnover 75%→60%, autocorr 0.45→0.63 — works)
  KEEP   : Vol target de-lever (avg lev 0.87 in v4 — actively reducing risk)

v4.1 → v4.2 attempt (Sharpe +0.10 → -0.17 — ROLLED BACK):
  Cell 8 : Target switched to market-NEUTRALISED rank. Per-fold IC
           improved (0.046 → 0.052, ICIR 0.65 → 0.73) BUT long-short
           spread collapsed to +0.005 (D10 +2.35% vs D1 +2.45%). Model
           learned "beat the market" but backtest trades absolute returns
           and basket is 100% US. target_demeaned and target_demeaned_rank
           stay in fd2 for a future per-market backtest variant.

v4.2 → v4.3 (calibrate the signal POST-HOC, not at the target):
  Cell 13: Decile-return CALIBRATION. Across v3/v4/v4.2, D3-D6 (middle
           deciles) consistently beat D10 (long) and D1 (short) — the
           model's "best" prediction was empirically not its best decile.
           Each rebalance: bucket signal into 10 deciles, look up the
           past 6 rebalances' realised return PER PREDICTED DECILE, long
           the top-2 expected-return deciles, short the bottom-2.
           Adapts to whatever non-monotonic shape the model has, no
           hardcoding required. Warm-up uses standard top/bottom quintile.
           Result: Sharpe +0.10 → +0.68, return 0.5% → 30.5%, DD -17% → -11%.

v4.3 + (tighten vol target — realised vol was 53% on initial v4.3 run):
  Cell 13: MIN_LEVERAGE 0.30 → 0.10  (let the overlay deflate harder)
  Cell 13: VOL_TARGET_ANN 0.10 → 0.12
           Result: barely moved (53% → 53% realised vol, lev 0.80 → 0.78,
                   Sharpe 0.68 → 0.69). Overlay's rolling lookback sees
                   moderate vol; the 53% comes from outlier rebals it
                   can't pre-empt. Kept for safety; vol reduction must
                   now come from basket width, not the overlay.

v4.3+ → v4.4 (attack per-fold IC bimodality with regime awareness):
  Cell 6 : 4 NEW regime features per (date, market), as 252-day rolling
           z-scores: market_vol_21d, breadth (% above 50d SMA),
           cross-sectional dispersion, trend autocorr (63d).
  Cell 7 : Split FEATURE_COLS into STOCK_FEATURE_COLS + REGIME_FEATURE_COLS
           so we can treat them differently at rank time.
  Cell 8 : Stock features get per-date rank (as before).  Regime features
           kept as RAW z-scores (per-date ranking would collapse them
           since they're constant across stocks within a date).
  Cell 11: Exponential RECENCY sample weights, half-life 252 trading days.
           Recent training samples dominate so the model adapts to the
           current regime rather than averaging across 3 years equally.
           Targets F1/F2/F5 negative-IC folds directly (Jansen Ch.6).
  Result: ICIR 0.65 → 0.71, pooled OOF IC -0.0044 → +0.0061 (FIRST TIME
          positive!), win rate 50 → 63%, realised vol 53% → 14.2%, D10 went
          from worst (2.3%) to best (+14.7%). Model is now monotonic.
          BUT Sharpe 0.69 → 0.00 because decile calibration was still on,
          and a calibration designed for a broken model is adversarial
          against a fixed one. See v4.5 fix below.

v4.4 → v4.5 (turn calibration off — model no longer needs it):
  Cell 13: DECILE_CALIB = False. Vanilla top-quintile / bottom-quintile EW.
  Result: Sharpe 0.00 → 0.61, return -1.1% → 23.9%. Long-short.
          BUT v4.5's own long-only EW baseline = 0.88. So long-short cost
          us 0.27 Sharpe. Diagnosis: D1 = +0.29% (predicted worst still
          rises, just less). Shorting it loses 0.29%/rebal + costs.

v4.5 → v4.6 (drop the short book — pure drag on monotonic + positive IC):
  Cell 13: LONG_ONLY = True.
  Result: Sharpe 0.61 → 0.93 (gross 0.98), return 23.9% → 57.9%,
          DD -20.2% → -15.0%, win rate 50% → 58%.
          Beat own EW baseline (0.88). Sortino 5.5, Calmar 3.86.

v4.6 → v4.7 (concentrate basket on D10 via signal-weighting):
  Cell 13: SIGNAL_WEIGHTED = True.
  Result: Sharpe 0.93 → 0.85 (down) BUT return 57.9% → 96.2%,
          Sortino 5.5 → 9.42, Calmar 3.86 → 6.66, DD -15% → -14.4%.
          Vol blew from 64% to 125% — all upside. Sharpe-as-defined
          penalised the upside vol; risk-adjusted on Sortino/Calmar
          v4.7 is dramatically better than v4.6.

v4.7 → v4.8 (let vol-target overlay deflate harder):
  Cell 13: MIN_LEVERAGE 0.10 → 0.03.
  Result: NULL — Sharpe/vol/everything unchanged. The overlay's 6-period
          rolling lookback never asked for lev below 0.10 anyway; the 125%
          headline vol comes from per-period spikes the overlay can't see
          in advance, not from clamped deflation. Lever exhausted.

v4.8 → v4.9 (try per-market geographic diversification):
  Cell 13: PER_MARKET_BACKTEST = True.
  Result: Sharpe 0.85 → 0.90, return 95.7% → 70.8%, DD -14.4% → -8.2%
          (best DD ever), Calmar 6.66 → 8.66 (best ever), Win 63% → 67%.
          Real diversification reduced vol 125% → 82.8%. Sharpe modest
          lift because MY/HK signal weaker than US, equal-weight average
          dilutes the strong US alpha.

v4.9 → v4.10 (Kelly criterion sizing on the top-decile basket):
  Cell 13: KELLY_CRITERION = True (replaces vol-target as leverage decision).
           Tracks unlevered basket return history. Leverage =
           KELLY_FRACTION × (rolling mu / rolling sigma^2). Quarter-Kelly
           (0.25) used for safety; full Kelly is famously aggressive and
           blows up on over-estimated mu. Caps: [0.03, 1.5] leverage band.
           Kelly should lever UP when the basket has high Sharpe-equivalent
           (mu high, vol low) and DOWN when it doesn't — adaptive position
           sizing on the actual asset characteristics, not realised P&L.
           Falls back to vol-target during 8-rebal warm-up.

Run end-to-end:    python jansen_pipeline_v3.py
"""

# ═══════════════════════════════════════════════════════════════════════
# CELL 1 — Auto-install dependencies
# ═══════════════════════════════════════════════════════════════════════
import subprocess, sys

PKGS = [
    "pyarrow", "alphalens-reloaded", "lightgbm", "requests",
    "scikit-learn", "pandas", "numpy", "shap", "matplotlib",
    "scipy", "tqdm",
]
for pkg in PKGS:
    try:
        __import__(pkg.replace('-', '_').split('==')[0])
    except ImportError:
        print(f'Installing {pkg} ...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                               pkg, '--quiet', '--break-system-packages'])


# ═══════════════════════════════════════════════════════════════════════
# CELL 2 — Configuration
# ═══════════════════════════════════════════════════════════════════════
import os, pathlib
import numpy as np
import pandas as pd

BASE        = pathlib.Path(r'C:/Users/pc/Documents/Quant Series 2026/ml_stefan_jansen3')
EODHD_KEY   = os.environ.get('EODHD_API_KEY', '6a01d9bb03ae95.33277051')

RAW_PARQUET  = BASE / 'raw_ohlcv.parquet'
FEAT_PARQUET = BASE / 'features_long_v4.parquet'         # v4.4 adds regime features

# ── Universe
TRAIN_START = '2010-01-01'
TRAIN_END   = '2024-12-31'

# ── Walk-forward CV  (Jansen Ch.7)
TARGET_HORIZON = 21
EMBARGO        = 21
N_SPLITS       = 8
TRAIN_PERIOD   = 756
TEST_PERIOD    = 63

# ── Liquidity (ADV in local currency, 30-day)
LIQUIDITY = {'US': 500_000, 'MY': 200_000, 'HK': 500_000}

# ── Survivorship
INCLUDE_DELISTED = True
MAX_DELISTED     = 300

print('Config loaded')
print(f'  BASE = {BASE}')
print(f'  RAW_PARQUET  exists: {RAW_PARQUET.exists()}')
print(f'  FEAT_PARQUET exists: {FEAT_PARQUET.exists()}')


# ═══════════════════════════════════════════════════════════════════════
# CELL 3 — Shariah-screened ticker universe
# ═══════════════════════════════════════════════════════════════════════
# (Truncated for readability — see v2 notebook for full lists; identical here)
US_TICKERS = [
    'NVDA','AAPL','MSFT','GOOGL','AVGO','TSLA','XOM','LLY','JNJ','MU',
    'ABBV','PG','AMD','HD','CSCO','MRK','AMAT','LRCX','ORCL','GEV',
    'LIN','PEP','KLAC','ABT','TMO','TJX','CRM','TXN','COP','GILD',
    'ISRG','ADI','UBER','UNP','WELL','QCOM','BKNG','LOW','ANET','PLD',
    'ACN','DHR','MDT','NEM','SYK','VRTX','GLW','MCK','BSX','NOW',
    'ADBE','PANW','VRT','CEG','CRWD','EQIX','TT','SNDK','PWR','FCX',
    'STX','SLB','EOG','WM','JCI','MMM','ORLY','CSX','MDLZ','REGN',
    'CDNS','SHW','CMI','ROST','UPS','CL','SNPS','EMR','ITW','ECL',
    'NSC','APD','BKR','TEL','NKE','COR','CIEN','CTAS','CTVA','DASH',
    'TGT','FAST','MNST','TER','AZO','ADSK','FTNT','NXPI','CAH','MPWR',
    'BDX','IDXX','GWW','EW','COHR','FIX','RSG','CARR','EBAY','ROK',
    'NUE','WAB','HAL','GRMN','ODFL','ROP','DHI','MLM','VMC','KMB',
    'MCHP','DVN','KVUE','ADM','TPL','A','CTSH','GEHC','RMD','HSY',
    'IR','OTIS','TPR','CTRA','JBL','DOV','EXPE','XYL','CPRT','BIIB',
    'WDAY','WAT','TSCO','CHD','ON','PPG','HUBB','DXCM','STLD','NTAP',
    'VRSN','ULTA','ALB','CF','STE','VLTO','FICO','WSM','AVB','FSLR',
    'EFX','PHM','LH','MTD','PKG','WY','CHRW','EXPD','DD','AKAM',
    'EL','CDW','FFIV','WST','MAA','DECK','LULU','GPC','PNR','JBHT',
    'TRMB','INCY','NDSN','ROL','PTC','PODD','COO','HOLX','GNRC','RL',
    'IEX','ALGN','MAS','SMCI','AVY','ALLE','BBY','CLX','LII','IT',
    'GDDY','MKC','TYL','ZBRA','SWKS','BLDR','TECH','CPT','RVTY','TTD',
    'AOS','CRL','EPAM','POOL',
]
# NOTE: MY_TICKERS and HK_TICKERS lists are identical to v2 — load them
# from v2's notebook or define here if running standalone. Omitted to keep
# the file readable; uncomment the import below if you save them separately.
# from tickers_v2 import MY_TICKERS, HK_TICKERS

MY_TICKERS: list[str] = []
HK_TICKERS: list[str] = []
# NOTE: MY/HK lists omitted here to keep the file readable.
# If raw_ohlcv.parquet already exists, the universe is read from the parquet —
# the lists below are only used when re-downloading.

if RAW_PARQUET.exists() and not (MY_TICKERS or HK_TICKERS):
    _raw_meta = pd.read_parquet(RAW_PARQUET, columns=['ticker','market']).drop_duplicates()
    MY_TICKERS = sorted(_raw_meta[_raw_meta['market'] == 'MY']['ticker'].tolist())
    HK_TICKERS = sorted(_raw_meta[_raw_meta['market'] == 'HK']['ticker'].tolist())
    print(f'  → Loaded MY ({len(MY_TICKERS)}) and HK ({len(HK_TICKERS)}) tickers from cached parquet.')

MARKET_MAP = {t: 'US' for t in US_TICKERS}
MARKET_MAP.update({t: 'MY' for t in MY_TICKERS})
MARKET_MAP.update({t: 'HK' for t in HK_TICKERS})
ALL_TICKERS = US_TICKERS + MY_TICKERS + HK_TICKERS
print(f'Universe: {len(ALL_TICKERS)} tickers  '
      f'(US={len(US_TICKERS)}, MY={len(MY_TICKERS)}, HK={len(HK_TICKERS)})')


# ═══════════════════════════════════════════════════════════════════════
# CELL 4 — EODHD downloader
# ═══════════════════════════════════════════════════════════════════════
import requests, time

EOD_BASE = 'https://eodhd.com/api'

def fetch_eod(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    sym, exch = (ticker.split('.') + ['US'])[:2]
    url = (f'{EOD_BASE}/eod/{sym}.{exch}'
           f'?api_token={EODHD_KEY}&fmt=json&from={start}&to={end}&period=d')
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30); r.raise_for_status()
            data = r.json()
            if not data: return pd.DataFrame()
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            if 'adjusted_close' in df.columns and 'close' in df.columns:
                ratio = df['adjusted_close'] / df['close'].replace(0, np.nan)
                for col in ['open','high','low']:
                    df[col] = df[col] * ratio
                df['close'] = df['adjusted_close']
            return df[['open','high','low','close','volume']].copy()
        except Exception as e:
            if attempt < retries - 1: time.sleep(2 ** attempt)
            else: print(f'  [WARN] {ticker}: {e}'); return pd.DataFrame()

EXCHANGE_CODES = {'US':'US','MY':'KLSE','HK':'HK'}

def get_delisted(market: str, max_stocks: int = 300) -> list[str]:
    exch = EXCHANGE_CODES.get(market, market)
    url  = (f'{EOD_BASE}/exchange-symbol-list/{exch}'
            f'?api_token={EODHD_KEY}&fmt=json&delisted=1')
    try:
        r = requests.get(url, timeout=60); r.raise_for_status()
        data = r.json()
        tickers = [f"{row['Code']}.{exch}" for row in data
                   if row.get('Type') in ('Common Stock','ETF', None)]
        print(f'  {market}: {len(tickers)} delisted found, capping at {max_stocks}')
        return tickers[:max_stocks]
    except Exception as e:
        print(f'  [WARN] delisted {market}: {e}'); return []


# ═══════════════════════════════════════════════════════════════════════
# CELL 5 — Download raw OHLCV (skips if RAW_PARQUET exists)
# ═══════════════════════════════════════════════════════════════════════
FORCE_DOWNLOAD = False

if RAW_PARQUET.exists() and not FORCE_DOWNLOAD:
    print(f'RAW_PARQUET already exists ({RAW_PARQUET.stat().st_size/1e6:.1f} MB). Skipping download.')
    raw = pd.read_parquet(RAW_PARQUET)
else:
    print(f'Downloading {TRAIN_START} → {TRAIN_END} ...')
    live_tickers = list(ALL_TICKERS)
    delisted_tickers = []
    if INCLUDE_DELISTED:
        for mkt in ['US','MY','HK']:
            delisted_tickers.extend(get_delisted(mkt, MAX_DELISTED))
    all_download = live_tickers + [t for t in delisted_tickers if t not in live_tickers]
    frames = []
    for i, ticker in enumerate(all_download):
        df = fetch_eod(ticker, TRAIN_START, TRAIN_END)
        if df.empty: continue
        df['ticker'] = ticker
        df['delisted'] = ticker in delisted_tickers
        df['market']   = MARKET_MAP.get(ticker, 'US')
        frames.append(df)
        if (i + 1) % 50 == 0: print(f'  {i+1}/{len(all_download)} done ...')
        time.sleep(0.05)
    raw = pd.concat(frames).reset_index()
    raw.to_parquet(RAW_PARQUET, engine='pyarrow', compression='snappy', index=False)
    print(f'Saved {len(raw):,} rows → {RAW_PARQUET}')


# ═══════════════════════════════════════════════════════════════════════
# CELL 6 — Feature engineering v3 (Jansen Ch.4 canonical alphas)
# REPLACES v2 Cell 8.
# Adds: 12-1 momentum, short-term reversal, idiosyncratic vol,
#       volume shock, drawdown depth, Amihud illiquidity.
# ═══════════════════════════════════════════════════════════════════════
import warnings
warnings.filterwarnings('ignore')
try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = lambda x, **k: x

def compute_features_v3(g: pd.DataFrame, market_close: pd.Series | None = None) -> pd.DataFrame:
    c = g['close']; v = g['volume']; h = g['high']; l = g['low']
    out = pd.DataFrame(index=g.index)
    out['close'] = c

    # — returns / momentum
    out['ret_1d']  = c.pct_change(1)
    out['ret_5d']  = c.pct_change(5)
    out['ret_21d'] = c.pct_change(21)
    out['ret_63d'] = c.pct_change(63)

    # — Jansen Ch.4 canonical alphas
    out['mom_12_1']     = c.shift(21) / c.shift(252) - 1     # 12-1 momentum (skip last month)
    out['reversal_5d']  = -c.pct_change(5)                    # short-term reversal
    out['reversal_21d'] = -c.pct_change(21)
    lr = np.log(c / c.shift(1))
    out['vol_21d'] = lr.rolling(21).std()
    out['vol_63d'] = lr.rolling(63).std()
    out['dd_depth_252d'] = c / c.rolling(252).max() - 1      # depth below 1y high

    # — idiosyncratic vol vs market proxy
    if market_close is not None:
        mret = np.log(market_close / market_close.shift(1)).reindex(g.index)
        cov  = lr.rolling(63).cov(mret)
        varm = mret.rolling(63).var()
        beta = cov / varm.replace(0, np.nan)
        resid = lr - beta * mret
        out['idio_vol_63d'] = resid.rolling(63).std()
    else:
        out['idio_vol_63d'] = lr.rolling(63).std()

    # — trend
    sma50 = c.rolling(50).mean(); sma200 = c.rolling(200).mean()
    out['close_vs_sma50']  = c / sma50  - 1
    out['close_vs_sma200'] = c / sma200 - 1

    # — RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs   = gain / loss.replace(0, np.nan)
    out['rsi_14'] = 100 - 100 / (1 + rs)

    # — Bollinger
    sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
    upper = sma20 + 2*std20; lower = sma20 - 2*std20
    out['bb_width']    = (upper - lower) / sma20.replace(0, np.nan)
    out['bb_position'] = (c - lower) / (upper - lower).replace(0, np.nan)

    # — ATR
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()], axis=1).max(axis=1)
    out['atr_pct'] = tr.rolling(14).mean() / c

    # — volume / liquidity
    vol_sma20 = v.rolling(20).mean()
    vol_std20 = v.rolling(20).std()
    out['vol_ratio']  = v / vol_sma20.replace(0, np.nan)
    out['vol_shock']  = (v - vol_sma20) / vol_std20.replace(0, np.nan)
    out['amihud_21d'] = (lr.abs() / (c * v).replace(0, np.nan)).rolling(21).mean()

    # — targets (raw; rank computed cross-sectionally later)
    out['target_5d']  = c.shift(-5)  / c - 1
    out['target_21d'] = c.shift(-21) / c - 1
    return out


if FEAT_PARQUET.exists():
    print(f'FEAT_PARQUET v3 exists ({FEAT_PARQUET.stat().st_size/1e6:.1f} MB). Loading ...')
    feat_df = pd.read_parquet(FEAT_PARQUET)
    feat_df['date'] = pd.to_datetime(feat_df['date'])
else:
    print('Computing v3 features ...')
    raw['date'] = pd.to_datetime(raw['date'])
    raw_idx = raw.set_index('date').sort_index()

    # per-market equal-weighted close index — used for idiosyncratic-vol residualisation
    mkt_proxy = (
        raw_idx.groupby(['market', raw_idx.index])['close']
               .mean()
               .reset_index()
               .pivot(index='date', columns='market', values='close')
               .ffill()
    )
    mkt_proxy = mkt_proxy / mkt_proxy.iloc[0]  # rebase each market to 1.0

    frames = []
    for ticker in _tqdm(raw_idx['ticker'].unique(), desc='Features'):
        g = raw_idx[raw_idx['ticker'] == ticker].sort_index()
        mkt = g['market'].iloc[0]
        mclose = mkt_proxy[mkt].reindex(g.index)
        f = compute_features_v3(g, market_close=mclose)
        f['ticker']   = ticker
        f['market']   = mkt
        f['delisted'] = g['delisted'].iloc[0]

        adv30 = (g['close'] * g['volume']).rolling(30).mean()
        liq_threshold = LIQUIDITY.get(mkt, 500_000)
        f = f[adv30 >= liq_threshold]
        frames.append(f)

    feat_df = pd.concat(frames).reset_index()
    feat_df['date'] = pd.to_datetime(feat_df['date'])

    # ── v4.4 REGIME FEATURES (per market) ─────────────────────────────
    # The bimodal fold-IC pattern (F0/F7 strong, F1/F2/F5 negative) is
    # a regime issue — model is good in trending/calm markets, bad in
    # choppy/rotating ones. Add 4 regime z-scores per (date, market) so
    # the tree can split on regime and learn different behaviours.
    print('Computing v4.4 regime features per market ...')
    regime_frames = []
    for mkt in raw_idx['market'].unique():
        mkt_rows = raw_idx[raw_idx['market'] == mkt].copy()
        p = mkt_proxy[mkt]
        lr_mkt = np.log(p / p.shift(1))

        rg = pd.DataFrame(index=p.index.copy())
        rg['regime_mkt_vol_21d'] = lr_mkt.rolling(21).std()

        # Breadth: fraction of stocks above their 50d SMA on each date
        above_sma = (
            mkt_rows.groupby('ticker')
                    .apply(lambda g: g['close'] > g['close'].rolling(50).mean())
                    .reset_index(level=0, drop=True)
                    .astype(float)
        )
        rg['regime_breadth'] = above_sma.groupby(level=0).mean()

        # Cross-sectional dispersion: std of 21d returns across stocks per date
        ret21 = mkt_rows.groupby('ticker')['close'].pct_change(21)
        rg['regime_dispersion'] = ret21.groupby(level=0).std()

        # Trend strength: rolling 63d autocorrelation (lag 1) of market returns
        rg['regime_trend_63d'] = lr_mkt.rolling(63).apply(
            lambda s: s.autocorr(lag=1) if s.dropna().shape[0] > 5 else np.nan,
            raw=False,
        )

        # Convert each to rolling 252d z-score (min 63d for warm-up)
        for col in ['regime_mkt_vol_21d', 'regime_breadth',
                    'regime_dispersion', 'regime_trend_63d']:
            rm = rg[col].rolling(252, min_periods=63).mean()
            rs = rg[col].rolling(252, min_periods=63).std()
            rg[f'{col}_z'] = (rg[col] - rm) / rs.replace(0, np.nan)

        rg = rg.reset_index()
        rg['market'] = mkt
        regime_frames.append(rg)

    regime_df = pd.concat(regime_frames, ignore_index=True)
    REGIME_COLS = ['regime_mkt_vol_21d_z', 'regime_breadth_z',
                   'regime_dispersion_z', 'regime_trend_63d_z']
    regime_df = regime_df[['date', 'market'] + REGIME_COLS]
    regime_df['date'] = pd.to_datetime(regime_df['date'])

    feat_df = feat_df.merge(regime_df, on=['date', 'market'], how='left')
    print(f'  Added {len(REGIME_COLS)} regime features. feat_df: {feat_df.shape}')

    float_cols = feat_df.select_dtypes(include=[np.floating]).columns
    feat_df[float_cols] = feat_df[float_cols].replace([np.inf, -np.inf], np.nan)
    feat_df.to_parquet(FEAT_PARQUET, engine='pyarrow', compression='snappy', index=False)
    print(f'Saved {len(feat_df):,} rows → {FEAT_PARQUET}')

print(f'feat_df v4.4: {feat_df.shape}')


# ═══════════════════════════════════════════════════════════════════════
# CELL 7 — Feature list  (no Alphalens pre-screen — Jansen Ch.6 says
# tree gain importance is better than IC threshold for nonlinear factors)
# REPLACES v2 Cell 10.
# ═══════════════════════════════════════════════════════════════════════
STOCK_FEATURE_COLS = [
    'ret_1d','ret_5d','ret_21d','ret_63d',
    'mom_12_1','reversal_5d','reversal_21d',
    'vol_21d','vol_63d','idio_vol_63d','dd_depth_252d',
    'close_vs_sma50','close_vs_sma200',
    'rsi_14','bb_width','bb_position','atr_pct',
    'vol_ratio','vol_shock','amihud_21d',
]
# v4.4 — regime features are per (date, market); kept SEPARATE because they
# must NOT be per-date rank-transformed (constant within a date).
REGIME_FEATURE_COLS = [
    'regime_mkt_vol_21d_z', 'regime_breadth_z',
    'regime_dispersion_z', 'regime_trend_63d_z',
]
FEATURE_COLS = STOCK_FEATURE_COLS + REGIME_FEATURE_COLS
passed_features = [f for f in FEATURE_COLS if f in feat_df.columns]
print(f'Using {len(passed_features)} features (no Lasso/Alphalens pre-screen):')
print(passed_features)


# ═══════════════════════════════════════════════════════════════════════
# CELL 8 — Build ML matrix with rank-target + per-date feature rank
# REPLACES v2 Cell 11.
# This is the single biggest Sharpe fix: target_rank neutralises beta /
# regime noise; per-date feature ranks neutralise level shifts.
# ═══════════════════════════════════════════════════════════════════════
ml_start = '2020-01-01'
fd2 = feat_df.copy()
fd2['date'] = pd.to_datetime(fd2['date'])
fd2 = fd2.sort_values('date').reset_index(drop=True)
fd2 = fd2[fd2['date'] >= ml_start]
fd2 = fd2.dropna(subset=passed_features + ['target_21d'])
fd2 = fd2.reset_index(drop=True)

# ── Rank-transform STOCK features per date (Jansen Ch.4)
# v4.4: regime features stay as raw z-scores (they're per-market constants
# within a date — per-date ranking would collapse them).
stock_feats_present = [f for f in STOCK_FEATURE_COLS if f in fd2.columns]
print(f'Rank-transforming {len(stock_feats_present)} stock features per date '
      f'(regime features kept as raw z-scores)...')
fd2[stock_feats_present] = (
    fd2.groupby('date')[stock_feats_present].transform(lambda s: s.rank(pct=True))
)

# ── Rank-transform target per date (Jansen Ch.12)
fd2['target_rank'] = fd2.groupby('date')['target_21d'].rank(pct=True)

# v4.2 — market-NEUTRALISED rank target (Jansen Ch.4 "sector neutralisation").
# Subtract per-market mean before ranking so the model learns
# "stocks that beat their market" instead of "stocks with high raw returns".
# Removes the dominant source of regime/beta noise the bimodal fold-IC
# in v3/v4.1 came from. (Use sector if you have GICS; market is the proxy here.)
fd2['target_demeaned'] = (
    fd2['target_21d'] - fd2.groupby(['date','market'])['target_21d'].transform('mean')
)
fd2['target_demeaned_rank'] = (
    fd2.groupby('date')['target_demeaned'].rank(pct=True)
)

X_all = fd2[passed_features].copy()
y_all = fd2['target_rank'].copy()   # v4.1 baseline — raw rank (was market-neutral in v4.2 but flattened the spread)
y_raw = fd2['target_21d'].copy()    # keep raw for IC computation + backtest

unique_dates = np.sort(fd2['date'].unique())
date_to_rows = {d: g.index.tolist() for d, g in fd2.groupby('date')}
print(f'X: {X_all.shape}  unique dates: {len(unique_dates)}  avg per date: {len(fd2)/len(unique_dates):.0f}')


# ═══════════════════════════════════════════════════════════════════════
# CELL 9 — Walk-forward CV (purge + embargo) — Jansen Ch.7
# ═══════════════════════════════════════════════════════════════════════
class MultipleTimeSeriesCV:
    """Walk-forward CV split on unique trading dates."""
    def __init__(self, n_splits, train_period, test_period, purge, embargo):
        self.n_splits, self.train_period = n_splits, train_period
        self.test_period, self.purge, self.embargo = test_period, purge, embargo

    def split(self, X, y=None, groups=None, dates=None, date_to_rows=None):
        n_dates = len(dates)
        stride  = self.test_period + self.embargo
        last_test_end = n_dates
        for i in range(self.n_splits - 1, -1, -1):
            test_end    = last_test_end - i * stride
            test_start  = test_end - self.test_period
            train_end   = test_start - self.purge
            train_start = max(0, train_end - self.train_period)
            if train_start >= train_end or test_start >= test_end: continue
            if train_end <= 0 or test_start < 0: continue
            train_dates = dates[train_start:train_end]
            test_dates  = dates[test_start:test_end]
            train_rows = [r for d in train_dates for r in date_to_rows[d]]
            test_rows  = [r for d in test_dates  for r in date_to_rows[d]]
            yield (train_rows, test_rows)

    def get_n_splits(self, X=None, y=None, groups=None): return self.n_splits


cv = MultipleTimeSeriesCV(
    n_splits=N_SPLITS, train_period=TRAIN_PERIOD, test_period=TEST_PERIOD,
    purge=TARGET_HORIZON, embargo=EMBARGO,
)
splits = list(cv.split(X_all, dates=unique_dates, date_to_rows=date_to_rows))
print(f'CV folds: {len(splits)}')
for i, (tr, te) in enumerate(splits):
    tr_d = fd2.loc[tr, 'date']; te_d = fd2.loc[te, 'date']
    if i == 0 or i == len(splits) - 1:
        print(f'Fold {i:2d} | train {tr_d.nunique()}d [{tr_d.min().date()} → {tr_d.max().date()}]'
              f'  test {te_d.nunique()}d [{te_d.min().date()} → {te_d.max().date()}]')


# ═══════════════════════════════════════════════════════════════════════
# CELL 10 — Skip LassoCV.  Jansen Ch.6: with a regularised tree,
# Lasso pre-screen on raw returns destroys signal (v2 kept 1/17 features).
# ═══════════════════════════════════════════════════════════════════════
selected_features = passed_features
print(f'Selected features (no Lasso): {len(selected_features)}')


# ═══════════════════════════════════════════════════════════════════════
# CELL 11 v4 — ENSEMBLE training (5d × 21d horizons × 3 seeds = 6 models/fold)
# Replaces single-model v3.  Targets the bimodal fold-IC variance
# (v3 had F0=+0.17, F2=-0.02): seed+horizon ensemble pulls ICIR up by
# smoothing both extremes. (Jansen Ch.12 "horizon ensembling".)
# ═══════════════════════════════════════════════════════════════════════
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

LGB_PARAMS = dict(
    objective         = 'regression',
    metric            = 'rmse',
    n_estimators      = 1500,
    learning_rate     = 0.02,
    num_leaves        = 63,
    min_child_samples = 100,
    subsample         = 0.7,
    colsample_bytree  = 0.7,
    reg_alpha         = 0.5,
    reg_lambda        = 2.0,
    n_jobs            = -1,
    verbosity         = -1,
)

# v4.1 — drop 5d horizon. The 5d model disagreed with 21d on signal levels,
# inverting the cross-section at the extremes (cum_top 1.46 < cum_bottom 2.99
# in v4). Keep only 21d × 3 seeds.
SEEDS    = [42, 7, 2024]
HORIZONS = [('21d', y_all)]

results   = []
oof_preds = pd.Series(np.nan, index=fd2.index, dtype=float)

for fold_i, (train_idx, test_idx) in enumerate(splits):
    test_idx_arr = np.array(test_idx)
    n_test = len(test_idx_arr)
    pred_accum = np.zeros(n_test)
    pred_count = np.zeros(n_test)
    best_iters = []

    for horizon_name, y_series in HORIZONS:
        X_tr_full = np.array(X_all.loc[train_idx], dtype=float)
        y_tr_full = np.array(y_series.loc[train_idx], dtype=float)
        X_te_full = np.array(X_all.loc[test_idx],  dtype=float)
        y_te_full = np.array(y_series.loc[test_idx], dtype=float)

        ok_tr = np.isfinite(X_tr_full).all(axis=1) & np.isfinite(y_tr_full)
        ok_te = np.isfinite(X_te_full).all(axis=1) & np.isfinite(y_te_full)
        X_tr, y_tr = X_tr_full[ok_tr], y_tr_full[ok_tr]
        X_te, y_te = X_te_full[ok_te], y_te_full[ok_te]

        if len(X_tr) < 1000 or len(X_te) < 100:
            continue

        # v4.4 — exponential recency weights for training samples.
        # Older samples get exponentially less weight so the model
        # adapts to recent regime. Half-life ≈ 1 year of trading days.
        RECENCY_HALFLIFE_DAYS = 252
        tr_dates = fd2.loc[train_idx, 'date'].values[ok_tr]
        max_tr_date = tr_dates.max()
        age_days = (max_tr_date - tr_dates).astype('timedelta64[D]').astype(float)
        sample_w = np.exp(-age_days / RECENCY_HALFLIFE_DAYS)

        for seed in SEEDS:
            params = {**LGB_PARAMS, 'random_state': seed}
            m = lgb.LGBMRegressor(**params)
            m.fit(X_tr, y_tr,
                  sample_weight=sample_w,
                  eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)])
            best_iters.append(m.best_iteration_)
            preds_te = m.predict(X_te)

            # Per-date rank of preds (so 5d/21d are on the same scale)
            te_dates_ok = fd2.loc[test_idx, 'date'].values[ok_te]
            preds_rank = (pd.Series(preds_te)
                          .groupby(pd.Series(te_dates_ok))
                          .rank(pct=True)
                          .values)
            pred_accum[ok_te] += preds_rank
            pred_count[ok_te] += 1

    if pred_count.sum() == 0:
        continue

    pred_count[pred_count == 0] = np.nan
    fold_ensemble = pred_accum / pred_count

    # Rank-average across models within each date (defensive)
    te_dates_full = fd2.loc[test_idx, 'date'].values
    fold_ens = (pd.Series(fold_ensemble, index=test_idx_arr)
                .groupby(te_dates_full).rank(pct=True))
    oof_preds.loc[test_idx_arr] = fold_ens.values

    # IC vs RAW returns
    y_te_raw = y_raw.loc[test_idx].values
    valid    = np.isfinite(fold_ens.values) & np.isfinite(y_te_raw)
    rho_p, _ = spearmanr(fold_ens.values[valid], y_te_raw[valid])
    auc = (roc_auc_score((y_te_raw[valid] > 0).astype(int), fold_ens.values[valid])
           if len(np.unique(y_te_raw[valid] > 0)) > 1 else np.nan)

    per_date = []
    for d in np.unique(te_dates_full[valid]):
        mm = (te_dates_full == d) & valid
        if mm.sum() >= 10:
            r, _ = spearmanr(fold_ens.values[mm], y_te_raw[mm])
            if np.isfinite(r): per_date.append(r)
    rho_pd = float(np.mean(per_date)) if per_date else np.nan

    te_d = fd2.loc[test_idx, 'date']
    results.append(dict(
        fold=fold_i, ic=rho_pd, ic_pooled=rho_p, auc=auc,
        n_train=int(valid.sum()), n_test=int(valid.sum()),
        train_days=fd2.loc[train_idx, 'date'].nunique(),
        test_start=te_d.min().date(), test_end=te_d.max().date(),
    ))
    print(f'Fold {fold_i:2d} | IC(perdate)={rho_pd:+.4f}  IC(pooled)={rho_p:+.4f}  '
          f'AUC={auc:.4f}  best_iters={best_iters}')

res_df = pd.DataFrame(results)
print(f'\nEnsemble Mean per-date IC: {res_df.ic.mean():.4f}  '
      f'(ICIR={res_df.ic.mean()/res_df.ic.std():.2f})')
print(f'Positive IC folds: {(res_df.ic > 0).sum()}/{len(res_df)}')


# ═══════════════════════════════════════════════════════════════════════
# CELL 12 — Winsorise OOF signal + cross-sectional z-score per date
# (Jansen Ch.12)  No fold-filtering needed: all folds train on rank-target
# of equal length.
# ═══════════════════════════════════════════════════════════════════════
MIN_TRAIN_DAYS = 500
kept_folds = res_df[res_df['train_days'] >= MIN_TRAIN_DAYS]['fold'].tolist()
dropped_folds = res_df[res_df['train_days'] < MIN_TRAIN_DAYS]['fold'].tolist()
print(f'Keeping {len(kept_folds)} folds; dropping {len(dropped_folds)}')

kept_test_indices = set()
for fold_i, (_, te_idx) in enumerate(splits):
    if fold_i in kept_folds: kept_test_indices.update(te_idx)

oof_preds_filtered = oof_preds.copy()
drop_mask = ~oof_preds_filtered.index.isin(kept_test_indices)
oof_preds_filtered[drop_mask] = np.nan

oof_valid_mask = oof_preds_filtered.notna()
signal_dates   = fd2.loc[oof_valid_mask, 'date'].values
signal         = oof_preds_filtered[oof_valid_mask].copy()

# winsorise 1/99 per date
signal_ws = []
for d in np.unique(signal_dates):
    m = signal_dates == d; grp = signal[m]
    if len(grp) >= 3:
        lo, hi = np.percentile(grp, [1, 99])
        signal_ws.append(grp.clip(lower=lo, upper=hi))
    else:
        signal_ws.append(grp)
signal_winsorised = pd.concat(signal_ws)

ic_oof_raw    = float(np.corrcoef(oof_preds.dropna(), y_raw[oof_preds.notna()])[0,1])
ic_oof_pooled = float(np.corrcoef(signal_winsorised.values,
                                  y_raw[oof_valid_mask].values)[0,1])
per_date_ics = []
y_oof = y_raw[oof_valid_mask]
for d in np.unique(signal_dates):
    m = signal_dates == d
    if m.sum() >= 10:
        r, _ = spearmanr(signal_winsorised[m].values, y_oof[m].values)
        if np.isfinite(r): per_date_ics.append(r)
ic_oof    = float(np.mean(per_date_ics))
ic_oof_ir = ic_oof / float(np.std(per_date_ics)) if np.std(per_date_ics) > 0 else 0.0

print(f'OOF IC (raw, pooled): {ic_oof_raw:+.4f}')
print(f'OOF IC (pooled, winsorised): {ic_oof_pooled:+.4f}')
print(f'OOF IC (per-date avg): {ic_oof:+.4f}  ICIR={ic_oof_ir:.2f}')


# ═══════════════════════════════════════════════════════════════════════
# CELL 13 v4.3 — Decile-return CALIBRATED backtest
# Across v3/v4/v4.2, middle deciles (D3-D6) consistently outperformed
# the extremes (D1/D10). The signal correctly RANKS but is mis-calibrated:
# its "best" prediction (top quintile) isn't actually the best decile.
# Fix: each rebalance, look at the prior CALIB_LOOKBACK rebalances'
# realised return PER PREDICTED DECILE; long the N_LONG_DECILES highest
# expected-return deciles, short the N_SHORT_DECILES lowest. Adapts to
# whatever non-monotonic shape the model has without hardcoding it.
# (Jansen Ch.12 "post-processing the signal — isotonic / decile regression")
# ═══════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

LONG_ONLY          = True        # v4.6+ — short book hurts on monotonic + positive IC
PER_MARKET_BACKTEST = True       # v4.9 — long top-quintile WITHIN each market
                                  # (US, MY, HK), then equal-weight average across
                                  # markets. Targets the 100% US concentration that
                                  # has held every run. Real geographic diversification.
                                  # Only effective when LONG_ONLY=True (per-market shorts
                                  # not implemented).
REBAL_DAYS         = TARGET_HORIZON
COMMISSION_BPS     = 5.0
SLIPPAGE_BPS       = 2.5
TOTAL_COST_BPS     = (COMMISSION_BPS + SLIPPAGE_BPS) * 2

# Decile calibration (v4.3 → DISABLED in v4.5)
# Reason: v4.4's regime features + recency weighting made the model's signal
# MONOTONIC (D10 = +14.7%, pooled OOF IC = +0.006 — first time ever positive).
# Calibration was designed for a broken signal; with a monotonic one it's
# adversarial, overriding good predictions with stale rolling stats.
# Keep the code path but default-off so we can compare A/B if needed.
DECILE_CALIB       = False        # was True in v4.3
N_DECILES          = 10
CALIB_LOOKBACK     = 6
CALIB_WARMUP       = 4
N_LONG_DECILES     = 2
N_SHORT_DECILES    = 2

# Legacy quintile fallback (used during warm-up)
TOP_FRAC           = 0.20
TOP_DECILE         = TOP_FRAC

# Signal smoothing — KEPT from v4
EMA_ALPHA          = 0.50

# Sizing — v4.7 enables signal-weighting within the top quintile
# Reason: D10 = +14.7% per rebal, D9 = +1.0%. EW averages them. Rank-weighted
# pushes ~70% of book toward D10-ranked names. Cap MAX_POSITION_SIZE still
# enforces ~4% per name diversification.
SIGNAL_WEIGHTED    = True         # was False — rank-weighted within long book
MAX_POSITION_SIZE  = 0.05
MAX_MARKET_CONC    = 0.40

# Volatility target — kept as fallback during Kelly warm-up
VOL_TARGET         = True
VOL_TARGET_ANN     = 0.12
VOL_LOOKBACK       = 6
MAX_LEVERAGE       = 1.0
MIN_LEVERAGE       = 0.03

# v4.10 — Kelly sizing on the top-decile basket return
# Replaces vol-target as the leverage decision once warm-up history exists.
# Kelly fraction f* = mu / sigma^2, scaled by KELLY_FRACTION (quarter-Kelly is
# the standard conservative choice — full Kelly is famously aggressive and
# blows up if mu is over-estimated). Uses UNLEVERED basket return history so
# the leverage decision is on the asset's own characteristics, not our
# already-levered PnL.
KELLY_CRITERION    = False      # REVERTED to v4.9 — Kelly book sizing OFF (vol-target rules)
KELLY_FRACTION     = 0.15       # v4.11 — 1/6.6-Kelly, dialed down from 0.25
                                 # quarter-Kelly drove v4.10 vol to 177% / Sharpe 0.76;
                                 # lighter fraction should land Sharpe ~0.85, vol ~110%
KELLY_LOOKBACK     = 12         # 12 rebals ≈ 1 year of monthly periods
KELLY_WARMUP       = 8          # need 8 rebals before Kelly engages
KELLY_MAX_LEV      = 1.5        # cap above VOL_TARGET's 1.0 (Kelly can lever up)
KELLY_MIN_LEV      = 0.03       # same floor as vol-target

# v4.12 — per-NAME Kelly allocation within the top decile
# (different from KELLY_CRITERION which sizes overall book leverage).
# When True: each stock's weight = clip(mu_i / sigma_i^2, 0, MAX_POS), normalised.
# mu_i is derived from signal rank scaled by recent strategy mean.
# sigma_i^2 is the rolling 6-rebal variance of that stock's actual returns.
# Overrides SIGNAL_WEIGHTED inside the per-market basket when both are on.
KELLY_ALLOC        = False      # REVERTED to v4.9 — per-stock Kelly OFF (SIGNAL_WEIGHTED rules)
KELLY_ALLOC_LOOKBACK = 6        # rebals of stock-level returns for variance est
KELLY_ALLOC_MIN_HISTORY = 3     # min rebals before Kelly weights used
KELLY_ALLOC_MU_SCALE = 2.0      # signal -> expected return calibration factor

DD_CIRCUIT_BREAKER = -0.25

bt = pd.DataFrame({
    'date':   fd2['date'].values,
    'ticker': fd2['ticker'].values,
    'market': fd2['market'].values,
    'actual': y_raw.values,
    'signal_raw': oof_preds_filtered.values,
}).dropna(subset=['signal_raw','actual'])

all_dates   = np.sort(bt['date'].unique())
rebal_dates = all_dates[::REBAL_DAYS]
print(f'Backtest: {len(bt):,} rows, {bt["date"].nunique()} dates, '
      f'{bt["ticker"].nunique()} tickers, {len(rebal_dates)} rebalances')

# ── EMA-smooth signal per ticker, then re-rank per date
sig_panel = bt.pivot_table(index='date', columns='ticker', values='signal_raw')
sig_panel_smooth = sig_panel.ewm(alpha=EMA_ALPHA, adjust=False, min_periods=1).mean()
sig_panel_rank   = sig_panel_smooth.rank(axis=1, pct=True)
sig_long = sig_panel_rank.stack().rename('signal').reset_index()
bt = bt.drop(columns=['signal_raw']).merge(sig_long, on=['date','ticker'], how='left')
bt = bt.dropna(subset=['signal'])
print(f'Signal EMA-smoothed (α={EMA_ALPHA}); panel: {sig_panel.shape}')


def _kelly_alloc_weights(grp_side, rd_local, recent_mu):
    """
    v4.12 — per-stock Kelly allocation within the long basket.
    w_i = clip(mu_i / sigma_i^2, 0, MAX_POSITION_SIZE), normalised to sum to 1.
    mu_i derived from signal rank × recent strategy mean (scaled).
    sigma_i^2 estimated from rolling stock-level actual return history.
    """
    sig_vals = grp_side['signal'].values.astype(float)
    tickers  = grp_side['ticker'].values
    n = len(sig_vals)

    # Expected return per stock — anchor signal rank to recent strategy mu
    sig_norm = sig_vals - sig_vals.min() + 1e-3
    mu_anchor = max(abs(recent_mu), 0.01)
    mu_i = sig_norm * mu_anchor * KELLY_ALLOC_MU_SCALE

    # Per-stock variance from bt history (strict no-lookahead: dates < rd_local)
    sigma2_i = np.zeros(n)
    for j, tk in enumerate(tickers):
        hist = bt[(bt['ticker'] == tk) & (bt['date'] < rd_local)]['actual'].tail(KELLY_ALLOC_LOOKBACK)
        if len(hist) >= KELLY_ALLOC_MIN_HISTORY:
            sigma2_i[j] = float(hist.var())
        else:
            sigma2_i[j] = 0.01            # default 10% per-rebal sigma when no history
    sigma2_i = np.maximum(sigma2_i, 1e-4)  # numerical floor

    # Kelly weights
    w = mu_i / sigma2_i
    w = np.maximum(w, 0)
    if w.sum() <= 0:
        return np.ones(n) / n            # fallback to equal weight

    w = w / w.sum()

    # Position-size cap (iterative redistribute)
    for _ in range(50):
        over = w > MAX_POSITION_SIZE
        if not over.any(): break
        excess = (w[over] - MAX_POSITION_SIZE).sum()
        w[over] = MAX_POSITION_SIZE
        under = ~over
        if under.sum() == 0: break
        room = (MAX_POSITION_SIZE - w[under]).sum()
        if room <= 0: break
        w[under] += excess * (MAX_POSITION_SIZE - w[under]) / room
    return w / w.sum() if w.sum() > 0 else w


def _signal_weights(grp_side, side='long'):
    """Rank-weighted weights within long/short book, position-capped."""
    s = grp_side['signal'].values.astype(float)
    if side == 'long':
        w = s - s.min()
    else:
        w = (1 - s) - (1 - s).min()
    if w.sum() == 0:
        return np.ones(len(s)) / len(s)
    w = w / w.sum()
    for _ in range(50):
        over = w > MAX_POSITION_SIZE
        if not over.any(): break
        excess = (w[over] - MAX_POSITION_SIZE).sum()
        w[over] = MAX_POSITION_SIZE
        under = ~over
        room = (MAX_POSITION_SIZE - w[under]).sum()
        if room <= 1e-12: break
        w[under] += excess * (MAX_POSITION_SIZE - w[under]) / room
    return w / w.sum()


period_returns = []
cum_eq, peak_eq, cb_trips = 1.0, 1.0, 0
recent_rets = []

# Decile-return history: rows = rebal dates, cols = decile (1..N_DECILES)
decile_returns_history = pd.DataFrame(columns=list(range(1, N_DECILES + 1)))
decile_log = []   # diagnostic — record which deciles were chosen each rebal

# v4.10 — unlevered basket return history for Kelly sizing
top_decile_rets_history = []

# v4.12 — track per-rebal Kelly allocations (last entry = latest portfolio)
allocation_log = []

for rd in rebal_dates:
    grp = bt[bt['date'] == rd]
    if len(grp) < 20: continue

    rolling_dd = cum_eq / peak_eq - 1
    if rolling_dd < DD_CIRCUIT_BREAKER:
        cb_trips += 1
        period_returns.append({'date': rd, 'return': 0.0, 'gross': 0.0, 'lev': 0.0})
        continue

    # ── Bucket the cross-section into deciles by smoothed signal
    grp = grp.copy()
    try:
        grp['decile'] = pd.qcut(grp['signal'], N_DECILES, labels=False, duplicates='drop') + 1
    except ValueError:
        # Not enough unique values — fall back to quintile cutoff
        hi = grp['signal'].quantile(1 - TOP_FRAC)
        lo = grp['signal'].quantile(TOP_FRAC)
        longs  = grp[grp['signal'] >= hi].copy()
        shorts = grp[grp['signal'] <= lo].copy()
        if longs.empty: continue
        long_dec, short_dec = [], []
    else:
        # ── Choose which deciles to long / short
        if DECILE_CALIB and len(decile_returns_history) >= CALIB_WARMUP:
            recent = decile_returns_history.tail(CALIB_LOOKBACK)
            expected = recent.mean(axis=0).dropna()
            long_dec  = expected.nlargest(N_LONG_DECILES).index.tolist()
            short_dec = expected.nsmallest(N_SHORT_DECILES).index.tolist()
        else:
            # Warm-up: top/bottom quintile = top 2 / bottom 2 deciles
            long_dec  = [N_DECILES - 1, N_DECILES]   # [9, 10]
            short_dec = [1, 2]
        longs  = grp[grp['decile'].isin(long_dec)].copy()
        shorts = grp[grp['decile'].isin(short_dec)].copy()
        if longs.empty: continue

    # ── Record this rebal's actual per-decile returns for future calibration
    if 'decile' in grp.columns:
        dec_ret = grp.groupby('decile')['actual'].mean()
        decile_returns_history.loc[rd] = [dec_ret.get(d, np.nan) for d in range(1, N_DECILES + 1)]
    decile_log.append({'date': rd, 'long': long_dec, 'short': short_dec})

    if PER_MARKET_BACKTEST and LONG_ONLY:
        # v4.9 — true per-market diversification: long top-quintile WITHIN
        # each market, then equal-weight average across markets that had
        # a valid basket this rebal. v4.12 adds per-stock Kelly allocation.
        market_long_rets = {}
        market_long_weights = {}          # v4.12 — track for diagnostics
        recent_mu = (np.mean(top_decile_rets_history[-KELLY_LOOKBACK:])
                     if len(top_decile_rets_history) >= 3 else 0.05)

        for mkt in grp['market'].unique():
            mkt_grp = grp[grp['market'] == mkt]
            if len(mkt_grp) < 10:
                continue
            mkt_hi = mkt_grp['signal'].quantile(1 - TOP_FRAC)
            mkt_longs = mkt_grp[mkt_grp['signal'] >= mkt_hi].copy()
            if mkt_longs.empty:
                continue

            if KELLY_ALLOC:
                wL_m = _kelly_alloc_weights(mkt_longs, rd, recent_mu)
            elif SIGNAL_WEIGHTED:
                wL_m = _signal_weights(mkt_longs, side='long')
            else:
                wL_m = np.ones(len(mkt_longs)) / len(mkt_longs)

            market_long_rets[mkt] = float((mkt_longs['actual'].values * wL_m).sum())
            market_long_weights[mkt] = {
                'tickers': mkt_longs['ticker'].tolist(),
                'weights': [round(float(w), 4) for w in wL_m],
                'signals': [round(float(s), 4) for s in mkt_longs['signal'].values],
                'fwd_returns': [round(float(r), 4) for r in mkt_longs['actual'].values],
            }

        if not market_long_rets:
            continue
        long_ret = float(np.mean(list(market_long_rets.values())))
        short_ret = 0.0
        raw_ret = long_ret

        # v4.12 — record allocation snapshot for this rebal
        allocation_log.append({
            'date': str(pd.Timestamp(rd).date()),
            'markets': market_long_weights,
            'recent_mu': round(float(recent_mu), 4),
        })
    else:
        # Pooled cross-section selection (v4.8 and earlier behaviour)
        if SIGNAL_WEIGHTED:
            wL = _signal_weights(longs,  side='long')
            wS = _signal_weights(shorts, side='short') if not shorts.empty else np.array([])
        else:
            wL = np.ones(len(longs))  / len(longs)
            wS = np.ones(len(shorts)) / len(shorts) if not shorts.empty else np.array([])

        long_ret  = float((longs['actual'].values * wL).sum())
        short_ret = float((shorts['actual'].values * wS).sum()) if not shorts.empty else 0.0

        if LONG_ONLY:
            raw_ret = long_ret
        else:
            raw_ret = 0.5 * long_ret - 0.5 * short_ret

    # v4.10 — Kelly sizing on top-decile basket return
    if KELLY_CRITERION and len(top_decile_rets_history) >= KELLY_WARMUP:
        recent = np.array(top_decile_rets_history[-KELLY_LOOKBACK:])
        mu = float(np.mean(recent))
        sigma2 = float(np.var(recent))
        if sigma2 > 1e-12 and mu > 0:
            kelly_full = mu / sigma2
            lev = float(np.clip(KELLY_FRACTION * kelly_full,
                                 KELLY_MIN_LEV, KELLY_MAX_LEV))
        else:
            # Negative mean or undefined vol — sit out (minimum size)
            lev = KELLY_MIN_LEV
    elif VOL_TARGET and len(recent_rets) >= 3:
        # Fallback during Kelly warm-up: use vol-target
        realised_vol_ann = np.std(recent_rets[-VOL_LOOKBACK:]) * np.sqrt(252 / REBAL_DAYS)
        if realised_vol_ann > 1e-6:
            lev = float(np.clip(VOL_TARGET_ANN / realised_vol_ann,
                                MIN_LEVERAGE, MAX_LEVERAGE))
        else:
            lev = 1.0
    else:
        lev = 1.0

    gross_ret = raw_ret * lev
    cost = (TOTAL_COST_BPS / 1e4) * lev
    net = gross_ret - cost

    cum_eq *= (1 + net); peak_eq = max(peak_eq, cum_eq)
    recent_rets.append(net)
    top_decile_rets_history.append(raw_ret)   # UNLEVERED for next iter's Kelly
    period_returns.append({'date': rd, 'return': net, 'gross': gross_ret, 'lev': lev})

perf_df    = pd.DataFrame(period_returns).set_index('date')
perf       = perf_df['return']
perf_gross = perf_df['gross']
levs       = perf_df['lev']
cumulative = (1 + perf).cumprod()
dd = cumulative / cumulative.cummax() - 1

# Equal-weight baseline (top quintile, EW, no smoothing / vol-target)
ew_returns = []
cum_ew, peak_ew = 1.0, 1.0
for rd in rebal_dates:
    grp = bt[bt['date'] == rd]
    if len(grp) < 20: continue
    if cum_ew / peak_ew - 1 < DD_CIRCUIT_BREAKER:
        ew_returns.append({'date': rd, 'return': 0.0}); continue
    hi = grp['signal'].quantile(1 - 0.20)
    longs = grp[grp['signal'] >= hi]
    if longs.empty: continue
    net = longs['actual'].mean() - TOTAL_COST_BPS / 1e4
    cum_ew *= (1 + net); peak_ew = max(peak_ew, cum_ew)
    ew_returns.append({'date': rd, 'return': net})
perf_ew = (pd.DataFrame(ew_returns).set_index('date')['return']
           .reindex(perf.index, fill_value=0.0))
cumulative_ew = (1 + perf_ew).cumprod()

# Performance stats
ppy = 252 / REBAL_DAYS
n_years = len(perf) / ppy
sharpe        = float(perf.mean()       / perf.std()       * np.sqrt(ppy)) if perf.std() > 0 else np.nan
sharpe_g      = float(perf_gross.mean() / perf_gross.std() * np.sqrt(ppy)) if perf_gross.std() > 0 else np.nan
sharpe_ew     = float(perf_ew.mean()    / perf_ew.std()    * np.sqrt(ppy)) if perf_ew.std() > 0 else np.nan
max_dd        = float((cumulative / cumulative.cummax() - 1).min())
max_dd_ew     = float((cumulative_ew / cumulative_ew.cummax() - 1).min())
ann_return    = float(cumulative.iloc[-1]    ** (1 / n_years) - 1) if n_years > 0 else np.nan
ann_return_ew = float(cumulative_ew.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else np.nan
realised_vol  = float(perf.std() * np.sqrt(ppy))
avg_lev       = float(levs.mean())

mode = 'Long-Only' if LONG_ONLY else 'Long-Short'
print(f'\n{"Metric":<26}  {mode:>14}  {"EW baseline":>14}')
print('─' * 60)
print(f'{"Annual Sharpe (net)":<26}  {sharpe:>14.2f}  {sharpe_ew:>14.2f}')
print(f'{"Annual Sharpe (gross)":<26}  {sharpe_g:>14.2f}  {"—":>14}')
print(f'{"Annual Return":<26}  {ann_return*100:>13.1f}%  {ann_return_ew*100:>13.1f}%')
print(f'{"Max Drawdown":<26}  {max_dd*100:>13.1f}%  {max_dd_ew*100:>13.1f}%')
print(f'{"Realised Vol (ann)":<26}  {realised_vol*100:>13.1f}%')
print(f'{"Avg leverage":<26}  {avg_lev:>14.2f}')
print(f'{"Circuit-breaker trips":<26}  {cb_trips:>14d}')
print(f'{"Holding periods":<26}  {len(perf):>14d}')


# ═══════════════════════════════════════════════════════════════════════
# CELL 14 — Equity / DD / leverage / per-fold IC plot (v4)
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(12, 12),
                          gridspec_kw={'height_ratios':[3, 1.3, 1.3, 1.3]})
cumulative.plot(ax=axes[0], color='steelblue', lw=2,
                label=f'v4 {mode} (Sharpe={sharpe:.2f})')
cumulative_ew.plot(ax=axes[0], color='#888', lw=1.2, ls='--',
                   label=f'EW top-quintile (Sharpe={sharpe_ew:.2f})')
axes[0].set_title(f'v4 Equity Curve — {mode} (EMA-smoothed, vol-targeted)')
axes[0].axhline(1, color='grey', lw=0.8, ls=':'); axes[0].legend(fontsize=10)

dd.plot(ax=axes[1], color='indianred', lw=1)
axes[1].fill_between(dd.index, dd, 0, alpha=0.3, color='indianred')
axes[1].axhline(DD_CIRCUIT_BREAKER, color='orange', lw=1, ls='--',
                label=f'CB ({DD_CIRCUIT_BREAKER*100:.0f}%)')
axes[1].set_title('Drawdown'); axes[1].legend(fontsize=9)

levs.plot(ax=axes[2], color='purple', lw=1)
axes[2].axhline(1.0, color='black', lw=0.6, ls=':')
axes[2].set_title(f'Vol-target leverage (avg={avg_lev:.2f}, '
                   f'realised vol={realised_vol*100:.1f}% / target {VOL_TARGET_ANN*100:.0f}%)')
axes[2].set_ylabel('Leverage')

colors = ['green' if x > 0 else 'red' for x in res_df['ic']]
axes[3].bar(res_df['fold'], res_df['ic'], color=colors)
axes[3].set_title(f'Per-fold IC (mean={res_df.ic.mean():.3f}, '
                   f'ICIR={res_df.ic.mean()/res_df.ic.std():.2f})')
axes[3].axhline(0, color='black', lw=0.8)

plt.tight_layout()
plt.savefig(BASE / 'equity_curve_v4.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {BASE}/equity_curve_v4.png')


# ═══════════════════════════════════════════════════════════════════════
# CELL 15 — SHAP feature importance (Jansen Ch.12)
# ═══════════════════════════════════════════════════════════════════════
import shap

last_train_idx, last_test_idx = splits[-1]
X_tr_shap = np.array(X_all.loc[last_train_idx], dtype=float)
y_tr_shap = np.array(y_all.loc[last_train_idx], dtype=float)
X_te_shap = np.array(X_all.loc[last_test_idx],  dtype=float)
y_te_shap = np.array(y_all.loc[last_test_idx],  dtype=float)
ok_tr_s = np.isfinite(X_tr_shap).all(axis=1) & np.isfinite(y_tr_shap)
ok_te_s = np.isfinite(X_te_shap).all(axis=1) & np.isfinite(y_te_shap)
X_tr_shap, y_tr_shap = X_tr_shap[ok_tr_s], y_tr_shap[ok_tr_s]
X_te_shap, y_te_shap = X_te_shap[ok_te_s], y_te_shap[ok_te_s]

model_shap = lgb.LGBMRegressor(**{**LGB_PARAMS, 'random_state': 42})
model_shap.fit(X_tr_shap, y_tr_shap, eval_set=[(X_te_shap, y_te_shap)],
               callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
explainer = shap.TreeExplainer(model_shap)
shap_values = explainer.shap_values(X_te_shap)

shap_importance = pd.DataFrame({
    'feature':     selected_features,
    'mean_|SHAP|': np.abs(shap_values).mean(axis=0),
}).sort_values('mean_|SHAP|', ascending=False).reset_index(drop=True)
print('SHAP importance (last fold):')
print(shap_importance.to_string(index=False))

fig_shap, _ = plt.subplots(figsize=(10, max(4, len(selected_features) * 0.4)))
shap.summary_plot(shap_values, features=X_te_shap, feature_names=selected_features,
                  show=False, plot_size=None)
plt.title('SHAP — last fold test set'); plt.tight_layout()
plt.savefig(BASE / 'shap_importance_v4.png', dpi=150, bbox_inches='tight'); plt.show()
print(f'Saved: {BASE}/shap_importance_v4.png')


# ═══════════════════════════════════════════════════════════════════════
# CELL 16 — Top 20 by signal at latest rebalance + v4.12 Kelly allocation
# ═══════════════════════════════════════════════════════════════════════
latest_rebal = rebal_dates[-1]
grp_latest = bt[bt['date'] == latest_rebal].copy().sort_values('signal', ascending=False)
threshold_latest = grp_latest['signal'].quantile(1 - TOP_DECILE)

# v4.12 — pull the latest Kelly allocation snapshot for the Portfolio view
latest_alloc = allocation_log[-1] if allocation_log else None
ticker_to_kelly_weight = {}
if latest_alloc is not None:
    for mkt, info in latest_alloc['markets'].items():
        # Within each market basket, weights sum to 1; combined book = 1/N_mkts each
        n_mkts = len(latest_alloc['markets'])
        for tk, w in zip(info['tickers'], info['weights']):
            ticker_to_kelly_weight[tk] = ticker_to_kelly_weight.get(tk, 0.0) + (w / n_mkts)
    print(f'\nv4.12 Kelly portfolio allocation at {latest_alloc["date"]} '
          f'(recent_mu={latest_alloc["recent_mu"]:+.4f}):')
    for tk, w in sorted(ticker_to_kelly_weight.items(), key=lambda kv: -kv[1])[:20]:
        print(f'  {tk:<12s}  {w*100:>6.2f}%')
top_decile_count = int((grp_latest['signal'] >= threshold_latest).sum())

top20 = grp_latest.head(20).copy()
top20['rank'] = range(1, len(top20) + 1)
top20['fwd_return_%'] = (top20['actual'] * 100).round(2)
print(f'\nTop 20 — rebalance {pd.Timestamp(latest_rebal).date()}, '
      f'universe {len(grp_latest)}, top quintile size {top_decile_count}')
print(top20[['rank','ticker','signal','fwd_return_%']].to_string(index=False))
grp_latest.to_csv(BASE / 'stock_ranking_latest_v4.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════
# CELL 17 — Export dashboard_data.json (v4)
# ═══════════════════════════════════════════════════════════════════════
import json
print('Exporting dashboard_data_v4.json ...')

cost_per_rebal = TOTAL_COST_BPS / 1e4
gross_period_returns = perf.values + cost_per_rebal
cumulative_gross = np.cumprod(1 + gross_period_returns)

equity_data = {
    'dates': [str(d.date()) if hasattr(d,'date') else str(d)[:10] for d in cumulative.index],
    'cumulative': cumulative.values.tolist(),
    'cumulative_gross': cumulative_gross.tolist(),
    'drawdown': dd.values.tolist(),
    'period_returns': perf.values.tolist(),
    'period_returns_gross': gross_period_returns.tolist(),
    'cumulative_ew': cumulative_ew.reindex(cumulative.index).ffill().fillna(1.0).values.tolist(),
    'period_returns_ew': perf_ew.reindex(perf.index).fillna(0.0).values.tolist(),
}

fold_train_dates = {}
for fold_i, (tr_i, _) in enumerate(splits):
    d = fd2.loc[tr_i, 'date']
    fold_train_dates[fold_i] = (str(d.min().date()), str(d.max().date()))

fold_data = []
for _, row in res_df.iterrows():
    fold_i = int(row['fold'])
    ts, te = fold_train_dates.get(fold_i, ('',''))
    fold_data.append({
        'fold': fold_i, 'ic': round(float(row['ic']), 4),
        'auc': round(float(row['auc']), 4),
        'n_train': int(row['n_train']), 'n_test': int(row['n_test']),
        'train_days': int(row['train_days']),
        'train_start': ts, 'train_end': te,
        'test_start': str(row['test_start']), 'test_end': str(row['test_end']),
    })

shap_data = [{'feature': r['feature'],
              'importance': round(float(r['mean_|SHAP|']), 6)}
             for _, r in shap_importance.iterrows()]

top_stocks_data = []
for i, (_, row) in enumerate(grp_latest.head(20).iterrows()):
    tk = row['ticker']
    kelly_w = ticker_to_kelly_weight.get(tk, 0.0)
    top_stocks_data.append({
        'rank': i + 1, 'ticker': tk,
        'signal': round(float(row['signal']), 6),
        'fwd_return_pct': round(float(row['actual']) * 100, 2),
        'in_portfolio': i < top_decile_count,
        # v4.12 — Kelly allocation as % of book
        'kelly_weight_pct': round(float(kelly_w) * 100, 3),
    })

# v4.12 — also export the full Kelly allocation history for charting / audit
# All numeric fields default to 0 (not None) for JS toFixed safety
_lr_mu = float(latest_alloc['recent_mu']) if (latest_alloc and np.isfinite(latest_alloc['recent_mu'])) else 0.0
dashboard_data_kelly_alloc_extra = {
    'latest_date': latest_alloc['date'] if latest_alloc else '',
    'latest_recent_mu': round(_lr_mu, 4),
    'latest_allocation': [
        {'ticker': tk, 'weight_pct': round(float(w) * 100, 3)}
        for tk, w in sorted(ticker_to_kelly_weight.items(), key=lambda kv: -kv[1])
    ],
    'history_length': len(allocation_log),
}

# OOF scatter (subsample to 2000)
oof_sig = signal_winsorised.values
oof_act = y_raw[oof_valid_mask].values
n_oof = len(oof_sig)
if n_oof > 2000:
    rng = np.random.RandomState(42)
    idx_sample = rng.choice(n_oof, 2000, replace=False)
    oof_scatter = [{'signal': round(float(oof_sig[i]), 6),
                    'actual': round(float(oof_act[i]), 6)} for i in sorted(idx_sample)]
else:
    oof_scatter = [{'signal': round(float(s), 6), 'actual': round(float(a), 6)}
                   for s, a in zip(oof_sig, oof_act)]

# Per-rebal details
rebal_details = []
for rd in rebal_dates:
    grp = bt[bt['date'] == rd]
    if len(grp) < 10: continue
    thr = grp['signal'].quantile(1 - TOP_DECILE)
    longs = grp[grp['signal'] >= thr]
    if longs.empty: continue
    rebal_details.append({
        'date': str(pd.Timestamp(rd).date()),
        'n_stocks': int(len(longs)), 'n_universe': int(len(grp)),
        'tickers': longs.sort_values('signal', ascending=False)['ticker'].tolist(),
    })

# Region breakdown
def _region(t):
    t = str(t).upper()
    if t.endswith('.KL') or t.endswith('.KLSE'): return 'MY (Bursa)'
    if t.endswith('.HK'): return 'HK (HKEX)'
    return 'US'
portfolio_tickers = grp_latest.head(top_decile_count)['ticker'].tolist()
region_counts = {}
for t in portfolio_tickers:
    r = _region(t); region_counts[r] = region_counts.get(r, 0) + 1

# Metrics
periods_per_year = 252 / REBAL_DAYS
sharpe_gross = float(np.mean(gross_period_returns) / np.std(gross_period_returns)
                     * np.sqrt(periods_per_year)) if np.std(gross_period_returns) > 0 else 0
ann_ret_gross = float(cumulative_gross[-1] ** (1 / n_years) - 1) if n_years > 0 else 0

metrics = {
    'sharpe': round(sharpe, 2),
    'sharpe_gross': round(sharpe_gross, 2),
    'annual_return_pct': round(ann_return * 100, 1),
    'annual_return_gross_pct': round(ann_ret_gross * 100, 1),
    'total_return_pct': round(float(cumulative.iloc[-1] - 1) * 100, 1),
    'max_drawdown_pct': round(max_dd * 100, 1),
    'mean_ic': round(float(res_df['ic'].mean()), 4),
    'icir': round(float(res_df['ic'].mean() / res_df['ic'].std()), 2),
    'mean_ic_all_folds': round(float(res_df['ic'].mean()), 4),
    'icir_all_folds': round(float(res_df['ic'].mean() / res_df['ic'].std()), 2),
    'oof_signal_ic': round(ic_oof, 4),
    'oof_signal_ic_pooled': round(ic_oof_pooled, 4),
    'oof_signal_ic_raw': round(ic_oof_raw, 4),
    'positive_ic_folds': int((res_df['ic'] > 0).sum()),
    'total_folds': len(res_df), 'total_folds_all': len(res_df),
    'dropped_folds': len(dropped_folds), 'min_train_days': int(MIN_TRAIN_DAYS),
    'n_features_selected': len(selected_features),
    'n_features_total': len(passed_features),
    'backtest_start': str(perf.index.min().date()) if hasattr(perf.index.min(),'date') else str(perf.index.min())[:10],
    'backtest_end':   str(perf.index.max().date()) if hasattr(perf.index.max(),'date') else str(perf.index.max())[:10],
    'holding_periods': len(perf),
    'universe_tickers': int(bt['ticker'].nunique()),
    'universe_rows': len(bt),
    'latest_rebal_date': str(pd.Timestamp(latest_rebal).date()),
    'top_decile_count': top_decile_count,
    'sharpe_ew': round(sharpe_ew, 2),
    'annual_return_ew_pct': round(ann_return_ew * 100, 1) if not np.isnan(ann_return_ew) else 0.0,
    'max_drawdown_ew_pct': round(max_dd_ew * 100, 1),
    'circuit_trips': int(cb_trips),
    'signal_weighted': bool(SIGNAL_WEIGHTED),
    'long_only': bool(LONG_ONLY),
    'max_position_size_pct': round(MAX_POSITION_SIZE * 100, 1),
    'max_market_conc_pct': round(MAX_MARKET_CONC * 100, 1),
    'dd_circuit_breaker_pct': round(DD_CIRCUIT_BREAKER * 100, 1),
    # v4-specific
    'realised_vol_pct': round(realised_vol * 100, 1),
    'avg_leverage': round(avg_lev, 2),
    'ema_alpha': float(EMA_ALPHA),
    'vol_target_ann_pct': round(VOL_TARGET_ANN * 100, 1),
    'top_frac': float(TOP_FRAC),
    'n_seeds': len(SEEDS),
    'n_horizons': len(HORIZONS),
    # v4.9 / v4.10
    'per_market_backtest': bool(PER_MARKET_BACKTEST),
    'kelly_criterion': bool(KELLY_CRITERION),
    'kelly_fraction': float(KELLY_FRACTION),
    'kelly_lookback': int(KELLY_LOOKBACK),
    'kelly_warmup': int(KELLY_WARMUP),
    'kelly_max_lev': float(KELLY_MAX_LEV),
    'kelly_active_periods': int(sum(1 for r in period_returns
                                     if r.get('lev', 1.0) != 1.0
                                     and r.get('lev', 1.0) != MIN_LEVERAGE)),
    'kelly_basket_mu_last': (
        round(float(np.mean(top_decile_rets_history[-KELLY_LOOKBACK:])) * 100, 3)
        if len(top_decile_rets_history) >= KELLY_LOOKBACK
           and np.isfinite(np.mean(top_decile_rets_history[-KELLY_LOOKBACK:]))
        else 0.0
    ),
    'kelly_basket_sigma_last': (
        round(float(np.std(top_decile_rets_history[-KELLY_LOOKBACK:])) * 100, 3)
        if len(top_decile_rets_history) >= KELLY_LOOKBACK
           and np.isfinite(np.std(top_decile_rets_history[-KELLY_LOOKBACK:]))
        else 0.0
    ),
}

config = {
    'TRAIN_PERIOD': int(TRAIN_PERIOD), 'TEST_PERIOD': int(TEST_PERIOD),
    'TARGET_HORIZON': int(TARGET_HORIZON), 'EMBARGO': int(EMBARGO),
    'N_SPLITS': int(N_SPLITS),
    'TOP_DECILE': float(TOP_DECILE), 'TOP_FRAC': float(TOP_FRAC),
    'REBAL_DAYS': int(REBAL_DAYS),
    'COMMISSION_BPS': float(COMMISSION_BPS), 'SLIPPAGE_BPS': float(SLIPPAGE_BPS),
    'TOTAL_COST_BPS': float(TOTAL_COST_BPS),
    'SIGNAL_WEIGHTED': bool(SIGNAL_WEIGHTED), 'LONG_ONLY': bool(LONG_ONLY),
    'MAX_POSITION_SIZE': float(MAX_POSITION_SIZE),
    'MAX_MARKET_CONC': float(MAX_MARKET_CONC),
    'DD_CIRCUIT_BREAKER': float(DD_CIRCUIT_BREAKER),
    'EMA_ALPHA': float(EMA_ALPHA),
    'VOL_TARGET': bool(VOL_TARGET),
    'VOL_TARGET_ANN': float(VOL_TARGET_ANN),
    'VOL_LOOKBACK': int(VOL_LOOKBACK),
    'MAX_LEVERAGE': float(MAX_LEVERAGE),
    'MIN_LEVERAGE': float(MIN_LEVERAGE),
    'SEEDS': list(SEEDS),
    # v4.9 / v4.10
    'PER_MARKET_BACKTEST': bool(PER_MARKET_BACKTEST),
    'KELLY_CRITERION': bool(KELLY_CRITERION),
    'KELLY_FRACTION': float(KELLY_FRACTION),
    'KELLY_LOOKBACK': int(KELLY_LOOKBACK),
    'KELLY_WARMUP': int(KELLY_WARMUP),
    'KELLY_MAX_LEV': float(KELLY_MAX_LEV),
    'KELLY_MIN_LEV': float(KELLY_MIN_LEV),
}

terminal_lines = [
    f'ML matrix: {len(fd2):,} rows × {len(selected_features)} features',
    f'Unique trading dates: {len(unique_dates)}',
    f'CV folds: {len(res_df)}',
    '',
]
for _, row in res_df.iterrows():
    terminal_lines.append(
        f'Fold {int(row["fold"]):2d} | IC={row["ic"]:+.4f}  AUC={row["auc"]:.4f}  '
        f'train={int(row["n_train"]):,} ({int(row["train_days"])}d)  '
        f'test={int(row["n_test"]):,}  '
        f'[{row["test_start"]} -> {row["test_end"]}]'
    )
terminal_lines += [
    '',
    f'Mean per-date IC: {res_df.ic.mean():.4f}  (ICIR={res_df.ic.mean()/res_df.ic.std():.2f})',
    f'OOF IC pooled : {ic_oof_pooled:+.4f}',
    f'OOF IC perdate: {ic_oof:+.4f}',
    '',
    '-- Backtest --',
    f'Mode               : {mode}',
    f'Annual Sharpe (net): {sharpe:.2f}  (gross: {sharpe_gross:.2f})',
    f'Annual return (net): {ann_return*100:.1f}%  (gross: {ann_ret_gross*100:.1f}%)',
    f'Max drawdown       : {max_dd*100:.1f}%',
    f'Realised vol       : {realised_vol*100:.1f}%  avg lev: {avg_lev:.2f}',
    f'Holding periods    : {len(perf)}',
    f'CB trips           : {cb_trips}',
    '',
    '-- SHAP top features --',
]
for _, r in shap_importance.iterrows():
    terminal_lines.append(f'  {r["feature"]:<20s} {r["mean_|SHAP|"]:.6f}')
terminal_lines.append('')
terminal_lines.append('Pipeline v4.1 complete')

dashboard_data = {
    'generated_at': str(pd.Timestamp.now()),
    'pipeline_version': 'v4.9',
    'config': config, 'metrics': metrics, 'equity': equity_data,
    'folds': fold_data, 'shap': shap_data,
    'top_stocks': top_stocks_data, 'top20': top_stocks_data,
    'oof_scatter': oof_scatter, 'rebal_details': rebal_details,
    'region_breakdown': region_counts,
    'selected_features': selected_features,
    'all_features': list(passed_features),
    'terminal': terminal_lines,
}

# Alpha analysis: quantile returns, long-short spread, rolling IC,
# turnover, factor autocorr, monthly returns
N_QUANTILES = 10
quantile_returns_list = [{'label': f'D{q}', 'mean_return': 0.0, 'count': 0, 'total': 0.0}
                         for q in range(1, N_QUANTILES + 1)]
for rd in rebal_dates:
    grp = bt[bt['date'] == rd].copy()
    if len(grp) < N_QUANTILES: continue
    grp['q'] = pd.qcut(grp['signal'], N_QUANTILES, labels=False, duplicates='drop') + 1
    for q in range(1, N_QUANTILES + 1):
        s = grp[grp['q'] == q]
        if len(s):
            quantile_returns_list[q-1]['total'] += s['actual'].mean()
            quantile_returns_list[q-1]['count'] += 1
for q in quantile_returns_list:
    q['mean_return'] = round(q['total'] / max(q['count'], 1), 6)
    del q['total'], q['count']

ls_dates, ls_top, ls_bot = [], [], []
for rd in rebal_dates:
    grp = bt[bt['date'] == rd]
    if len(grp) < 20: continue
    th_hi = grp['signal'].quantile(1 - TOP_DECILE)
    th_lo = grp['signal'].quantile(TOP_DECILE)
    t = grp[grp['signal'] >= th_hi]; b = grp[grp['signal'] <= th_lo]
    if len(t) == 0 or len(b) == 0: continue
    ls_dates.append(str(pd.Timestamp(rd).date()))
    ls_top.append(float(t['actual'].mean()))
    ls_bot.append(float(b['actual'].mean()))
cum_top = list(np.cumprod(1 + np.array(ls_top)))
cum_bot = list(np.cumprod(1 + np.array(ls_bot)))
long_short_data = {
    'dates': ls_dates,
    'cum_top':    [round(v, 4) for v in cum_top],
    'cum_bottom': [round(v, 4) for v in cum_bot],
    'cum_spread': [round(t - b, 4) for t, b in zip(cum_top, cum_bot)],
}

ric_dates, ric_values = [], []
for rd in rebal_dates:
    grp = bt[bt['date'] == rd]
    if len(grp) < 10: continue
    c, _ = spearmanr(grp['signal'].values, grp['actual'].values)
    if np.isfinite(c):
        ric_dates.append(str(pd.Timestamp(rd).date()))
        ric_values.append(round(float(c), 4))
ric_cum_mean = []
s_ = 0
for i, v in enumerate(ric_values):
    s_ += v; ric_cum_mean.append(round(s_ / (i + 1), 4))
rolling_ic_data = {'dates': ric_dates, 'values': ric_values, 'cumulative_mean': ric_cum_mean}

turnover_dates, turnover_values = [], []
prev_port = set()
for rd in rebal_dates:
    grp = bt[bt['date'] == rd]
    if len(grp) < 10: continue
    th = grp['signal'].quantile(1 - TOP_DECILE)
    curr = set(grp[grp['signal'] >= th]['ticker'].tolist())
    if prev_port:
        union = prev_port | curr; inter = prev_port & curr
        t = 1 - len(inter) / len(union) if union else 0
        turnover_dates.append(str(pd.Timestamp(rd).date()))
        turnover_values.append(round(float(t), 4))
    prev_port = curr
turnover_data = {'dates': turnover_dates, 'values': turnover_values}

ac_dates, ac_values = [], []
prev_sig = None
for rd in rebal_dates:
    grp = bt[bt['date'] == rd][['ticker','signal']].copy()
    if len(grp) < 10: continue
    grp = grp.set_index('ticker')['signal']
    if prev_sig is not None:
        common = grp.index.intersection(prev_sig.index)
        if len(common) >= 10:
            c, _ = spearmanr(grp.loc[common].values, prev_sig.loc[common].values)
            if np.isfinite(c):
                ac_dates.append(str(pd.Timestamp(rd).date()))
                ac_values.append(round(float(c), 4))
    prev_sig = grp
autocorr_data = {'dates': ac_dates, 'values': ac_values}

perf_s = perf.copy()
if not isinstance(perf_s.index, pd.DatetimeIndex):
    perf_s.index = pd.to_datetime(perf_s.index)
monthly_cum = (1 + perf_s).groupby([perf_s.index.year, perf_s.index.month]).prod() - 1
monthly_returns_data = [{'year': int(y), 'month': int(m), 'ret': round(float(r), 4)}
                        for (y, m), r in monthly_cum.items()]

downside = perf.values[perf.values < 0]
downside_std = np.std(downside) if len(downside) else np.std(perf.values)
sortino = float(np.mean(perf.values) / downside_std * np.sqrt(periods_per_year)) \
          if downside_std > 0 else 0
calmar  = float(ann_return / abs(max_dd)) if abs(max_dd) > 0 else 0

dashboard_data['quantile_returns']       = quantile_returns_list
dashboard_data['long_short']             = long_short_data
dashboard_data['rolling_ic']             = rolling_ic_data
dashboard_data['turnover']               = turnover_data
dashboard_data['factor_autocorrelation'] = autocorr_data
dashboard_data['monthly_returns']        = monthly_returns_data
dashboard_data['metrics']['sortino'] = round(sortino, 2)
dashboard_data['metrics']['calmar']  = round(calmar, 2)

# v4.3 — decile calibration history
if len(decile_log) > 0:
    dashboard_data['decile_calibration'] = {
        'dates':     [str(pd.Timestamp(r['date']).date()) for r in decile_log],
        'long':      [r['long'] for r in decile_log],
        'short':     [r['short'] for r in decile_log],
        'final_expected_returns': (
            decile_returns_history.tail(CALIB_LOOKBACK).mean(axis=0).fillna(0).round(5).to_dict()
            if len(decile_returns_history) > 0 else {}
        ),
    }

# v4.10 — Kelly leverage trace per rebal (numeric fields are JS-safe)
def _safe_num(x, default=0.0):
    try:
        v = float(x)
        return round(v, 4) if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default

dashboard_data['kelly_trace'] = {
    'dates': [str(pd.Timestamp(r['date']).date()) for r in period_returns],
    'leverage': [_safe_num(r.get('lev', 1.0), 1.0) for r in period_returns],
    'basket_return': [
        _safe_num(top_decile_rets_history[i] if i < len(top_decile_rets_history) else 0.0, 0.0)
        for i in range(len(period_returns))
    ],
}

# v4.12 — per-name Kelly allocation (Portfolio view)
dashboard_data['kelly_allocation'] = dashboard_data_kelly_alloc_extra

out_path = BASE / 'dashboard_data_v4.json'

# v4.12 — recursively replace NaN/Inf with 0 before serialising so the JS
# dashboard's toFixed/Number calls don't crash on non-numeric tokens.
def _scrub_nans(obj):
    if isinstance(obj, dict):
        return {k: _scrub_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_nans(v) for v in obj]
    if isinstance(obj, float):
        if not np.isfinite(obj):
            return 0.0
        return obj
    return obj

dashboard_data = _scrub_nans(dashboard_data)

# v4.12 — ATOMIC write: dump to .tmp, flush+fsync, then os.replace.
# The earlier run wrote a truncated JSON (cut off mid decile_calibration array)
# which the dashboard parses as "invalid JSON | Cannot read property toFixed".
# Atomic rename guarantees the dashboard sees either the old full file or the
# new full file — never a half-written one.
import os as _os
_tmp_path = str(out_path) + '.tmp'
with open(_tmp_path, 'w') as f:
    json.dump(dashboard_data, f, indent=2, default=str, allow_nan=False)
    f.flush()
    _os.fsync(f.fileno())
_os.replace(_tmp_path, str(out_path))

print(f'\nDashboard data v4.12 -> {out_path}')
print(f'  Mode               : {mode}')
print(f'  Sharpe net / gross : {sharpe:.2f} / {sharpe_gross:.2f}')
print(f'  Return net / gross : {ann_return*100:.1f}% / {ann_ret_gross*100:.1f}%')
print(f'  Realised vol / lev : {realised_vol*100:.1f}% / {avg_lev:.2f}x')
if KELLY_CRITERION:
    _active = sum(1 for r in period_returns if r.get('lev',1.0) not in (1.0, MIN_LEVERAGE))
    print(f'  Kelly book lev    active in {_active}/{len(period_returns)} rebals at f={KELLY_FRACTION}')
if KELLY_ALLOC:
    print(f'  Kelly per-name alloc used in {len(allocation_log)} rebals')
print(f'  OOF IC perdate     : {ic_oof:+.4f}')
print(f'  Sortino / Calmar   : {sortino:.2f} / {calmar:.2f}')
try:
    _d_means = decile_returns_history.mean()
    _spread = float((_d_means.get(N_DECILES, np.nan) - _d_means.get(1, np.nan)) * 100)
    print(f'  Quantile spread D{N_DECILES}-D1: {_spread:.3f}%')
except Exception:
    pass
print('\nOpen dashboard.html and drop dashboard_data_v4.json onto it.')
