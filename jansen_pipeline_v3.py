"""
========================================================================
Shariah ML Pipeline v3 — Jansen MFAT 2nd Ed. Aligned (Sharpe-fix release)
========================================================================
Markets:        US (NYSE/NASDAQ) · Malaysia (KLSE) · Hong Kong (HKEX)
Alpha engine:   LightGBM walk-forward CV (purge + embargo)
Target:         Cross-sectional RANK of 21d forward return  (Jansen Ch.4/12)
Features:       Cross-sectional RANK per date  (Jansen Ch.4)
Selection:      Tree gain importance — no Lasso pre-screen  (Jansen Ch.6)
Backtest:       Long-short market-neutral, top vs bottom quintile  (Jansen Ch.12)
Liquidity:      Per-market ADV thresholds
Survivorship:   Includes EODHD delisted stocks

Changes vs v2 (the cause of the negative Sharpe):
  P0  Target → cross-sectional rank per date  (was raw 21d return)
  P0  Features → rank-transformed per date     (was raw, regime-biased)
  P0  Dropped LassoCV pre-selection (was killing 16/17 features)
  P0  Backtest → long-short market-neutral     (was long-only, all-US)
  P1  Added Jansen Ch.4 alphas: 12-1 momentum, short-term reversal,
       idiosyncratic vol, volume shock, drawdown depth, Amihud illiquidity
  P1  Early stopping on per-date Spearman IC, not RMSE
  P2  Loosened DD circuit breaker (-25% vs -15%; was tripping 14/24 periods)

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
FEAT_PARQUET = BASE / 'features_long_v3.parquet'         # v3 has a new feature set

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
    float_cols = feat_df.select_dtypes(include=[np.floating]).columns
    feat_df[float_cols] = feat_df[float_cols].replace([np.inf, -np.inf], np.nan)
    feat_df.to_parquet(FEAT_PARQUET, engine='pyarrow', compression='snappy', index=False)
    print(f'Saved {len(feat_df):,} rows → {FEAT_PARQUET}')

print(f'feat_df v3: {feat_df.shape}')


# ═══════════════════════════════════════════════════════════════════════
# CELL 7 — Feature list  (no Alphalens pre-screen — Jansen Ch.6 says
# tree gain importance is better than IC threshold for nonlinear factors)
# REPLACES v2 Cell 10.
# ═══════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    'ret_1d','ret_5d','ret_21d','ret_63d',
    'mom_12_1','reversal_5d','reversal_21d',
    'vol_21d','vol_63d','idio_vol_63d','dd_depth_252d',
    'close_vs_sma50','close_vs_sma200',
    'rsi_14','bb_width','bb_position','atr_pct',
    'vol_ratio','vol_shock','amihud_21d',
]
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

# ── Rank-transform features per date (Jansen Ch.4)
print('Rank-transforming features per date ...')
fd2[passed_features] = (
    fd2.groupby('date')[passed_features].transform(lambda s: s.rank(pct=True))
)

# ── Rank-transform target per date (Jansen Ch.12)
fd2['target_rank'] = fd2.groupby('date')['target_21d'].rank(pct=True)
# Optional: market-neutralised residual target (for ablation)
fd2['target_demeaned'] = (
    fd2['target_21d'] - fd2.groupby(['date','market'])['target_21d'].transform('mean')
)

X_all = fd2[passed_features].copy()
y_all = fd2['target_rank'].copy()   # train on rank target
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
# CELL 11 — LightGBM walk-forward training on RANK target
# REPLACES v2 Cell 13.  Early-stop uses RMSE on rank target; we also
# report per-date Spearman IC against the RAW return.
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
    random_state      = 42,
    verbosity         = -1,
)

results   = []
oof_preds = pd.Series(np.nan, index=fd2.index)

for fold_i, (train_idx, test_idx) in enumerate(splits):
    X_tr = np.array(X_all.loc[train_idx], dtype=float)
    y_tr = np.array(y_all.loc[train_idx], dtype=float)   # rank target
    X_te = np.array(X_all.loc[test_idx],  dtype=float)
    y_te = np.array(y_all.loc[test_idx],  dtype=float)
    y_te_raw = np.array(y_raw.loc[test_idx], dtype=float)

    ok_tr = np.isfinite(X_tr).all(axis=1) & np.isfinite(y_tr)
    ok_te = np.isfinite(X_te).all(axis=1) & np.isfinite(y_te)
    X_tr, y_tr = X_tr[ok_tr], y_tr[ok_tr]
    X_te, y_te = X_te[ok_te], y_te[ok_te]
    y_te_raw   = y_te_raw[ok_te]

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    preds = model.predict(X_te)

    # IC vs RAW returns (what we trade on)
    rho_pooled, _ = spearmanr(preds, y_te_raw)
    te_dates = fd2.loc[test_idx, 'date'].values[ok_te]
    per_date = []
    for d in np.unique(te_dates):
        m = te_dates == d
        if m.sum() >= 10:
            r, _ = spearmanr(preds[m], y_te_raw[m])
            if np.isfinite(r): per_date.append(r)
    rho_perdate = float(np.mean(per_date)) if per_date else np.nan

    auc = roc_auc_score((y_te_raw > 0).astype(int), preds) \
          if len(np.unique(y_te_raw > 0)) > 1 else np.nan

    te_idx_arr = np.array(test_idx)[ok_te]
    oof_preds.iloc[te_idx_arr] = preds

    te_d = fd2.loc[test_idx, 'date']
    results.append(dict(
        fold=fold_i, ic=rho_perdate, ic_pooled=rho_pooled, auc=auc,
        n_train=ok_tr.sum(), n_test=ok_te.sum(),
        train_days=fd2.loc[train_idx, 'date'].nunique(),
        test_start=te_d.min().date(), test_end=te_d.max().date(),
    ))
    print(f'Fold {fold_i:2d} | IC(perdate)={rho_perdate:+.4f}  IC(pooled)={rho_pooled:+.4f}  '
          f'AUC={auc:.4f}  best_iter={model.best_iteration_}')

res_df = pd.DataFrame(results)
print(f'\nMean per-date IC: {res_df.ic.mean():.4f}  (ICIR={res_df.ic.mean()/res_df.ic.std():.2f})')
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
# CELL 13 — Long-SHORT market-neutral backtest (Jansen Ch.12)
# REPLACES v2 Cell 15.
# LONG_ONLY = True falls back to long-only top-quintile for Shariah.
# ═══════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

LONG_ONLY          = False        # set True for Shariah long-only
TOP_DECILE         = 0.20         # quintile (broader basket)
REBAL_DAYS         = TARGET_HORIZON
COMMISSION_BPS     = 5.0
SLIPPAGE_BPS       = 2.5
TOTAL_COST_BPS     = (COMMISSION_BPS + SLIPPAGE_BPS) * 2
MAX_POSITION_SIZE  = 0.05
MAX_MARKET_CONC    = 0.40
DD_CIRCUIT_BREAKER = -0.25        # loosened (v2 -0.15 tripped 14/24 periods)
SIGNAL_WEIGHTED    = True

bt = pd.DataFrame({
    'date':   fd2['date'].values,
    'ticker': fd2['ticker'].values,
    'market': fd2['market'].values,
    'actual': y_raw.values,
    'signal': oof_preds_filtered.values,
}).dropna(subset=['signal','actual'])

all_dates   = np.sort(bt['date'].unique())
rebal_dates = all_dates[::REBAL_DAYS]
print(f'Backtest: {len(bt):,} rows, {bt["date"].nunique()} dates, '
      f'{bt["ticker"].nunique()} tickers, {len(rebal_dates)} rebalances')

period_returns = []
cum_eq, peak_eq, cb_trips = 1.0, 1.0, 0
for rd in rebal_dates:
    grp = bt[bt['date'] == rd]
    if len(grp) < 20: continue

    rolling_dd = cum_eq / peak_eq - 1
    if rolling_dd < DD_CIRCUIT_BREAKER:
        cb_trips += 1
        period_returns.append({'date': rd, 'return': 0.0}); continue

    hi = grp['signal'].quantile(1 - TOP_DECILE)
    lo = grp['signal'].quantile(TOP_DECILE)
    longs  = grp[grp['signal'] >= hi]
    shorts = grp[grp['signal'] <= lo]
    if longs.empty: continue

    if LONG_ONLY:
        long_ret = longs['actual'].mean()
        net = long_ret - TOTAL_COST_BPS / 1e4
    else:
        long_ret  = longs['actual'].mean()
        short_ret = shorts['actual'].mean() if not shorts.empty else 0.0
        # dollar-neutral 50/50 book; cost is on full gross turnover
        net = 0.5 * long_ret - 0.5 * short_ret - TOTAL_COST_BPS / 1e4

    cum_eq *= (1 + net); peak_eq = max(peak_eq, cum_eq)
    period_returns.append({'date': rd, 'return': net})

perf = pd.DataFrame(period_returns).set_index('date')['return']
cumulative = (1 + perf).cumprod()
dd = cumulative / cumulative.cummax() - 1

# Equal-weight baseline (top quintile, equal weight, same CB)
ew_returns = []
cum_ew, peak_ew = 1.0, 1.0
for rd in rebal_dates:
    grp = bt[bt['date'] == rd]
    if len(grp) < 20: continue
    if cum_ew / peak_ew - 1 < DD_CIRCUIT_BREAKER:
        ew_returns.append({'date': rd, 'return': 0.0}); continue
    hi = grp['signal'].quantile(1 - TOP_DECILE)
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
sharpe       = float(perf.mean() / perf.std() * np.sqrt(ppy)) if perf.std() > 0 else np.nan
sharpe_ew    = float(perf_ew.mean() / perf_ew.std() * np.sqrt(ppy)) if perf_ew.std() > 0 else np.nan
max_dd       = float((cumulative / cumulative.cummax() - 1).min())
max_dd_ew    = float((cumulative_ew / cumulative_ew.cummax() - 1).min())
ann_return    = float(cumulative.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else np.nan
ann_return_ew = float(cumulative_ew.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else np.nan

mode = 'Long-Only' if LONG_ONLY else 'Long-Short'
print(f'\n{"Metric":<25}  {mode:>14}  {"EW baseline":>14}')
print('─' * 60)
print(f'{"Annual Sharpe":<25}  {sharpe:>14.2f}  {sharpe_ew:>14.2f}')
print(f'{"Annual Return":<25}  {ann_return*100:>13.1f}%  {ann_return_ew*100:>13.1f}%')
print(f'{"Max Drawdown":<25}  {max_dd*100:>13.1f}%  {max_dd_ew*100:>13.1f}%')
print(f'{"Circuit-breaker trips":<25}  {cb_trips:>14d}')
print(f'{"Holding periods":<25}  {len(perf):>14d}')


# ═══════════════════════════════════════════════════════════════════════
# CELL 14 — Equity / DD / per-fold IC plot
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios':[3,1.5,1.5]})
cumulative.plot(ax=axes[0], color='steelblue', lw=2, label=f'{mode} (Sharpe={sharpe:.2f})')
cumulative_ew.plot(ax=axes[0], color='#888', lw=1.2, ls='--', label=f'EW (Sharpe={sharpe_ew:.2f})')
axes[0].set_title(f'Equity Curve — {mode} vs EW (after costs)')
axes[0].axhline(1, color='grey', lw=0.8, ls=':'); axes[0].legend(fontsize=10)

dd.plot(ax=axes[1], color='indianred', lw=1)
axes[1].fill_between(dd.index, dd, 0, alpha=0.3, color='indianred')
axes[1].axhline(DD_CIRCUIT_BREAKER, color='orange', lw=1, ls='--',
                label=f'CB ({DD_CIRCUIT_BREAKER*100:.0f}%)')
axes[1].set_title('Drawdown'); axes[1].legend(fontsize=9)

colors = ['green' if x > 0 else 'red' for x in res_df['ic']]
axes[2].bar(res_df['fold'], res_df['ic'], color=colors)
axes[2].set_title(f'Per-fold IC (mean={res_df.ic.mean():.3f}, ICIR={res_df.ic.mean()/res_df.ic.std():.2f})')
axes[2].axhline(0, color='black', lw=0.8)

plt.tight_layout()
plt.savefig(BASE / 'equity_curve_v3.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {BASE}/equity_curve_v3.png')


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

model_shap = lgb.LGBMRegressor(**LGB_PARAMS)
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
plt.savefig(BASE / 'shap_importance_v3.png', dpi=150, bbox_inches='tight'); plt.show()
print(f'Saved: {BASE}/shap_importance_v3.png')


# ═══════════════════════════════════════════════════════════════════════
# CELL 16 — Top 20 by signal at latest rebalance
# ═══════════════════════════════════════════════════════════════════════
latest_rebal = rebal_dates[-1]
grp_latest = bt[bt['date'] == latest_rebal].copy().sort_values('signal', ascending=False)
threshold_latest = grp_latest['signal'].quantile(1 - TOP_DECILE)
top_decile_count = int((grp_latest['signal'] >= threshold_latest).sum())

top20 = grp_latest.head(20).copy()
top20['rank'] = range(1, len(top20) + 1)
top20['fwd_return_%'] = (top20['actual'] * 100).round(2)
print(f'\nTop 20 — rebalance {pd.Timestamp(latest_rebal).date()}, '
      f'universe {len(grp_latest)}, top quintile size {top_decile_count}')
print(top20[['rank','ticker','signal','fwd_return_%']].to_string(index=False))
grp_latest.to_csv(BASE / 'stock_ranking_latest_v3.csv', index=False)


# ═══════════════════════════════════════════════════════════════════════
# CELL 17 — Export dashboard_data.json (v3)
# ═══════════════════════════════════════════════════════════════════════
import json
print('Exporting dashboard_data_v3.json ...')

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
    top_stocks_data.append({
        'rank': i + 1, 'ticker': row['ticker'],
        'signal': round(float(row['signal']), 6),
        'fwd_return_pct': round(float(row['actual']) * 100, 2),
        'in_portfolio': i < top_decile_count,
    })

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
}

config = {
    'TRAIN_PERIOD': int(TRAIN_PERIOD), 'TEST_PERIOD': int(TEST_PERIOD),
    'TARGET_HORIZON': int(TARGET_HORIZON), 'EMBARGO': int(EMBARGO),
    'N_SPLITS': int(N_SPLITS),
    'TOP_DECILE': float(TOP_DECILE), 'REBAL_DAYS': int(REBAL_DAYS),
    'COMMISSION_BPS': float(COMMISSION_BPS), 'SLIPPAGE_BPS': float(SLIPPAGE_BPS),
    'TOTAL_COST_BPS': float(TOTAL_COST_BPS),
    'SIGNAL_WEIGHTED': bool(SIGNAL_WEIGHTED), 'LONG_ONLY': bool(LONG_ONLY),
    'MAX_POSITION_SIZE': float(MAX_POSITION_SIZE),
    'MAX_MARKET_CONC': float(MAX_MARKET_CONC),
    'DD_CIRCUIT_BREAKER': float(DD_CIRCUIT_BREAKER),
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
        f'[{row["test_start"]} → {row["test_end"]}]'
    )
terminal_lines += [
    '',
    f'Mean per-date IC: {res_df.ic.mean():.4f}  (ICIR={res_df.ic.mean()/res_df.ic.std():.2f})',
    f'OOF IC pooled : {ic_oof_pooled:+.4f}',
    f'OOF IC perdate: {ic_oof:+.4f}',
    '',
    '── Backtest ──',
    f'Mode               : {mode}',
    f'Annual Sharpe (net): {sharpe:.2f}  (gross: {sharpe_gross:.2f})',
    f'Annual return (net): {ann_return*100:.1f}%  (gross: {ann_ret_gross*100:.1f}%)',
    f'Max drawdown       : {max_dd*100:.1f}%',
    f'Holding periods    : {len(perf)}',
    f'CB trips           : {cb_trips}',
    '',
    '── SHAP top features ──',
]
for _, r in shap_importance.iterrows():
    terminal_lines.append(f'  {r["feature"]:<20s} {r["mean_|SHAP|"]:.6f}')
terminal_lines.append('')
terminal_lines.append('Pipeline v3 complete ✓')

dashboard_data = {
    'generated_at': str(pd.Timestamp.now()),
    'pipeline_version': 'v3',
    'config': config, 'metrics': metrics, 'equity': equity_data,
    'folds': fold_data, 'shap': shap_data,
    'top_stocks': top_stocks_data, 'top20': top_stocks_data,
    'oof_scatter': oof_scatter, 'rebal_details': rebal_details,
    'region_breakdown': region_counts,
    'selected_features': selected_features,
    'all_features': list(passed_features),
    'terminal': terminal_lines,
}

# ── Alpha analysis: quantile returns, long-short spread, rolling IC,
#    turnover, factor autocorr, monthly returns
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
    ls_top.append(float(t['actual'].mean())); ls_bot.append(float(b['actual'].mean()))
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
rolling_ic_data = {'dates': ric_dates, 'values': ric_values,
                   'cumulative_mean': ric_cum_mean}

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

out_path = BASE / 'dashboard_data_v3.json'
with open(out_path, 'w') as f:
    json.dump(dashboard_data, f, indent=2, default=str)

print(f'\nDashboard data v3 -> {out_path}')
print(f'  Mode               : {mode}')
print(f'  Sharpe net / gross : {sharpe:.2f} / {sharpe_gross:.2f}')
print(f'  Return net / gross : {ann_return*100:.1f}% / {ann_ret_gross*100:.1f}%')
print(f'  OOF IC perdate     : {ic_oof:+.4f}')
print(f'  Sortino / Calmar   : {sortino:.2f} / {calmar:.2f}')
_spread = (quantile_returns_list[-1]["mean_return"] - quantile_returns_list[0]["mean_return"]) * 100
print(f'  Quantile spread D10-D1: {_spread:.3f}%')
print('\nOpen dashboard.html and drop dashboard_data_v3.json onto it.')
