"""
v3 PATCH CELLS — Jansen MFAT 2e-aligned Sharpe improvements
============================================================
Paste each block as a notebook cell, in order, REPLACING the originals
referenced in the header. Old cell numbers refer to the v2 notebook.

Required upstream variables: feat_df, raw, LIQUIDITY, ml_start (='2020-01-01'),
TARGET_HORIZON, TRAIN_PERIOD, TEST_PERIOD, EMBARGO, N_SPLITS, MIN_TRAIN_DAYS.
"""

# ═══════════════════════════════════════════════════════════════════════
# CELL 8 (REPLACE) — Expanded feature engineering, Jansen Ch.4
# Adds: 12-1 momentum, short-term reversal, idio vol, volume shock,
#       drawdown depth, Amihud illiquidity. Keeps original features.
# ═══════════════════════════════════════════════════════════════════════
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

def compute_features_v3(g: pd.DataFrame, market_close: pd.Series | None = None) -> pd.DataFrame:
    c = g['close']; v = g['volume']; h = g['high']; l = g['low']
    out = pd.DataFrame(index=g.index)
    out['close'] = c

    # — original return / momentum features (kept for continuity)
    out['ret_1d']   = c.pct_change(1)
    out['ret_5d']   = c.pct_change(5)
    out['ret_21d']  = c.pct_change(21)
    out['ret_63d']  = c.pct_change(63)

    # — Jansen Ch.4 canonical alphas
    out['mom_12_1']      = c.shift(21) / c.shift(252) - 1     # 12-1 momentum (skip last month)
    out['reversal_5d']   = -c.pct_change(5)                    # short-term reversal
    out['reversal_21d']  = -c.pct_change(21)
    lr = np.log(c / c.shift(1))
    out['vol_21d']       = lr.rolling(21).std()
    out['vol_63d']       = lr.rolling(63).std()
    out['dd_depth_252d'] = c / c.rolling(252).max() - 1        # how deep below 1y high

    # — Idiosyncratic vol (residual of stock return vs market return)
    if market_close is not None:
        mret = np.log(market_close / market_close.shift(1)).reindex(g.index)
        cov  = lr.rolling(63).cov(mret)
        var_m = mret.rolling(63).var()
        beta  = cov / var_m.replace(0, np.nan)
        resid = lr - beta * mret
        out['idio_vol_63d'] = resid.rolling(63).std()
    else:
        out['idio_vol_63d'] = lr.rolling(63).std()  # fallback

    # — Trend
    sma50 = c.rolling(50).mean(); sma200 = c.rolling(200).mean()
    out['close_vs_sma50']  = c / sma50  - 1
    out['close_vs_sma200'] = c / sma200 - 1

    # — RSI / oscillator
    delta = c.diff(); gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs   = gain / loss.replace(0, np.nan)
    out['rsi_14'] = 100 - 100 / (1 + rs)

    # — Bollinger
    sma20 = c.rolling(20).mean(); std20 = c.rolling(20).std()
    upper = sma20 + 2*std20; lower = sma20 - 2*std20
    out['bb_width']    = (upper - lower) / sma20.replace(0, np.nan)
    out['bb_position'] = (c - lower) / (upper - lower).replace(0, np.nan)

    # — ATR
    tr  = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    out['atr_pct'] = tr.rolling(14).mean() / c

    # — Volume / liquidity (Jansen Ch.4)
    vol_sma20 = v.rolling(20).mean()
    vol_std20 = v.rolling(20).std()
    out['vol_ratio']    = v / vol_sma20.replace(0, np.nan)
    out['vol_shock']    = (v - vol_sma20) / vol_std20.replace(0, np.nan)
    out['amihud_21d']   = (lr.abs() / (c * v).replace(0, np.nan)).rolling(21).mean()

    # — Targets (raw + rank computed later cross-sectionally)
    out['target_5d']   = c.shift(-5)  / c - 1
    out['target_21d']  = c.shift(-21) / c - 1
    return out

# Build a per-market index proxy (equal-weighted close) for idio-vol residualisation
raw = raw if 'raw' in dir() else pd.read_parquet(RAW_PARQUET)
raw['date'] = pd.to_datetime(raw['date'])
mkt_proxy = (
    raw.groupby(['market', 'date'])['close']
       .mean()
       .groupby(level=0)
       .apply(lambda s: s / s.iloc[0])  # rebase per market
)

# Recompute features (overwrites the cached parquet; delete features_long_all.parquet first)
print('Recomputing v3 features ...')
frames = []
for ticker, g in raw.set_index('date').groupby('ticker'):
    g = g.sort_index()
    mkt = g['market'].iloc[0]
    mclose = mkt_proxy.loc[mkt].reindex(g.index)
    f = compute_features_v3(g, market_close=mclose)
    f['ticker'] = ticker; f['market'] = mkt
    adv30 = (g['close'] * g['volume']).rolling(30).mean()
    liq_threshold = LIQUIDITY.get(mkt, 500_000)
    f = f[adv30 >= liq_threshold]
    frames.append(f)

feat_df = pd.concat(frames).reset_index()
feat_df['date'] = pd.to_datetime(feat_df['date'])
float_cols = feat_df.select_dtypes(include=[np.floating]).columns
feat_df[float_cols] = feat_df[float_cols].replace([np.inf, -np.inf], np.nan)
print(f'feat_df v3: {feat_df.shape}')


# ═══════════════════════════════════════════════════════════════════════
# CELL 10 (REPLACE) — Skip Alphalens IC gate (it kills good alphas
# when measured with raw 5d/21d horizons on a regime-shifted window).
# Jansen Ch.6: let the tree model + early stopping select features.
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
# CELL 11 (REPLACE) — ML matrix with cross-sectional rank target + features
# (Jansen Ch.4 + Ch.12: rank-normalise per date to neutralise regime/beta)
# ═══════════════════════════════════════════════════════════════════════
ml_start = '2020-01-01'
fd2 = feat_df.copy()
fd2['date'] = pd.to_datetime(fd2['date'])
fd2 = fd2.sort_values('date').reset_index(drop=True)
fd2 = fd2[fd2['date'] >= ml_start]
fd2 = fd2.dropna(subset=passed_features + ['target_21d'])
fd2 = fd2.reset_index(drop=True)

# ── Cross-sectional rank-transform of FEATURES per date  (Jansen Ch.4)
print('Rank-transforming features per date ...')
fd2[passed_features] = (
    fd2.groupby('date')[passed_features]
       .transform(lambda s: s.rank(pct=True))
)

# ── Cross-sectional rank TARGET per date  (Jansen Ch.12)
fd2['target_rank'] = fd2.groupby('date')['target_21d'].rank(pct=True)

# ── Optional market-neutral residual target (for ablations)
fd2['target_demeaned'] = (
    fd2['target_21d']
    - fd2.groupby(['date', 'market'])['target_21d'].transform('mean')
)

X_all = fd2[passed_features].copy()
y_all = fd2['target_rank'].copy()          # train on rank target
y_raw = fd2['target_21d'].copy()           # keep raw for backtest

unique_dates = np.sort(fd2['date'].unique())
date_to_rows = {d: g.index.tolist() for d, g in fd2.groupby('date')}
print(f'X: {X_all.shape}  unique dates: {len(unique_dates)}  avg per date: {len(fd2)/len(unique_dates):.0f}')

# (Re-use the MultipleTimeSeriesCV class from v2 — splits unchanged)
cv = MultipleTimeSeriesCV(
    n_splits=N_SPLITS, train_period=TRAIN_PERIOD, test_period=TEST_PERIOD,
    purge=TARGET_HORIZON, embargo=EMBARGO,
)
splits = list(cv.split(X_all, dates=unique_dates, date_to_rows=date_to_rows))
print(f'Folds: {len(splits)}')


# ═══════════════════════════════════════════════════════════════════════
# CELL 12 (DELETE) — Remove LassoCV pre-selection entirely.
# Just set `selected_features = passed_features`.
# ═══════════════════════════════════════════════════════════════════════
selected_features = passed_features


# ═══════════════════════════════════════════════════════════════════════
# CELL 13 (REPLACE) — LightGBM on rank target + per-date Spearman early stop
# (Jansen Ch.12: optimise the metric you'll actually trade on)
# ═══════════════════════════════════════════════════════════════════════
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

LGB_PARAMS = dict(
    objective        = 'regression',
    metric           = 'rmse',
    n_estimators     = 1500,
    learning_rate    = 0.02,
    num_leaves       = 63,
    min_child_samples= 100,
    subsample        = 0.7,
    colsample_bytree = 0.7,
    reg_alpha        = 0.5,
    reg_lambda       = 2.0,
    n_jobs           = -1,
    random_state     = 42,
    verbosity        = -1,
)

results, oof_preds = [], pd.Series(np.nan, index=fd2.index)

for fold_i, (train_idx, test_idx) in enumerate(splits):
    X_tr = np.array(X_all.loc[train_idx],  dtype=float)
    y_tr = np.array(y_all.loc[train_idx],  dtype=float)
    X_te = np.array(X_all.loc[test_idx],   dtype=float)
    y_te = np.array(y_all.loc[test_idx],   dtype=float)
    y_te_raw = np.array(y_raw.loc[test_idx], dtype=float)

    ok_tr = np.isfinite(X_tr).all(axis=1) & np.isfinite(y_tr)
    ok_te = np.isfinite(X_te).all(axis=1) & np.isfinite(y_te)
    X_tr, y_tr = X_tr[ok_tr], y_tr[ok_tr]
    X_te, y_te = X_te[ok_te], y_te[ok_te]
    y_te_raw = y_te_raw[ok_te]

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    preds = model.predict(X_te)

    # IC vs RAW returns (what we actually trade on)
    rho_pooled, _ = spearmanr(preds, y_te_raw)

    # Per-date IC on the test fold
    te_dates = fd2.loc[test_idx, 'date'].values[ok_te]
    per_date = []
    for d in np.unique(te_dates):
        m = te_dates == d
        if m.sum() >= 10:
            r, _ = spearmanr(preds[m], y_te_raw[m])
            if np.isfinite(r): per_date.append(r)
    rho_perdate = float(np.mean(per_date)) if per_date else np.nan

    auc = roc_auc_score((y_te_raw > 0).astype(int), preds) if len(np.unique(y_te_raw>0))>1 else np.nan

    te_idx_arr = np.array(test_idx)[ok_te]
    oof_preds.iloc[te_idx_arr] = preds

    te_d = fd2.loc[test_idx, 'date']
    results.append(dict(
        fold=fold_i, ic_pooled=rho_pooled, ic_perdate=rho_perdate, auc=auc,
        n_train=ok_tr.sum(), n_test=ok_te.sum(),
        train_days=fd2.loc[train_idx,'date'].nunique(),
        test_start=te_d.min().date(), test_end=te_d.max().date(),
    ))
    print(f'Fold {fold_i} | IC(pooled)={rho_pooled:+.4f}  IC(perdate)={rho_perdate:+.4f}  '
          f'AUC={auc:.4f}  best_iter={model.best_iteration_}')

res_df = pd.DataFrame(results)
print(f'\nMean per-date IC: {res_df.ic_perdate.mean():.4f}  '
      f'(ICIR={res_df.ic_perdate.mean()/res_df.ic_perdate.std():.2f})')


# ═══════════════════════════════════════════════════════════════════════
# CELL 15 (REPLACE) — Long-SHORT backtest (Jansen Ch.12, market-neutral)
# If shorting is not allowed, set LONG_ONLY=True for the legacy long-only.
# ═══════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

LONG_ONLY          = False        # set True for Shariah long-only
TOP_DECILE         = 0.20         # broader basket → more diversification
REBAL_DAYS         = TARGET_HORIZON
TOTAL_COST_BPS     = 15.0
MAX_POSITION_SIZE  = 0.05
MAX_MARKET_CONC    = 0.40
DD_CIRCUIT_BREAKER = -0.25        # loosened (v2 was too trigger-happy)

bt = pd.DataFrame({
    'date':   fd2['date'].values,
    'ticker': fd2['ticker'].values,
    'market': fd2['market'].values,
    'actual': y_raw.values,
    'signal': oof_preds.values,
}).dropna(subset=['signal','actual'])

all_dates   = np.sort(bt['date'].unique())
rebal_dates = all_dates[::REBAL_DAYS]

period_returns = []
cum_eq, peak_eq, cb_trips = 1.0, 1.0, 0
for rd in rebal_dates:
    grp = bt[bt['date'] == rd]
    if len(grp) < 20: continue
    if (cum_eq/peak_eq - 1) < DD_CIRCUIT_BREAKER:
        cb_trips += 1
        period_returns.append({'date': rd, 'return': 0.0}); continue

    hi = grp['signal'].quantile(1 - TOP_DECILE)
    lo = grp['signal'].quantile(TOP_DECILE)
    longs  = grp[grp['signal'] >= hi]
    shorts = grp[grp['signal'] <= lo]

    if LONG_ONLY:
        long_ret = longs['actual'].mean()
        net = long_ret - TOTAL_COST_BPS/1e4
    else:
        long_ret  = longs['actual'].mean()
        short_ret = shorts['actual'].mean()
        # dollar-neutral 50/50 book; cost is on full gross turnover
        net = 0.5*long_ret - 0.5*short_ret - TOTAL_COST_BPS/1e4

    cum_eq *= (1 + net); peak_eq = max(peak_eq, cum_eq)
    period_returns.append({'date': rd, 'return': net})

perf = pd.DataFrame(period_returns).set_index('date')['return']
cumret = (1 + perf).cumprod()
ppy = 252 / REBAL_DAYS
n_years = len(perf) / ppy
sharpe = perf.mean()/perf.std()*np.sqrt(ppy) if perf.std()>0 else np.nan
max_dd = float((cumret/cumret.cummax() - 1).min())
ann_ret = cumret.iloc[-1]**(1/n_years) - 1 if n_years > 0 else np.nan

print(f'\n{"":<25}  {"Long-Short" if not LONG_ONLY else "Long-Only"}')
print(f'{"Annual Sharpe":<25}  {sharpe:>8.2f}')
print(f'{"Annual Return":<25}  {ann_ret*100:>7.1f}%')
print(f'{"Max Drawdown":<25}  {max_dd*100:>7.1f}%')
print(f'{"CB trips":<25}  {cb_trips:>8d} / {len(rebal_dates)}')
print(f'{"Periods":<25}  {len(perf):>8d}')


# ═══════════════════════════════════════════════════════════════════════
# OPTIONAL — Multi-horizon ensemble (Jansen Ch.12 "horizon ensembling")
# Train second model on target_5d_rank, average rank-scaled predictions.
# ═══════════════════════════════════════════════════════════════════════
# fd2['target_5d_rank'] = fd2.groupby('date')['target_5d'].rank(pct=True)
# y_all_5d = fd2['target_5d_rank']
# … repeat CELL 13 training with y_all_5d → oof_preds_5d
# ensemble_signal = (oof_preds.rank(pct=True) + oof_preds_5d.rank(pct=True)) / 2
# Use `ensemble_signal` in the backtest instead of `oof_preds`.
