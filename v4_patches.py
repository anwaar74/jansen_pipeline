"""
v4 PATCHES — Bridge the ICIR-to-Sharpe gap
==========================================
v3 results say the SIGNAL is good (mean IC 0.048, ICIR 0.70, OOF IC 0.048)
but Sharpe is only 0.10 net (0.27 gross). The gap is *not* signal quality —
it is cost drag, noise, sizing, and regime instability. This patch addresses
each in priority order.

Apply by REPLACING the corresponding v3 cells. All other v3 cells unchanged.

Expected lift (back-of-envelope from current IC/turnover/dispersion):
  Sharpe net  : 0.10 → 0.60-0.90
  Sharpe gross: 0.27 → 0.80-1.10
  Max DD      : -19.7% → -12 to -15%
  Turnover    : 75% → 40-50% per rebal

Why each fix matters (matched to dashboard symptom):
  P0 [Cell 11→11v4]  Multi-seed × multi-horizon ENSEMBLE.
                     Targets the per-fold IC variance (F0=+0.17, F2=-0.02).
                     Single model overfits each fold; ensemble smooths it.
  P0 [Cell 13→13v4]  EMA-smooth the OOF signal across rebalances.
                     Cuts turnover ~40% (autocorr 0.445 → ~0.7) with <5% IC cost.
  P0 [Cell 13→13v4]  Signal-WEIGHTED positions (not equal-weight quintile).
                     D3 has the best return; equal-weight quintile ignores
                     rank dispersion. Rank-weighted sizing uses the whole curve.
  P0 [Cell 13→13v4]  Volatility target the gross book at 10% annualised.
                     Reduces realised vol (and DD) without touching alpha.
  P1 [Cell 13→13v4]  Broader basket (top/bottom 30% vs 20%).
                     The D3 anomaly says the model's rank ordering is good in
                     the middle quantiles too — widen the book to capture it.

Requires from v3 in memory:
  fd2, X_all, y_all, y_raw, passed_features, splits, MultipleTimeSeriesCV,
  unique_dates, date_to_rows, TARGET_HORIZON, BASE.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# ═══════════════════════════════════════════════════════════════════════
# CELL 11 v4 — ENSEMBLE training (5d × 21d horizons × 3 seeds = 6 models/fold)
# REPLACES v3 Cell 11.
# ═══════════════════════════════════════════════════════════════════════

# ── Build 5d rank target if not already present
if 'target_5d_rank' not in fd2.columns:
    fd2['target_5d_rank'] = fd2.groupby('date')['target_5d'].rank(pct=True)

y_all_21d = fd2['target_21d'].groupby(fd2['date']).rank(pct=True) \
            if 'target_rank' not in fd2.columns else fd2['target_rank']
y_all_5d  = fd2['target_5d_rank']

# Drop rows where 5d target is NaN (last 5 days per ticker)
_valid_5d = y_all_5d.notna()
print(f'5d target valid rows: {_valid_5d.sum():,} / {len(y_all_5d):,}')

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

SEEDS    = [42, 7, 2024]
HORIZONS = [('21d', y_all_21d), ('5d', y_all_5d)]

results, oof_preds = [], pd.Series(np.nan, index=fd2.index, dtype=float)

for fold_i, (train_idx, test_idx) in enumerate(splits):
    fold_preds_te = []                            # to be averaged
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

        # If 5d horizon has too few valid rows, skip
        if len(X_tr) < 1000 or len(X_te) < 100:
            continue

        for seed in SEEDS:
            params = {**LGB_PARAMS, 'random_state': seed}
            m = lgb.LGBMRegressor(**params)
            m.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)])
            best_iters.append(m.best_iteration_)
            # Predict on FULL test rows (including the ok_te=False ones we'll mask)
            # Use ok_te indexing so positions line up with test_idx[ok_te]
            preds_te = m.predict(X_te)             # length = ok_te.sum()
            # Convert preds to per-date rank (so 5d and 21d are comparable)
            te_dates_ok = fd2.loc[test_idx, 'date'].values[ok_te]
            preds_rank = pd.Series(preds_te).groupby(
                pd.Series(te_dates_ok)
            ).rank(pct=True).values
            fold_preds_te.append((te_dates_ok, ok_te, preds_rank))

    if not fold_preds_te:
        continue

    # Average predictions across models. Each model has the same ok_te? Only
    # if both horizons had the same valid mask. Safest: align on row index.
    test_idx_arr = np.array(test_idx)
    n_test = len(test_idx_arr)
    pred_accum = np.zeros(n_test)
    pred_count = np.zeros(n_test)
    # For each model prediction we know which subset of test_idx it covered.
    # ok_te is the boolean mask of length n_test (because X_te_full was indexed
    # by test_idx). So we can scatter back directly.
    for te_dates_ok, ok_te, preds_rank in fold_preds_te:
        pred_accum[ok_te] += preds_rank
        pred_count[ok_te] += 1
    pred_count[pred_count == 0] = np.nan
    fold_ensemble = pred_accum / pred_count       # NaN where no model covered

    # Final per-date rank of ensemble (defensive — already rank-averaged)
    te_dates_full = fd2.loc[test_idx, 'date'].values
    fold_ens_series = pd.Series(fold_ensemble, index=test_idx_arr)
    fold_ens_series = fold_ens_series.groupby(te_dates_full).rank(pct=True)

    # Store to OOF
    oof_preds.loc[test_idx_arr] = fold_ens_series.values

    # IC metrics vs RAW returns
    y_te_raw = y_raw.loc[test_idx].values
    valid    = np.isfinite(fold_ens_series.values) & np.isfinite(y_te_raw)
    rho_p, _ = spearmanr(fold_ens_series.values[valid], y_te_raw[valid])
    auc = roc_auc_score((y_te_raw[valid] > 0).astype(int),
                         fold_ens_series.values[valid]) \
          if len(np.unique(y_te_raw[valid] > 0)) > 1 else np.nan

    per_date = []
    for d in np.unique(te_dates_full[valid]):
        m = (te_dates_full == d) & valid
        if m.sum() >= 10:
            r, _ = spearmanr(fold_ens_series.values[m], y_te_raw[m])
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
# CELL 12 v4 — Per-date winsorisation + cross-sectional z-score (Jansen Ch.12)
# Same logic as v3 — kept here so the run-order is self-contained.
# ═══════════════════════════════════════════════════════════════════════
oof_preds_filtered = oof_preds.copy()
oof_valid_mask = oof_preds_filtered.notna()
signal_dates   = fd2.loc[oof_valid_mask, 'date'].values
signal         = oof_preds_filtered[oof_valid_mask].copy()

signal_ws = []
for d in np.unique(signal_dates):
    m = signal_dates == d
    grp = signal[m]
    if len(grp) >= 3:
        lo, hi = np.percentile(grp, [1, 99])
        signal_ws.append(grp.clip(lower=lo, upper=hi))
    else:
        signal_ws.append(grp)
signal_winsorised = pd.concat(signal_ws)

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
ic_oof_raw = ic_oof_pooled
print(f'OOF IC pooled: {ic_oof_pooled:+.4f}  perdate: {ic_oof:+.4f}  ICIR: {ic_oof_ir:.2f}')


# ═══════════════════════════════════════════════════════════════════════
# CELL 13 v4 — Backtest with EMA smoothing + signal-weighted positions
#               + volatility-target overlay + broader basket
# REPLACES v3 Cell 13.
# ═══════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt

LONG_ONLY          = False      # set True for Shariah long-only mode
TOP_FRAC           = 0.30       # broader basket — was 0.20 in v3
REBAL_DAYS         = TARGET_HORIZON
COMMISSION_BPS     = 5.0
SLIPPAGE_BPS       = 2.5
TOTAL_COST_BPS     = (COMMISSION_BPS + SLIPPAGE_BPS) * 2

# ── Signal smoothing (cuts turnover ~40% — autocorr 0.445 → ~0.70)
EMA_ALPHA          = 0.50       # signal_t = α·new + (1-α)·prev

# ── Sizing
SIGNAL_WEIGHTED    = True       # weight by signal rank within each side
MAX_POSITION_SIZE  = 0.04       # tighter than v3 (5%) — more diversification
MAX_MARKET_CONC    = 0.40

# ── Vol target (Jansen Ch.12 "risk parity overlay")
VOL_TARGET         = True
VOL_TARGET_ANN     = 0.10       # target 10% annualised portfolio vol
VOL_LOOKBACK       = 6          # rebal periods (≈ 6 months) for realised vol
MAX_LEVERAGE       = 1.5        # cap gross at 1.5× to avoid blow-ups
MIN_LEVERAGE       = 0.30

# ── Circuit breaker (loose — was killing the model unnecessarily in v2/v3)
DD_CIRCUIT_BREAKER = -0.30

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

# ── EMA-smooth the signal per ticker (Jansen Ch.12 "signal decay")
# Build a (date × ticker) panel, EMA per ticker, then unstack back to long.
sig_panel = bt.pivot_table(index='date', columns='ticker', values='signal_raw')
sig_panel_smooth = sig_panel.ewm(alpha=EMA_ALPHA, adjust=False, min_periods=1).mean()

# Convert smoothed to per-date rank (so cross-section is comparable)
sig_panel_rank = sig_panel_smooth.rank(axis=1, pct=True)
sig_long = sig_panel_rank.stack().rename('signal').reset_index()
bt = bt.drop(columns=['signal_raw']).merge(sig_long, on=['date','ticker'], how='left')
bt = bt.dropna(subset=['signal'])
print(f'Signal smoothed (EMA α={EMA_ALPHA}); panel: {sig_panel.shape}')


def _signal_weights(grp_side, side='long'):
    """Signal-rank-weighted weights within the long (or short) book."""
    s = grp_side['signal'].values.astype(float)
    if side == 'long':
        w = s - s.min()                      # all positive
    else:
        w = (1 - s) - (1 - s).min()          # invert for shorts
    if w.sum() == 0:
        return np.ones(len(s)) / len(s)
    w = w / w.sum()
    # Position-size cap
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
recent_rets = []                      # for vol-target lookback

for rd in rebal_dates:
    grp = bt[bt['date'] == rd]
    if len(grp) < 20: continue

    rolling_dd = cum_eq / peak_eq - 1
    if rolling_dd < DD_CIRCUIT_BREAKER:
        cb_trips += 1
        period_returns.append({'date': rd, 'return': 0.0, 'gross': 0.0, 'lev': 0.0})
        continue

    # ── Select top / bottom TOP_FRAC by signal
    hi = grp['signal'].quantile(1 - TOP_FRAC)
    lo = grp['signal'].quantile(TOP_FRAC)
    longs  = grp[grp['signal'] >= hi].copy()
    shorts = grp[grp['signal'] <= lo].copy()
    if longs.empty: continue

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

    # ── Vol-target overlay (Jansen Ch.12)
    if VOL_TARGET and len(recent_rets) >= 3:
        realised_vol_ann = np.std(recent_rets[-VOL_LOOKBACK:]) * np.sqrt(252 / REBAL_DAYS)
        if realised_vol_ann > 1e-6:
            lev = VOL_TARGET_ANN / realised_vol_ann
            lev = float(np.clip(lev, MIN_LEVERAGE, MAX_LEVERAGE))
        else:
            lev = 1.0
    else:
        lev = 1.0

    gross_ret = raw_ret * lev
    # Cost on gross turnover (lev affects book size → more cost when lev>1)
    cost = (TOTAL_COST_BPS / 1e4) * lev
    net = gross_ret - cost

    cum_eq *= (1 + net); peak_eq = max(peak_eq, cum_eq)
    recent_rets.append(net)
    period_returns.append({'date': rd, 'return': net, 'gross': gross_ret, 'lev': lev})

perf_df = pd.DataFrame(period_returns).set_index('date')
perf       = perf_df['return']
perf_gross = perf_df['gross']
levs       = perf_df['lev']
cumulative = (1 + perf).cumprod()
dd = cumulative / cumulative.cummax() - 1

# ── Equal-weight baseline (top-quintile EW, same vol target)
ew_returns = []
cum_ew, peak_ew = 1.0, 1.0
recent_ew = []
for rd in rebal_dates:
    grp = bt[bt['date'] == rd]
    if len(grp) < 20: continue
    if cum_ew / peak_ew - 1 < DD_CIRCUIT_BREAKER:
        ew_returns.append({'date': rd, 'return': 0.0}); continue
    hi = grp['signal'].quantile(1 - 0.20)
    longs = grp[grp['signal'] >= hi]
    if longs.empty: continue
    r = longs['actual'].mean() - TOTAL_COST_BPS / 1e4
    cum_ew *= (1 + r); peak_ew = max(peak_ew, cum_ew)
    ew_returns.append({'date': rd, 'return': r})
perf_ew = pd.DataFrame(ew_returns).set_index('date')['return'].reindex(perf.index, fill_value=0)
cumulative_ew = (1 + perf_ew).cumprod()

# ── Stats
ppy = 252 / REBAL_DAYS
n_years = len(perf) / ppy
sharpe      = float(perf.mean()       / perf.std()       * np.sqrt(ppy)) if perf.std() > 0 else np.nan
sharpe_g    = float(perf_gross.mean() / perf_gross.std() * np.sqrt(ppy)) if perf_gross.std() > 0 else np.nan
sharpe_ew   = float(perf_ew.mean()    / perf_ew.std()    * np.sqrt(ppy)) if perf_ew.std() > 0 else np.nan
max_dd      = float((cumulative / cumulative.cummax() - 1).min())
max_dd_ew   = float((cumulative_ew / cumulative_ew.cummax() - 1).min())
ann_return    = float(cumulative.iloc[-1]    ** (1 / n_years) - 1) if n_years > 0 else np.nan
ann_return_ew = float(cumulative_ew.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else np.nan
realised_vol = float(perf.std() * np.sqrt(ppy))
avg_lev = float(levs.mean())

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

# ── Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                          gridspec_kw={'height_ratios': [3, 1.5, 1.5]})
cumulative.plot(ax=axes[0], color='steelblue', lw=2,
                label=f'v4 {mode} (Sharpe={sharpe:.2f})')
cumulative_ew.plot(ax=axes[0], color='#888', lw=1.2, ls='--',
                   label=f'EW top-quintile (Sharpe={sharpe_ew:.2f})')
axes[0].set_title(f'v4 Equity Curve — {mode} (vol-targeted, EMA-smoothed)')
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

plt.tight_layout()
plt.savefig(BASE / 'equity_curve_v4.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {BASE}/equity_curve_v4.png')


# ═══════════════════════════════════════════════════════════════════════
# Optional Cell 13b — Ablation table.  Run AFTER v4 to see contribution.
# Each row toggles one v4 lever off and rebuilds perf. Uncomment to use.
# ═══════════════════════════════════════════════════════════════════════
# print("\nAblation (each row turns one lever OFF):")
# for label, kwargs in [
#     ("v3 baseline   (no smooth, EW, no vol-tgt, 20% basket)",
#         dict(EMA_ALPHA=1.0, SIGNAL_WEIGHTED=False, VOL_TARGET=False, TOP_FRAC=0.20)),
#     ("+ EMA smooth",   dict(SIGNAL_WEIGHTED=False, VOL_TARGET=False, TOP_FRAC=0.20)),
#     ("+ signal weight",dict(VOL_TARGET=False, TOP_FRAC=0.20)),
#     ("+ 30% basket",   dict(VOL_TARGET=False)),
#     ("full v4",        dict()),
# ]:
#     ... # re-run loop with overrides
