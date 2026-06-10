# Pipeline v3 — Sharpe Improvement Plan (Jansen-Aligned)

## Diagnosis (what the dashboard is actually telling us)

The dashboard says Sharpe = -0.20, but the key numbers underneath tell a much sharper story:

| Metric | Value | What it means |
|---|---|---|
| `sharpe_ew` (equal-weight top decile) | **1.55** | The universe + monthly rebal cadence is *already* good. |
| `sharpe` (signal-weighted) | **-0.20** | The ML signal is **destroying** alpha that exists in the data. |
| `quantile_returns` D1 vs D10 | 3.06% vs 1.58% | The model is **anti-predictive**: its "best" decile underperforms its "worst". |
| `oof_signal_ic_pooled` | **-0.0027** | Pooled OOF IC is negative — confirms inversion. |
| `n_features_selected` / `n_features_total` | **1 / 17** | LassoCV killed every feature except `vol_mom`. |
| `circuit_trips` | **14 / 24 periods** | The drawdown breaker is firing more often than it's flat — it's masking, not protecting. |
| `region_breakdown` | `{US: 22}` | MY/HK never get picked — diversification is theoretical only. |

**Root cause:** the target (`target_21d` = raw 21d forward return) is dominated by market beta and regime noise. LassoCV on raw forward returns collapses to one feature; LightGBM trained with MSE on raw returns then optimises for explaining beta — not cross-sectional rank — and the top decile ends up correlated with high-beta names *that already moved*, which mean-revert.

Jansen MFAT 2e is explicit on this — Ch. 4 ("Alpha Factor Research") and Ch. 12 ("Backtesting") both prescribe **cross-sectional rank-residualised targets** and **rank-normalised features per date** for any cross-sectional ML signal. The current code does neither.

## Fixes, ranked by expected Sharpe impact

### P0 — Fix the target (Jansen Ch. 4, Ch. 12)

The single highest-impact change. Replace raw `target_21d` with a cross-sectional rank target.

```python
# Per-date rank in [0, 1] — Jansen "rank IC" framing
fd2['target_rank'] = (
    fd2.groupby('date')['target_21d']
       .rank(pct=True)
)
# Or: market-neutral residual target (subtract market mean per date)
fd2['target_demeaned'] = (
    fd2['target_21d']
    - fd2.groupby(['date','market'])['target_21d'].transform('mean')
)
```

Train on `target_rank` (or `target_demeaned`). This alone should flip pooled OOF IC from −0.003 to clearly positive.

### P0 — Drop LassoCV pre-selection

LassoCV on raw forward returns is what produced "1/17 features kept". Jansen Ch. 6 / Ch. 12 uses **gain-based importance from the tree model itself** (or SHAP) — not L1 pre-filtering. Remove Cell 12 entirely, pass all 17 features to LightGBM, and let regularisation + early stopping do the selection. Use `feature_importance` post-hoc for the dashboard.

### P0 — Use a ranking objective

Cross-sectional alpha is a ranking problem, not a regression problem (Jansen Ch. 12, "Predicting Stock Returns with Boosted Trees"):

```python
LGB_PARAMS = dict(
    objective    = 'regression',   # keep regression but on rank target
    metric       = 'rmse',
    n_estimators = 800,
    learning_rate= 0.02,           # lower LR + more trees
    num_leaves   = 63,
    min_child_samples = 100,       # bigger leaves → less overfit
    subsample    = 0.7,
    colsample_bytree = 0.7,
    reg_alpha    = 0.5,
    reg_lambda   = 2.0,
)
```

Alternative: keep raw target but switch to `objective='rank_xendcg'` with `group=` set to per-date row counts.

### P1 — Cross-sectional feature normalisation (Jansen Ch. 4)

Right now features like `ret_21d`, `vol_21d`, `rsi_14` are global. Stocks in 2022 bear-market have systematically lower `ret_21d` than 2024 bull-market stocks, so the model sees regime, not rank. Rank-transform each feature per date *before* training:

```python
for f in passed_features:
    X_all[f] = X_all.groupby(fd2['date'])[f].transform(
        lambda s: s.rank(pct=True)
    )
```

This is what Alphalens does internally for IC computation — apply it consistently to the model inputs.

### P1 — Add Jansen's high-conviction alphas (Ch. 4)

17 features is sparse and biased toward trend/momentum. Add the canonical mean-reversion + low-vol + quality factors:

- **Short-term reversal**: `-ret_5d` (1-week reversal — Lo/MacKinlay)
- **12-1 momentum**: `c.shift(21)/c.shift(252) - 1` (skip last month — Jegadeesh-Titman)
- **Idiosyncratic vol**: residual of stock return vs market return, rolling 60d std
- **Volume shock**: `(v - v.rolling(20).mean()) / v.rolling(20).std()`
- **Drawdown depth**: `c / c.rolling(252).max() - 1`
- **Amihud illiquidity**: `|ret_1d| / (close*volume)` rolling 21d mean

### P1 — Make the backtest long-short (Jansen Ch. 12)

Current: long top decile only. With 22 names all in US, it's just a high-beta US long fund. Jansen Ch. 12's reference backtest is **long top decile / short bottom decile, dollar-neutral**. That's the right benchmark for a cross-sectional signal — and crucially it isolates whether your signal actually has spread.

```python
threshold_hi = grp['signal'].quantile(1 - TOP_DECILE)
threshold_lo = grp['signal'].quantile(TOP_DECILE)
longs  = grp[grp['signal'] >= threshold_hi]
shorts = grp[grp['signal'] <= threshold_lo]
ret = longs['actual'].mean() - shorts['actual'].mean() - TOTAL_COST_BPS/1e4
```

If shorting isn't allowed (Shariah), use **market-neutral via index hedge**: long top decile, short SPY/EWM/EWH proportional to per-market book weight. Still cleaner than long-only.

### P2 — Kill (or loosen) the circuit breaker

14 of 24 rebalances are CB-trips. The CB at -15% drawdown is so trigger-happy it's making the signal *un*-evaluable. Either:
- Disable it for evaluation runs, or
- Raise to -25% AND require CB to stay tripped for 2+ periods (Jansen-style regime overlay, Ch. 12).

### P2 — Multi-horizon target ensemble

Train two models (5d and 21d targets), average rank-scaled predictions. Jansen Ch. 12 calls this "horizon ensembling" — reduces target-specific noise.

### P2 — Replace Spearman per-date IC with rank-IC during training

Use a custom LGB metric that returns per-date Spearman correlation, so early stopping selects rounds that optimise *what you actually backtest*, not RMSE on raw returns.

## What to expect

The math: equal-weight Sharpe of the same top-decile names is **+1.55**. Your signal-weighted Sharpe is **-0.20**. The signal is currently subtracting ~1.7 units of Sharpe by ranking the wrong stocks high.

With the P0 fixes only (rank target + no Lasso + per-date feature rank), the model should at minimum match equal-weight (Sharpe ≈ 1.5) and exceed it once spread is real (target Sharpe 1.8–2.2 net, based on the existing D1 vs D5 spread which is ~15%).

P1 (long-short + extra alphas) should give another 0.3–0.5 in Sharpe and cut max-DD substantially because the short book hedges the long.

## Files

- `v3_DIAGNOSIS_AND_FIXES.md` — this document
- `v3_patches.py` — drop-in replacement cells (paste into the notebook in order)
