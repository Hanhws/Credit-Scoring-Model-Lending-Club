"""Lend less into a bad year -- read from the applicant pool, with no lag.

The channel this targets
------------------------
The shuffled-split model reaches a Sharpe of 1.32 while the honest one
sits near 1.2, and the difference is almost entirely one column: its acceptance
rate moves with a standard deviation of 0.104 across vintages, where every
honest rule moves by 0.01-0.06. It knows which years to sit out because it was
fitted on them. That gap -- roughly +0.6 Sharpe -- is the headroom available to
timing, and nothing in the cross-section can reach it: tightening the grade cut
leaves the share of losing vintages at 13.1% no matter how far it goes, because
one credit cycle moves every grade together.

Why the pool, and not the loans
-------------------------------
Two earlier attempts to read the level failed for the same reason. Recalibrating
on matured vintages sees only the era the model already knows (three-year lag).
Recalibrating on twelve-month seasoning recovers about a third of the bias (one-
year lag). The applicant pool has NO lag: the average FICO, DTI, utilisation and
inquiry count of the loans being written this month are on the application forms
being signed this month.

What the signal can and cannot do
---------------------------------
Fitted expanding-window and tested out of sample, pool composition cuts the RMSE
of the vintage forecast by 23% against a no-feature benchmark. It is real, and
it is weak: the forecast never goes negative -- it runs 5.7% down to 3.0% while
the vintages it is describing run +3.8% down to -5.7% -- so its sign carries no
information and it cannot be used as a stop-lending gate. Sign accuracy equals
the base rate exactly.

It CAN be used as a dial. Scaling the acceptance rate by the forecast relative
to its own history deploys less capital into the years the pool looks worse,
which is the one thing a constant acceptance rate provably cannot do: excess
return is accept_rate * (loan_irr - rf), so a constant rate cancels out of the
Sharpe entirely and only a time-VARYING one changes the shape of the series.

Note the honest correlation caveat: out-of-sample correlation with realised
returns is 0.87, and a forecaster using no features at all scores 0.95 on the
same test. Both series trend, so correlation measures nothing here. RMSE is the
only metric in this file that means what it appears to mean.
"""
import numpy as np
import pandas as pd

POOL_FEATURES = ["fico_avg", "dti", "int_rate", "revol_util", "inq_last_6mths"]
MIN_FIT_VINTAGES = 18
RIDGE_LAMBDA = 1.0
# the dial's travel. A floor well above zero is deliberate: the forecast is not
# accurate enough to justify closing the book, and a lender that stops
# originating loses the operation, not just the vintage.
MIN_ACCEPT, MAX_ACCEPT = 0.35, 1.00


def pool_table(df, features=POOL_FEATURES):
    """Dollar-weighted applicant-pool composition, one row per origination
    month. Every column is observable the day the loan is written."""
    feats = [f for f in features if f in df.columns]
    w = df["funded_amnt"].to_numpy(dtype=float)
    acc = {}
    for f in feats:
        x = df[f].to_numpy(dtype=float)
        ok = np.isfinite(x)
        acc[f] = pd.Series(np.where(ok, x, 0.0) * w * ok, index=df.index)
    frame = pd.DataFrame(acc)
    frame["__w"] = w
    g = frame.groupby(df["issue_month"]).sum()
    out = g[feats].div(g["__w"], axis=0)
    return out.sort_index()


def _ridge(X, y, lam=RIDGE_LAMBDA):
    mu, sd = X.mean(0), X.std(0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    P = np.eye(Z.shape[1]); P[0, 0] = 0.0
    beta = np.linalg.solve(Z.T @ Z + lam * P, Z.T @ y)
    return lambda Xn: np.column_stack([np.ones(len(Xn)), (Xn - mu) / sd]) @ beta


def forecast(pool, realised, term_months, min_fit=MIN_FIT_VINTAGES):
    """Expanding-window forecast of each vintage's excess return.

    At month d the fit uses only vintages that had fully matured by d, so their
    realised return was on the books; the features come from the pool being
    written at d. Months with too little history return NaN and the caller
    falls back to lending normally.
    """
    idx = pd.DatetimeIndex(pool.index)
    X = pool.to_numpy(dtype=float)
    y = realised.reindex(idx).to_numpy(dtype=float)
    out = np.full(len(idx), np.nan)
    for i, d in enumerate(idx):
        known = (idx <= d - pd.DateOffset(months=term_months)) & np.isfinite(y)
        if known.sum() < min_fit:
            continue
        out[i] = float(_ridge(X[known], y[known])(X[i:i + 1])[0])
    return pd.Series(out, index=idx, name="forecast")


def forecast_from(dial_fit, pool_new, term_months, min_fit=MIN_FIT_VINTAGES):
    """Forecast a fresh set of vintages using history frozen from the training file.

    `dial_fit` holds the pool composition and the realised excess return of
    training-file vintages. For each new decision month the mapping is refitted on
    the subset of that history whose loans had already matured by then, so the
    forecast for 2016-03 is built only from vintages whose three years were up
    before 2016-03. The new file contributes its application forms and nothing
    else -- none of its outcomes are read.
    """
    hist_pool, realised = dial_fit["pool"], dial_fit["realised"]
    feats = [c for c in hist_pool.columns if c in pool_new.columns]
    h_idx = pd.DatetimeIndex(hist_pool.index)
    Xh = hist_pool[feats].to_numpy(dtype=float)
    yh = realised.reindex(h_idx).to_numpy(dtype=float)

    n_idx = pd.DatetimeIndex(pool_new.index)
    Xn = pool_new[feats].to_numpy(dtype=float)
    out = np.full(len(n_idx), np.nan)
    for i, d in enumerate(n_idx):
        known = (h_idx <= d - pd.DateOffset(months=term_months)) & np.isfinite(yh)
        if known.sum() < min_fit:
            continue
        out[i] = float(_ridge(Xh[known], yh[known])(Xn[i:i + 1])[0])
    return pd.Series(out, index=n_idx, name="forecast")


def dial_with_history(dial_fit, pool_new, term_months, min_fit=MIN_FIT_VINTAGES,
                      **dial_kwargs):
    """The dial, referenced against the whole forecast history rather than against
    the evaluation window's own opening months.

    This corrects a specification error in the first version, and the error is
    worth recording because it produced a null result that looked like evidence.

    The dial asks "does this year look worse than the years I have already seen".
    The first implementation built "the years I have already seen" from forecasts
    made INSIDE the evaluation window, so the first decision month had nothing to
    compare against and the sixth had five months. Worse, the forecast level was
    rising across 2016-17, so the ratio sat above one and clipped at the ceiling:
    the dial stayed at 100% for 22 of 26 months and the timing rule was, in
    practice, never applied. Its measured contribution of +0.05 (p = 0.26) was
    therefore not a test of timing -- it was a test of doing almost nothing.

    The reference belongs in the training period, where a century of vintages
    establishes what a normal forecast looks like. Forecasts for the historical
    vintages are built the same expanding-window way, using only outcomes already
    matured at each historical decision month, and the two series are laid end to
    end. Every evaluation month then compares against roughly a hundred prior
    forecasts instead of a handful.

    No outcome from the new file is read at any point; the correction changes what
    the forecast is compared with, not what goes into it.
    """
    fc_hist = forecast(dial_fit["pool"], dial_fit["realised"], term_months,
                       min_fit=min_fit)
    fc_new = forecast_from(dial_fit, pool_new, term_months, min_fit=min_fit)
    combined = pd.concat([fc_hist, fc_new]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    dial = acceptance_dial(combined, **dial_kwargs)
    idx = pd.DatetimeIndex(pool_new.index)
    return fc_new.reindex(idx), dial.reindex(idx)


def acceptance_dial(fc, min_accept=MIN_ACCEPT, max_accept=MAX_ACCEPT):
    """Turn the forecast into an acceptance rate per vintage.

    The reference point is the expanding mean of PAST forecasts, so the dial
    asks "does this year look worse than the years I have already seen" rather
    than comparing against a level fixed with hindsight. A vintage forecast at
    half the running average gets half the capital, floored so the book stays
    open.
    """
    f = fc.to_numpy(dtype=float)
    ref = np.full(len(f), np.nan)
    for i in range(len(f)):
        past = f[:i][np.isfinite(f[:i])]
        if len(past):
            ref[i] = past.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(np.isfinite(ref) & (ref > 0), f / ref, 1.0)
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    return pd.Series(np.clip(ratio, min_accept, max_accept), index=fc.index,
                     name="accept")


def weights(df, score_col, dial, base_mask=None):
    """Fund the best `dial[v]` share of vintage v, ranked by the model, within
    whatever `base_mask` already allows."""
    funded = df["funded_amnt"].to_numpy(dtype=float)
    keep = np.ones(len(df), dtype=bool) if base_mask is None else np.asarray(base_mask)
    rank = df[score_col].where(keep).groupby(df["issue_month"]).rank(pct=True, ascending=False)
    quota = df["issue_month"].map(dial).to_numpy(dtype=float)
    quota = np.where(np.isfinite(quota), quota, 1.0)
    return np.where(keep & (rank.to_numpy(dtype=float) <= quota), funded, 0.0)
