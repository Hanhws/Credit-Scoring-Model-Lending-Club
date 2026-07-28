"""Cohort-based Sortino / Sharpe ratio for a loan portfolio.

Unit of analysis: origination cohort (issue month). Dollars are aggregated
first within each cohort (sum funded_amnt, sum total_pymnt) and *then*
converted to an annualized rate -- weight-averaging already-annualized
per-loan rates would let a single early default dominate a cohort out of
proportion to the dollars actually at risk. Sharpe/Sortino are then computed
across the resulting time series of cohort excess returns, which captures how
much the portfolio's risk-adjusted return swings across economic cycles --
the thing that determines whether LC can run this book sustainably.
"""
import numpy as np
import pandas as pd


def cohort_returns(df, cohort_col="issue_month", weight_col="funded_amnt"):
    def agg(g):
        invested = g[weight_col].sum()
        received = g["total_pymnt"].sum()
        simple_return = (received - invested) / invested
        holding_years = np.average(g["holding_years"], weights=g[weight_col])
        ann_return = (1 + simple_return) ** (1 / holding_years) - 1
        rf_annual = np.average(g["Rf"], weights=g[weight_col])
        return pd.Series({
            "invested": invested,
            "ann_return": ann_return,
            "rf_annual": rf_annual,
            "ER": ann_return - rf_annual,
        })

    return df.groupby(cohort_col).apply(agg, include_groups=False)


def sharpe_ratio(cohort_er):
    cohort_er = cohort_er.dropna()
    return cohort_er.mean() / cohort_er.std(ddof=1)


def sortino_ratio(cohort_er, mar=0.0):
    """mar is expressed in excess-return space (0 == the risk-free rate itself)."""
    cohort_er = cohort_er.dropna()
    downside = np.minimum(cohort_er - mar, 0.0)
    downside_dev = np.sqrt((downside ** 2).mean())
    if downside_dev == 0:
        return np.nan
    return (cohort_er.mean() - mar) / downside_dev


def threshold_selection(df, pred_col, threshold=0.0):
    """Invest wherever the model predicts a positive excess return over the
    risk-free benchmark -- a parameter-free decision rule (no K to search
    for), and one that lets the acceptance rate move on its own with the
    model's read of the cycle, rather than forcing a fixed quota every month."""
    return df[df[pred_col] > threshold]


def variable_k_selection(df, pred_col, k_col, group_cols):
    """Like top_k_selection, but each row carries its own K (df[k_col]) instead
    of one global constant -- used for the GDP-slope-driven K, where K varies
    by (state, year) instead of being a single grid-searched number."""
    rank = df.groupby(group_cols)[pred_col].rank(pct=True, ascending=False)
    return df[rank <= df[k_col]]


def top_k_selection(df, pred_col, k_frac, cohort_col="issue_month"):
    """Within each origination cohort, keep the top k_frac of loans ranked by
    predicted excess return. Selecting per-cohort (rather than globally) keeps
    the portfolio invested every month instead of the model just avoiding bad
    macro periods in hindsight."""
    rank = df.groupby(cohort_col)[pred_col].rank(pct=True, ascending=False)
    return df[rank <= k_frac]


def portfolio_summary(df, cohort_col="issue_month", weight_col="funded_amnt"):
    cohorts = cohort_returns(df, cohort_col, weight_col)
    er = cohorts["ER"]
    return {
        "n_loans": len(df),
        "n_cohorts": cohorts.shape[0],
        "funded_amnt": df[weight_col].sum(),
        "mean_excess_return": er.mean(),
        "sharpe": sharpe_ratio(er),
        "sortino": sortino_ratio(er),
    }
