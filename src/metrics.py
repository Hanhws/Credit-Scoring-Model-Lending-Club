"""Vintage-level risk-adjusted return: the headline measurement.

Choosing the unit of observation
--------------------------------
A Sharpe ratio needs a return series, and for a loan book there are three
candidates. Only one of them measures something a lender would recognise as
risk.

  per loan      mean(ER)/sd(ER) across individual loans. This is dominated by
                idiosyncratic borrower risk, which a book of 580,000 loans
                diversifies away almost entirely. It is an information ratio on
                the ranking model, not a portfolio Sharpe, and it lands near 0.1
                no matter how good the strategy is.

  per month     the book's monthly NAV return (portfolio.py). This is the
                textbook answer for a traded asset and the wrong answer here:
                held at amortised cost, an amortising loan book has almost no
                monthly innovation -- the coupon arrives on schedule and losses
                are recognised on an accounting trigger. It produces annualised
                Sharpes near 8, which is an artefact of the accounting, not a
                statement about risk. Kept as a cross-check, desmoothed, never
                as the headline.

  per VINTAGE   what this file computes. One observation per origination month:
                if I run this credit box, how much does my realised return
                depend on WHEN I lent? That is the risk a lender actually
                carries, it is the risk that ruins lenders, and in this dataset
                it is large -- mean realised return per dollar falls from 13.2%
                on 2013 vintages to -1.3% on 2018 vintages.

Two things this fixes versus the previous cohort measure
--------------------------------------------------------
1. IRR instead of (1+R)^(1/holding_years)-1. The old annualisation raises a
   cumulative return to a power that blows up as the holding period shrinks, so
   a vintage that resolved quickly got an enormous annualised number in either
   direction and dominated the standard deviation for a reason that has nothing
   to do with credit. The IRR of the vintage's actual monthly cash-flow stream
   is the same quantity done properly, and it is what private-credit books
   report.

2. The cash leg. Capital is measured per dollar OFFERED to the credit box, not
   per dollar lent, so rejecting an application is an allocation to Treasuries
   rather than a disappearance of capital:

       vintage return  = a * y  +  (1 - a) * rf          a = dollar accept rate
       vintage excess  = a * (y - rf)                    y = vintage loan IRR

   Note what this does and does not buy. Because excess return scales linearly
   in `a`, a CONSTANT acceptance rate cancels out of the Sharpe entirely --
   mixing in cash moves mean and standard deviation by the same factor, which is
   just the statement that Sharpe is the slope of the capital allocation line
   and is invariant to leverage. Blending cash is therefore not a way to
   manufacture a better Sharpe, and any analysis claiming otherwise is measuring
   something else.

   What it does buy is that a TIME-VARYING acceptance rate becomes measurable:
   a rule that lends less into a deteriorating vintage and parks the difference
   in Treasuries genuinely lowers the volatility of the excess-return series.
   That is the only channel through which selectivity-as-timing can pay, and
   this measurement is what exposes it.
"""
import numpy as np
import pandas as pd

from . import cashflow
from .config import SERVICING_FEE


def _irr(cashflows, lo=-0.99, hi=1.0, tol=1e-10, iters=200):
    """Monthly IRR by bisection on the NPV, which is monotone in the discount
    rate for a conventional (one outflow, then inflows) stream. Bisection rather
    than Newton because a vintage that lost most of its principal has its root
    pressed against the -100% boundary where Newton wanders off."""
    cf = np.asarray(cashflows, dtype=float)
    k = np.arange(len(cf))

    def npv(r):
        return float(np.sum(cf / (1.0 + r) ** k))

    f_lo, f_hi = npv(lo), npv(hi)
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_lo * f_hi > 0:
        return np.nan
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = npv(mid)
        if abs(f_mid) < tol:
            break
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def vintage_table(df, weights=None, rf_table=None, servicing_fee=SERVICING_FEE, **lag_kwargs):
    """One row per origination month: accept rate, loan IRR, blended return,
    excess return. `weights` is dollars funded per loan under the strategy
    (0 = rejected); None means fund everything.
    """
    months = np.sort(df["issue_month"].unique())
    month_to_idx = {m: i for i, m in enumerate(months)}
    issue_idx = df["issue_month"].map(month_to_idx).to_numpy(dtype=int)
    n_vint = len(months)

    funded = df["funded_amnt"].to_numpy(dtype=float)
    w = funded if weights is None else np.asarray(weights, dtype=float)

    pd_flows, approved = cashflow.vintage_flows(df, issue_idx, n_vint, weights=w, **lag_kwargs)
    pool = np.zeros(n_vint)
    np.add.at(pool, issue_idx, funded)
    accept = np.divide(approved, pool, out=np.zeros(n_vint), where=pool > 0)

    # net cash to the investor, per dollar invested: -1 at origination, then
    # every dollar the borrower hands over less the servicer's cut
    gross = pd_flows["interest"] + pd_flows["principal"] + pd_flows["recovery"]
    net = gross * (1.0 - servicing_fee)
    net[:, 0] -= 1.0

    irr_ann = np.full(n_vint, np.nan)
    life = np.full(n_vint, np.nan)
    for i in range(n_vint):
        if approved[i] <= 0:
            continue
        r_m = _irr(net[i])
        if np.isfinite(r_m):
            irr_ann[i] = (1.0 + r_m) ** 12 - 1.0
        life[i] = pd_flows["balance"][i].sum()

    rf_annual = np.full(n_vint, np.nan)
    if rf_table is not None:
        rf_series = rf_table["DGS3"].reindex(pd.DatetimeIndex(months)).ffill().bfill()
        rf_annual = rf_series.to_numpy(dtype=float)

    out = pd.DataFrame({
        "issue_month": pd.DatetimeIndex(months),
        "pool": pool,
        "approved": approved,
        "accept_rate": accept,
        "loan_irr": irr_ann,
        "rf": rf_annual,
        "exposure_months": life,
    }).set_index("issue_month")

    # an unfunded vintage sits entirely in Treasuries: zero excess, not missing
    out["loan_excess"] = np.where(out["accept_rate"] > 0, out["loan_irr"] - out["rf"], 0.0)
    out["excess"] = out["accept_rate"] * out["loan_excess"]
    out["blended_return"] = out["rf"] + out["excess"]
    return out


def trim_thin_vintages(vt, min_pool_share=0.001):
    """Drop origination months too small to estimate anything from.

    LC originated $1.8M in all of 2007 and $2.4BN in 2015 alone. Equal-weighting
    those as two observations of the same process lets a handful of loans from a
    near-empty month set the standard deviation of the whole series, and every
    strategy then gets scored mostly on noise from months where it deployed
    almost no capital. The cut is on share of total dollars, so it scales with
    the book rather than with a hard-coded loan count.
    """
    share = vt["pool"] / vt["pool"].sum()
    return vt[share >= min_pool_share]
