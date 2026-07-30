"""Reconstruct each loan's MONTHLY cash flow and outstanding balance from the
end-of-life totals Lending Club publishes.

Why this file exists
--------------------
Everything upstream measures return at the level of a whole loan: R =
(total_pymnt - funded_amnt) / funded_amnt, realised once, at whatever date the
loan happened to resolve. That is enough to rank loans, but it cannot produce a
Sharpe ratio, because a Sharpe ratio needs a *return time series* and a loan
that pays for 41 months contributes exactly one number.

The standard fix is to stop treating the loan as the unit and treat the BOOK as
the unit: reconstruct what the portfolio earned in each calendar month, and take
the Sharpe of that monthly series. That requires knowing when the money moved,
which LC does not give us directly -- it gives totals plus `last_pymnt_d`.

The reconstruction, and what it assumes
---------------------------------------
Per loan, with n = months from issue to last payment (clipped to the term):

  scheduled cash  = total_pymnt - recoveries      (LC's total_pymnt includes
                    recoveries, which arrive long after the loan stops paying)
  months 1..n-1   = the contractual `installment`
  month n         = whatever scheduled cash is left over

That single residual term is doing real work. For a loan that ran to term it is
just the last installment. For a loan that PREPAID it is a balloon -- the payoff
of the remaining principal -- which is exactly right and is why prepayment shows
up as an early return of capital rather than as a shortfall. If the residual
comes out negative (the borrower paid less than n-1 full installments, i.e. was
partially delinquent for a while before stopping), the schedule falls back to
spreading the cash evenly over the n months, because we have no information
about which months were missed.

Cash is then split into interest and principal by running the actual
amortisation at the loan's own rate:

  interest_k   = min(balance_{k-1} * int_rate/12, cash_k)
  principal_k  = min(cash_k - interest_k, balance_{k-1})
  balance_k    = balance_{k-1} - principal_k

The min() on interest means unpaid accrued interest is never capitalised: a
month where the borrower pays less than the interest due earns only what was
actually received, and the balance does not grow. Any cash beyond full payoff
(late fees) is booked as income, not as negative balance.

Charge-off and recovery timing are the two places where the reconstruction
makes a genuine choice rather than a derivation:

  write-off at  n + CHARGEOFF_LAG_MONTHS   (LC charges off at 120+ days past due,
                                            so ~4 months after the last payment)
  recovery at   n + RECOVERY_LAG_MONTHS    (collections/sale proceeds, ~a year)

Both are exposed as parameters precisely so the Sharpe can be re-run against
them -- see the sensitivity check in ladder.py. Moving a loss later smooths the
monthly series and RAISES the Sharpe, which is the single biggest way this kind
of analysis can flatter itself, so it must be shown rather than buried.

The accounting identity that keeps this honest
----------------------------------------------
Summed over the loan's life:

    interest + fees - write-off + recoveries  ==  total_pymnt - funded_amnt

i.e. the monthly P&L reconstruction reproduces the realised return R exactly,
loan by loan. It redistributes R over time; it does not invent or lose a dollar.
`verify_identity()` asserts this on real data.
"""
import numpy as np

from .config import LOSS_STATUSES

# 60m term + a year of recovery lag; index k = months since issue (k=0 is the
# origination month itself, when the money goes out and nothing comes back)
MAX_OFFSET = 73

CHARGEOFF_LAG_MONTHS = 4
RECOVERY_LAG_MONTHS = 12

# quantities the portfolio engine needs, all per loan-month
FLOW_KEYS = ("interest", "principal", "writeoff", "recovery", "balance")


def _loan_inputs(df):
    n = np.rint(df["holding_years"].to_numpy(dtype=float) * 12.0).astype(int)
    term = df["term_months"].to_numpy(dtype=int)
    return {
        "funded": df["funded_amnt"].to_numpy(dtype=float),
        "installment": df["installment"].to_numpy(dtype=float),
        "rate_m": df["int_rate"].to_numpy(dtype=float) / 100.0 / 12.0,
        "n": np.clip(n, 1, np.minimum(term, MAX_OFFSET - 1)),
        "total_pymnt": df["total_pymnt"].to_numpy(dtype=float),
        "recoveries": df["recoveries"].fillna(0.0).to_numpy(dtype=float),
        "is_loss": df["loan_status"].isin(LOSS_STATUSES).to_numpy(),
    }


def loan_flows(df, chargeoff_lag=CHARGEOFF_LAG_MONTHS, recovery_lag=RECOVERY_LAG_MONTHS):
    """Per-loan monthly schedules, shape (n_loans, MAX_OFFSET), in dollars.

    `balance` is the principal outstanding at the END of each month, so
    balance[:, 0] == funded_amnt (money is out the door, nothing repaid yet)
    and it is what the portfolio marks as its invested assets.
    """
    inp = _loan_inputs(df)
    c = len(df)
    T = MAX_OFFSET
    funded, installment, rate_m = inp["funded"], inp["installment"], inp["rate_m"]
    n, is_loss = inp["n"], inp["is_loss"]

    sched_cash = np.maximum(inp["total_pymnt"] - inp["recoveries"], 0.0)

    k = np.arange(T)
    # payments land in months 1..n
    active = (k[None, :] >= 1) & (k[None, :] <= n[:, None])
    cash = np.where(active, installment[:, None], 0.0)
    rows = np.arange(c)
    residual = sched_cash - installment * (n - 1)
    cash[rows, n] = residual
    # borrower paid less than n-1 full installments: we cannot know which months
    # were missed, so spread what was received evenly across the active window
    ragged = residual < 0
    if ragged.any():
        even = np.where(n > 0, sched_cash / np.maximum(n, 1), 0.0)
        cash[ragged] = np.where(active[ragged], even[ragged, None], 0.0)

    interest = np.zeros((c, T))
    principal = np.zeros((c, T))
    balance = np.zeros((c, T))
    b = funded.copy()
    balance[:, 0] = b
    for j in range(1, T):
        cj = cash[:, j]
        int_j = np.minimum(b * rate_m, cj)
        prin_j = np.minimum(cj - int_j, b)
        # cash beyond full payoff is fee income, not negative principal
        int_j = cj - prin_j
        b = b - prin_j
        interest[:, j] = int_j
        principal[:, j] = prin_j
        balance[:, j] = b

    # Whatever principal is still outstanding once the payments stop is a loss,
    # and it is written off. For a charge-off that happens ~4 months later (LC
    # charges off at 120+ dpd), so the loan sits on the books as a delinquent
    # asset in between -- which is what puts the loss in the month it is
    # actually recognised rather than the month the borrower went quiet. For a
    # loan LC marked Fully Paid that nonetheless did not amortise (a settlement),
    # there is nothing to wait for, so it is recognised immediately.
    writeoff = np.zeros((c, T))
    recovery = np.zeros((c, T))
    lag = np.where(is_loss, chargeoff_lag, 0)
    wo_at = np.minimum(n + lag, T - 1)
    rc_at = np.minimum(n + recovery_lag, T - 1)
    writeoff[rows, wo_at] = balance[rows, n]
    recovery[rows, rc_at] = inp["recoveries"]

    idx = np.arange(T)
    after_wo = idx[None, :] >= wo_at[:, None]
    balance = np.where(after_wo, 0.0, balance)

    return {
        "interest": interest,
        "principal": principal,
        "writeoff": writeoff,
        "recovery": recovery,
        "balance": balance,
    }


def vintage_flows(df, issue_idx, n_vintages, weights=None, chunk=150_000, **lags):
    """Collapse per-loan schedules into PER-VINTAGE, PER-DOLLAR-INVESTED
    schedules of shape (n_vintages, MAX_OFFSET).

    This is the pivot that makes the whole simulation cheap. Once a vintage's
    schedule is expressed per dollar invested, the portfolio engine only has to
    decide a single scalar per month -- how many dollars to deploy into that
    vintage -- instead of tracking 580,000 loans individually. `weights` (the
    dollars each loan represents in the strategy, normally funded_amnt, or 0 for
    a rejected loan) is what encodes the selection rule.

    Loans are aggregated in chunks because the intermediate (n_loans, 73)
    matrices are ~170MB apiece and there are five of them.
    """
    w = (df["funded_amnt"].to_numpy(dtype=float) if weights is None
         else np.asarray(weights, dtype=float))
    out = {key: np.zeros((n_vintages, MAX_OFFSET)) for key in FLOW_KEYS}
    invested = np.zeros(n_vintages)

    for start in range(0, len(df), chunk):
        sl = slice(start, start + chunk)
        sub = df.iloc[sl]
        w_sub = w[sl]
        keep = w_sub > 0
        if not keep.any():
            continue
        sub = sub[keep]
        w_sub = w_sub[keep]
        idx_sub = issue_idx[sl][keep]
        flows = loan_flows(sub, **lags)
        # each loan's schedule is scaled to the dollars the strategy puts in it
        scale = w_sub / sub["funded_amnt"].to_numpy(dtype=float)
        for key in FLOW_KEYS:
            np.add.at(out[key], idx_sub, flows[key] * scale[:, None])
        np.add.at(invested, idx_sub, w_sub)

    per_dollar = {key: out[key] / np.maximum(invested, 1e-12)[:, None] for key in FLOW_KEYS}
    return per_dollar, invested


def early_performance(df, age_months, **lags):
    """Net P&L per dollar lent, realised in a loan's FIRST `age_months` months.

    This is the one quantity in the file that is deliberately incomplete, and the
    incompleteness is the point. Everything else here reconstructs a loan's whole
    life, which a lender only knows three to five years after the fact. A rule
    that needs to know the level of returns in the era it is lending into cannot
    wait that long -- by the time the 2015 vintages have matured it is 2018 and
    the decision is long gone.

    What a lender does have, a year in, is a year of payment behaviour: coupons
    banked, loans already charged off, recoveries received. Truncating the
    reconstruction at `age_months` reproduces exactly that and nothing more. Read
    at age k it uses no cash flow dated later than k, so a rule built on it is
    using only what was on the servicer's books at the moment it decided.

    The number is not a return -- a healthy loan at month 12 has paid interest
    while most of its principal is still outstanding and still at risk, so the
    figure runs well below the loan's eventual return and is not comparable
    across terms. It is a *level indicator*: within a term, a vintage whose first
    year came in light is a vintage that will finish light.
    """
    flows = loan_flows(df, **lags)
    k = int(np.clip(age_months, 1, MAX_OFFSET - 1))
    pnl = (flows["interest"][:, :k + 1].sum(1)
           - flows["writeoff"][:, :k + 1].sum(1)
           + flows["recovery"][:, :k + 1].sum(1))
    return pnl / np.maximum(df["funded_amnt"].to_numpy(dtype=float), 1.0)


def verify_identity(df, tol=1e-6, **lags):
    """Assert that the monthly reconstruction reproduces each loan's realised
    return. This is the check that separates a cash-flow model from a story: if
    the monthly series did not sum back to R, every Sharpe built on it would be
    measuring an artefact of the reconstruction."""
    flows = loan_flows(df, **lags)
    pnl = (flows["interest"].sum(1) - flows["writeoff"].sum(1) + flows["recovery"].sum(1))
    realised = df["total_pymnt"].to_numpy(dtype=float) - df["funded_amnt"].to_numpy(dtype=float)
    err = np.abs(pnl - realised) / np.maximum(df["funded_amnt"].to_numpy(dtype=float), 1.0)
    return {
        "max_rel_error": float(err.max()),
        "mean_rel_error": float(err.mean()),
        "n_violations": int((err > tol).sum()),
        "n_loans": len(df),
    }
