"""Selection rules expressed as dollar weights, plus the control they are judged against.

A rule's output is one number per loan: how many dollars the strategy puts into
it, zero for a decline. Keeping the rules in this shape -- rather than as boolean
masks -- is what lets `metrics.vintage_table` treat a declined application as an
allocation to Treasuries instead of as capital that vanished, which is the whole
basis on which acceptance-rate decisions become measurable.
"""
import numpy as np


def w_all(df):
    """Fund every application. The baseline the whole report is measured against."""
    return df["funded_amnt"].to_numpy(dtype=float)


def w_mask(df, mask):
    """Fund where `mask` is true, at the full requested amount."""
    return np.where(np.asarray(mask), df["funded_amnt"].to_numpy(dtype=float), 0.0)


def w_random_at_rate(df, target_rate, seed):
    """The control that matters most: fund a RANDOM subset of each vintage, hitting
    the same dollar acceptance rate as the strategy under test.

    Any rule that funds fewer loans changes its risk profile for free, and without
    this control that free change would be credited to the model. Ranking is done
    within the origination month so the random book lends every month, exactly as
    the real rules do.
    """
    rng = np.random.default_rng(seed)
    u = df["funded_amnt"].to_frame().assign(u=rng.random(len(df)))["u"]
    rank = u.groupby(df["issue_month"]).rank(pct=True)
    return w_mask(df, rank <= target_rate)


def dollar_accept_rate(df, w):
    """Share of the offered dollars the rule actually deployed."""
    return float(np.sum(w) / df["funded_amnt"].sum())
