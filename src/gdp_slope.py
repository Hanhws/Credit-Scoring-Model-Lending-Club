"""State-level GDP-momentum-driven K: instead of one global top-K% (grid search,
see train.py/final_model.py) or a flat ER_hat>0 rule (final_model.py), let K vary
by (state, year) based on the *trend* (slope) of that state's real GDP growth
rate -- accelerating states get a looser (higher) K, decelerating states get a
tighter (lower) K.

Data: data/raw/SAGDP11_state_real_gdp_growth.csv (BEA SAGDP11, LineCode=1, real
GDP %% change, annual, by state -- see the guide doc for how this was sourced).

Two things this module is careful about, both lessons from earlier in this
project:
1. Reporting lag -- annual state GDP for year Y isn't published until ~June of
   year Y+1, so a loan issued in year Y can only "see" data through Y-1 (if
   issued June onward) or Y-2 (if issued before June). Ignoring this would be
   the same kind of look-ahead bias as the era-missingness bug (guide 8절).
2. The slope is a trend (rate of change of the growth rate), not the growth
   rate's level -- fit with a trailing 3-year OLS line so a single noisy year
   doesn't flip the signal.
"""
import numpy as np
import pandas as pd

from . import config

GDP_CSV = config.RAW_DIR / "SAGDP11_state_real_gdp_growth.csv"

K_MIN, K_MAX = 0.30, 1.00
SLOPE_WINDOW = 3
LAG_CUTOFF_MONTH = 6  # BEA typically publishes year Y's annual estimate ~June of Y+1


def _long_growth_table():
    df = pd.read_csv(GDP_CSV)
    year_cols = [c for c in df.columns if c.isdigit()]
    long = df.melt(id_vars=["addr_state"], value_vars=year_cols, var_name="year", value_name="growth")
    long["year"] = long["year"].astype(int)
    return long.sort_values(["addr_state", "year"]).reset_index(drop=True)


def _trailing_slope(group, window=SLOPE_WINDOW):
    group = group.sort_values("year").reset_index(drop=True)
    slopes = pd.Series(index=group.index, dtype=float)
    for i in range(window - 1, len(group)):
        y = group["growth"].iloc[i - window + 1:i + 1].values
        x = np.arange(window)
        slopes.iloc[i] = np.polyfit(x, y, 1)[0]
    group["slope"] = slopes
    return group


def load_state_gdp_slope():
    """Returns (addr_state, year, growth, slope) -- slope is the trailing
    3-year OLS trend of the growth rate, in percentage-points/year."""
    long = _long_growth_table()
    return long.groupby("addr_state", group_keys=False).apply(_trailing_slope).reset_index(drop=True)


def attach_gdp_k(df, slope_table=None):
    """Adds available_gdp_year, gdp_slope, gdp_percentile, gdp_k to df.
    Requires df to have addr_state and issue_d_dt (from returns.attach_returns)."""
    if slope_table is None:
        slope_table = load_state_gdp_slope()

    out = df.copy()
    original_state_dtype = out["addr_state"].dtype  # merge silently demotes category -> object below
    issue_year = out["issue_d_dt"].dt.year
    issue_month = out["issue_d_dt"].dt.month
    out["available_gdp_year"] = np.where(issue_month >= LAG_CUTOFF_MONTH, issue_year - 1, issue_year - 2)

    lookup = slope_table.rename(columns={"year": "available_gdp_year"})[
        ["addr_state", "available_gdp_year", "growth", "slope"]
    ]
    out = out.merge(lookup, on=["addr_state", "available_gdp_year"], how="left")
    out = out.rename(columns={"growth": "gdp_growth", "slope": "gdp_slope"})
    if str(original_state_dtype) == "category":
        out["addr_state"] = out["addr_state"].astype(original_state_dtype)

    # percentile rank computed over the full historical (state, year) slope
    # panel, not over the loan sample -- this is an exogenous macro transform,
    # not something tuned on loan outcomes
    all_slopes = slope_table["slope"].dropna()
    out["gdp_percentile"] = out["gdp_slope"].apply(
        lambda v: np.nan if pd.isna(v) else (all_slopes < v).mean()
    )
    out["gdp_k"] = K_MIN + (K_MAX - K_MIN) * out["gdp_percentile"]
    out["gdp_k"] = out["gdp_k"].fillna(out["gdp_k"].median())
    return out
