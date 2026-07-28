"""Realized loan return and maturity-matched risk-free benchmark.

R      : cumulative total return realized over the loan's life,
         (total_pymnt - funded_amnt) / funded_amnt.  Not annualized -- annualizing
         per loan blows up for early defaults (a default in month 1 of a 5y loan
         would show as -1200%/yr) and destabilizes both training and the
         portfolio-level risk stats. Cumulative return is well-behaved
         (bounded below by -1) and is what actually gets aggregated in dollars
         at the portfolio/cohort level later.
Rf     : annualized, maturity-matched risk-free rate (DGS3 for 36m, DGS5 for 60m)
         at the loan's issue month.
Rf_cum : Rf compounded over the loan's *actual* holding period (holding_years),
         not the nominal term. Loans routinely resolve before term -- prepayment
         or default -- so benchmarking against the full 3-5y term overstates
         the opportunity cost for any loan that didn't run the full distance
         and makes early payoffs look artificially bad. Comparable duration is
         what makes R and Rf_cum an apples-to-apples excess return.
ER     : R - Rf_cum, the cumulative excess return. This is the regression target.
"""
import pandas as pd

from . import config


def _load_rf_series(path, col_name):
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df = df.rename(columns={df.columns[1]: "rate_pct"})
    df["month"] = df["observation_date"].values.astype("datetime64[M]")
    monthly = df.groupby("month")["rate_pct"].mean().rename(col_name)
    return monthly


def load_risk_free_table():
    """Monthly-averaged risk-free rate (decimal, e.g. 0.03) per term."""
    dgs3 = _load_rf_series(config.DGS3_CSV, "DGS3")
    dgs5 = _load_rf_series(config.DGS5_CSV, "DGS5")
    rf = pd.concat([dgs3, dgs5], axis=1) / 100.0
    return rf


def attach_returns(df, rf_table):
    out = df.copy()
    out["term_months"] = out["term"].str.extract(r"(\d+)").astype(int)
    out["issue_d_dt"] = pd.to_datetime(out["issue_d"], format="%b-%Y")
    out["issue_month"] = out["issue_d_dt"].values.astype("datetime64[M]")

    last_pymnt_dt = pd.to_datetime(out["last_pymnt_d"], format="%b-%Y")
    holding_months = (
        (last_pymnt_dt.dt.year - out["issue_d_dt"].dt.year) * 12
        + (last_pymnt_dt.dt.month - out["issue_d_dt"].dt.month)
    )
    out["holding_years"] = holding_months.fillna(1).clip(lower=1, upper=out["term_months"]) / 12.0

    out["R"] = (out["total_pymnt"] - out["funded_amnt"]) / out["funded_amnt"]

    rf_col = out["term_months"].map({36: "DGS3", 60: "DGS5"})
    rf_lookup = rf_table.reset_index().melt(
        id_vars="month", value_vars=["DGS3", "DGS5"], var_name="rf_col", value_name="rf_rate"
    )
    key = pd.DataFrame({"month": out["issue_month"], "rf_col": rf_col})
    out["Rf"] = key.merge(rf_lookup, on=["month", "rf_col"], how="left")["rf_rate"].values
    out["Rf_cum"] = out["Rf"] * out["holding_years"]

    out["ER"] = out["R"] - out["Rf_cum"]

    total_months = out["issue_d_dt"].dt.year * 12 + out["issue_d_dt"].dt.month + out["term_months"]
    matures_year = (total_months - 1) // 12
    matures_month = (total_months - 1) % 12 + 1
    out["matures_at"] = pd.to_datetime(
        {"year": matures_year, "month": matures_month, "day": 1}
    )
    return out


def fully_matured_vintage(df, snapshot_date):
    """Keep only loans whose full nominal term has already elapsed by
    snapshot_date, so the matured-loan sample for that vintage is complete
    rather than skewed toward whichever loans resolved fastest."""
    return df[df["matures_at"] <= snapshot_date]
