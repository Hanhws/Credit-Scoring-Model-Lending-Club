"""Ex-ante feature engineering. No column here may depend on information that
only becomes known after the funding decision."""
import numpy as np
import pandas as pd

from . import config

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}

MODEL_CATEGORICALS = [
    "grade", "sub_grade", "home_ownership", "verification_status", "purpose",
    "addr_state", "initial_list_status", "application_type",
    "verification_status_joint",
]


def build_features(df):
    out = df.copy()

    earliest = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y")
    out["credit_history_months"] = (
        (out["issue_d_dt"].dt.year - earliest.dt.year) * 12
        + (out["issue_d_dt"].dt.month - earliest.dt.month)
    )

    out["fico_avg"] = (out["fico_range_low"] + out["fico_range_high"]) / 2.0

    out["emp_length_num"] = out["emp_length"].map(EMP_LENGTH_MAP)

    for col in ("int_rate", "revol_util"):
        out[col] = out[col].astype(str).str.rstrip("%").astype(float)

    for col in MODEL_CATEGORICALS:
        out[col] = out[col].astype("category")

    return out


_DROP_RAW = {"term", "emp_length", "fico_range_low", "fico_range_high", "earliest_cr_line"}


def feature_matrix(df, safe_only=False, extra_cols=None):
    base = config.SAFE_FEATURE_COLS if safe_only else config.FEATURE_COLS
    cols = [c for c in base if c not in _DROP_RAW]
    cols += ["term_months", "emp_length_num", "credit_history_months", "fico_avg"]
    cols += list(extra_cols or [])
    cols = [c for c in dict.fromkeys(cols) if c in df.columns]
    X = df[cols].copy()
    X["term_months"] = X["term_months"].astype("category")
    return X
