import pandas as pd

from . import config
from .returns import attach_returns, fully_matured_vintage
from .features import build_features


def _read_csv(path):
    cols = sorted(set(config.OUTCOME_COLS) | set(config.FEATURE_COLS))
    return pd.read_csv(path, usecols=cols, low_memory=False)


def load_dataset(path, rf_table, matured_only=True):
    df = _read_csv(path)
    if matured_only:
        df = df[df["loan_status"].isin(config.MATURED_STATUSES)].reset_index(drop=True)
    df = attach_returns(df, rf_table)
    if matured_only:
        df = fully_matured_vintage(df, config.SNAPSHOT_DATE).reset_index(drop=True)
    df = build_features(df)
    return df
