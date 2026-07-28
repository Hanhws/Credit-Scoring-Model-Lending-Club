"""K-fold cross-validated tuning of the top-K% selection threshold.

train.py picks K from a single 85/15 train/validation split, which turned out
to be unstable (K*=0.90 on that split did not hold up on the test set). This
does proper K-fold CV instead: for each fold, fit a model on the other folds
and score the held-out fold, so every K in the grid gets 5 independent
out-of-fold Sortino estimates instead of one. Averaging those gives both a
more robust K* and a sense of how noisy the choice is (std across folds).
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from . import config
from .data import load_dataset
from .features import feature_matrix
from .metrics import portfolio_summary, top_k_selection
from .returns import load_risk_free_table

N_FOLDS = 5
K_GRID = np.arange(0.05, 1.01, 0.05)
MIN_DOLLAR_ACCEPT_RATE = 0.05

PARAMS = {
    "objective": "regression_l1",
    "metric": "l1",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": config.RANDOM_STATE,
}


def train_fold(df_fit, cat_cols):
    idx_tr, idx_es = train_test_split(df_fit.index, test_size=0.1, random_state=config.RANDOM_STATE)
    X_tr, X_es = feature_matrix(df_fit.loc[idx_tr]), feature_matrix(df_fit.loc[idx_es])
    y_tr, y_es = df_fit.loc[idx_tr, "ER"], df_fit.loc[idx_es, "ER"]
    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
    es_set = lgb.Dataset(X_es, label=y_es, categorical_feature=cat_cols, reference=train_set, free_raw_data=False)
    booster = lgb.train(
        PARAMS, train_set, num_boost_round=3000, valid_sets=[es_set],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    return booster


def main():
    print("loading fully-matured training data...")
    rf_table = load_risk_free_table()
    df = load_dataset(config.TRAIN_CSV, rf_table, matured_only=True).reset_index(drop=True)
    print(f"  {len(df):,} loans")

    X_full = feature_matrix(df)
    cat_cols = [c for c in X_full.columns if str(X_full[c].dtype) == "category"]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_curves = []
    for fold_i, (tr_idx, hold_idx) in enumerate(kf.split(df)):
        print(f"\n--- fold {fold_i + 1}/{N_FOLDS} ---")
        df_fit = df.iloc[tr_idx]
        df_hold = df.iloc[hold_idx].copy()

        booster = train_fold(df_fit, cat_cols)
        X_hold = feature_matrix(df_hold)
        df_hold["ER_hat"] = booster.predict(X_hold, num_iteration=booster.best_iteration)

        rows = []
        for k in K_GRID:
            sel = top_k_selection(df_hold, "ER_hat", k)
            s = portfolio_summary(sel)
            s["k_frac"] = round(k, 2)
            s["dollar_accept_rate"] = sel["funded_amnt"].sum() / df_hold["funded_amnt"].sum()
            rows.append(s)
        curve = pd.DataFrame(rows).set_index("k_frac")
        fold_curves.append(curve)
        print(curve[["dollar_accept_rate", "sortino"]].to_string(float_format=lambda x: f"{x:0.4f}"))

    sortino_by_fold = pd.concat(
        [c["sortino"].rename(f"fold{i}") for i, c in enumerate(fold_curves)], axis=1
    )
    dollar_by_fold = pd.concat(
        [c["dollar_accept_rate"].rename(f"fold{i}") for i, c in enumerate(fold_curves)], axis=1
    )

    summary = pd.DataFrame({
        "dollar_accept_rate": dollar_by_fold.mean(axis=1),
        "cv_mean_sortino": sortino_by_fold.mean(axis=1),
        "cv_std_sortino": sortino_by_fold.std(axis=1),
    })
    print("\n=== cross-validated Sortino by K (mean +/- std across 5 folds) ===")
    print(summary.to_string(float_format=lambda x: f"{x:0.4f}"))

    eligible = summary[summary["dollar_accept_rate"] >= MIN_DOLLAR_ACCEPT_RATE]
    best_k = float(eligible["cv_mean_sortino"].idxmax())
    print(f"\nselected K*_cv = {best_k:.2f}  "
          f"(cv mean sortino={eligible.loc[best_k, 'cv_mean_sortino']:.4f} "
          f"+/- {eligible.loc[best_k, 'cv_std_sortino']:.4f})")

    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(config.OUTPUTS_DIR / "cv_k_sweep.csv")
    with open(config.MODELS_DIR / "best_k_cv.json", "w") as f:
        json.dump({"best_k_cv": best_k}, f)
    print(f"\nsaved {config.OUTPUTS_DIR / 'cv_k_sweep.csv'} and {config.MODELS_DIR / 'best_k_cv.json'}")


if __name__ == "__main__":
    main()
