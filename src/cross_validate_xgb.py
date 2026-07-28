"""Same 5-fold CV robustness check as cross_validate.py (for LightGBM), but for
XGBoost -- it showed the highest raw correlation(pred, actual ER) of the five
models compared in compare_models.py, so it's the one worth checking whether
its K-tuned Sortino edge survives cross-validation or is single-split noise."""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from . import config
from .compare_models import fit_xgboost
from .data import load_dataset
from .features import feature_matrix
from .metrics import portfolio_summary, top_k_selection
from .returns import load_risk_free_table

N_FOLDS = 5
K_GRID = np.arange(0.05, 1.01, 0.05)
MIN_DOLLAR_ACCEPT_RATE = 0.05


def main():
    print("loading fully-matured training data...")
    rf_table = load_risk_free_table()
    df = load_dataset(config.TRAIN_CSV, rf_table, matured_only=True).reset_index(drop=True)
    print(f"  {len(df):,} loans")

    X_full = feature_matrix(df)
    cat_cols = [c for c in X_full.columns if str(X_full[c].dtype) == "category"]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_curves = []
    fold_corrs = []
    for fold_i, (tr_idx, hold_idx) in enumerate(kf.split(df)):
        print(f"\n--- fold {fold_i + 1}/{N_FOLDS} ---")
        df_fit = df.iloc[tr_idx]
        df_hold = df.iloc[hold_idx].copy()

        idx_tr, idx_es = train_test_split(df_fit.index, test_size=0.1, random_state=config.RANDOM_STATE)
        X_tr, X_es = feature_matrix(df_fit.loc[idx_tr]), feature_matrix(df_fit.loc[idx_es])
        y_tr, y_es = df_fit.loc[idx_tr, "ER"], df_fit.loc[idx_es, "ER"]

        predict_fn = fit_xgboost(X_tr, y_tr, X_es, y_es, cat_cols)

        X_hold = feature_matrix(df_hold)
        for c in cat_cols:
            X_hold[c] = X_hold[c].astype("category")
        df_hold["ER_hat"] = predict_fn(X_hold)
        corr = np.corrcoef(df_hold["ER_hat"], df_hold["ER"])[0, 1]
        fold_corrs.append(corr)
        print(f"held-out corr(pred,actual) = {corr:.4f}")

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

    print(f"\nheld-out corr across folds: mean={np.mean(fold_corrs):.4f} std={np.std(fold_corrs):.4f}")

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
    print("\n=== XGBoost cross-validated Sortino by K (mean +/- std across 5 folds) ===")
    print(summary.to_string(float_format=lambda x: f"{x:0.4f}"))

    eligible = summary[summary["dollar_accept_rate"] >= MIN_DOLLAR_ACCEPT_RATE]
    best_k = float(eligible["cv_mean_sortino"].idxmax())
    print(f"\nselected K*_cv = {best_k:.2f}  "
          f"(cv mean sortino={eligible.loc[best_k, 'cv_mean_sortino']:.4f} "
          f"+/- {eligible.loc[best_k, 'cv_std_sortino']:.4f})")

    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(config.OUTPUTS_DIR / "cv_k_sweep_xgb.csv")
    sortino_by_fold.join(dollar_by_fold, rsuffix="_dollar").to_csv(config.OUTPUTS_DIR / "cv_fold_curves_xgb.csv")
    with open(config.MODELS_DIR / "best_k_cv_xgb.json", "w") as f:
        json.dump({"best_k_cv": best_k, "fold_corrs": fold_corrs}, f)
    print(f"\nsaved outputs")


if __name__ == "__main__":
    main()
