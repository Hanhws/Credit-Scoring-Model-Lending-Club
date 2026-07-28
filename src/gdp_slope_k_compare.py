"""For each of the 5 models compared earlier, replace the single global K
(grid search) with a K that varies by (borrower state, GDP data year) based on
the trailing slope of that state's real GDP growth rate -- accelerating
states get a looser K, decelerating states get a tighter one (src/gdp_slope.py).
All 5 models are retrained on the era-safe feature set (config.SAFE_FEATURE_COLS)
for a fair, leakage-checked comparison.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .compare_models import fit_catboost, fit_lightgbm, fit_linear, fit_random_forest, fit_xgboost, onehot_align, impute_numeric
from .data import load_dataset
from .features import feature_matrix
from .gdp_slope import attach_gdp_k, load_state_gdp_slope
from .metrics import portfolio_summary, variable_k_selection
from .returns import load_risk_free_table


def align_categories(X_ref, *others):
    """Pin every categorical column's category set to X_ref's (typically the
    training frame). feature_matrix() builds X_tr/X_val/X_test independently,
    so pandas infers each one's categories from whatever values happen to be
    present in that particular slice -- if they don't line up exactly,
    LightGBM's Dataset raises "categorical_feature do not match" at predict
    time (XGBoost/CatBoost are more lenient about this, which is why this only
    surfaced here)."""
    cat_cols = [c for c in X_ref.columns if str(X_ref[c].dtype) == "category"]
    out = []
    for X in others:
        X = X.copy()
        for c in cat_cols:
            X[c] = X[c].cat.set_categories(X_ref[c].cat.categories)
        out.append(X)
    return out


def main():
    print("loading data...")
    rf_table = load_risk_free_table()
    df_train_full = load_dataset(config.TRAIN_CSV, rf_table, matured_only=True)
    df_test = load_dataset(config.TEST_CSV, rf_table, matured_only=True)

    print("attaching GDP-slope-derived K...")
    slope_table = load_state_gdp_slope()
    df_test = attach_gdp_k(df_test, slope_table)
    print(f"  gdp_k stats: min={df_test['gdp_k'].min():.2f} median={df_test['gdp_k'].median():.2f} "
          f"max={df_test['gdp_k'].max():.2f}  (missing state match: {df_test['gdp_slope'].isna().mean():.1%})")

    idx_tr, idx_val = train_test_split(df_train_full.index, test_size=0.15, random_state=config.RANDOM_STATE)
    df_tr, df_val = df_train_full.loc[idx_tr], df_train_full.loc[idx_val]

    X_tr_native = feature_matrix(df_tr, safe_only=True)
    X_val_native = feature_matrix(df_val, safe_only=True)
    X_test_native = feature_matrix(df_test, safe_only=True)
    cat_cols = [c for c in X_tr_native.columns if str(X_tr_native[c].dtype) == "category"]
    X_val_native, X_test_native = align_categories(X_tr_native, X_val_native, X_test_native)
    y_tr, y_val = df_tr["ER"], df_val["ER"]

    print("one-hot encoding for linear/RF...")
    X_tr_oh, X_val_oh, X_test_oh = onehot_align(X_tr_native, X_val_native, X_test_native, cat_cols)
    X_tr_oh, X_val_oh, X_test_oh = impute_numeric(X_tr_oh, X_val_oh, X_test_oh)

    group_cols = ["addr_state", "available_gdp_year"]
    results = []

    def run(name, X_tr, X_test_scored):
        sel = variable_k_selection(df_test, "ER_hat", "gdp_k", group_cols)
        s = portfolio_summary(sel)
        results.append({
            "모델": name,
            "대출건수": s["n_loans"],
            "달러인수율": round(sel["funded_amnt"].sum() / df_test["funded_amnt"].sum(), 4),
            "Sharpe": round(s["sharpe"], 4),
            "Sortino": round(s["sortino"], 4),
        })
        print(results[-1])

    print("\n=== Linear Regression ===")
    pred_fn = fit_linear(X_tr_oh, y_tr)
    df_test["ER_hat"] = pred_fn(X_test_oh)
    run("Linear Regression", X_tr_oh, X_test_oh)

    print("\n=== Random Forest ===")
    pred_fn = fit_random_forest(X_tr_oh, y_tr)
    df_test["ER_hat"] = pred_fn(X_test_oh)
    run("Random Forest", X_tr_oh, X_test_oh)

    print("\n=== XGBoost ===")
    X_tr_x, X_val_x, X_test_x = X_tr_native.copy(), X_val_native.copy(), X_test_native.copy()
    pred_fn = fit_xgboost(X_tr_x, y_tr, X_val_x, y_val, cat_cols)
    for c in cat_cols:
        X_test_x[c] = X_test_x[c].astype("category")
    df_test["ER_hat"] = pred_fn(X_test_x)
    run("XGBoost", X_tr_x, X_test_x)

    print("\n=== CatBoost ===")
    pred_fn = fit_catboost(X_tr_native, y_tr, X_val_native, y_val, cat_cols)
    df_test["ER_hat"] = pred_fn(X_test_native)
    run("CatBoost", X_tr_native, X_test_native)

    print("\n=== LightGBM ===")
    pred_fn = fit_lightgbm(X_tr_native, y_tr, X_val_native, y_val, cat_cols)
    df_test["ER_hat"] = pred_fn(X_test_native)
    run("LightGBM", X_tr_native, X_test_native)

    out = pd.DataFrame(results)
    base_all = portfolio_summary(df_test)
    base_grade = portfolio_summary(df_test[df_test["grade"].isin(["A", "B", "C"])])
    out["베이스라인_전량투자_Sortino"] = base_all["sortino"]
    out["베이스라인_GradeA-C_Sortino"] = base_grade["sortino"]

    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    out.to_csv(config.OUTPUTS_DIR / "gdp_slope_k_model_comparison.csv", index=False, encoding="utf-8-sig")
    print("\n=== 최종 비교 (GDP 기울기 기반 K, era-안전 변수) ===")
    print(out.to_string(index=False))
    print(f"\n베이스라인 -- 전량투자 Sortino={base_all['sortino']:.4f}, Grade A-C Sortino={base_grade['sortino']:.4f}")
    print(f"saved to {config.OUTPUTS_DIR / 'gdp_slope_k_model_comparison.csv'}")


if __name__ == "__main__":
    main()
