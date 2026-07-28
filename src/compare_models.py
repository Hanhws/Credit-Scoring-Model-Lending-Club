"""Compare several model families the 우수사례 reports used (Linear Regression,
Random Forest, XGBoost, CatBoost) against our LightGBM, on the exact same
data/target/evaluation procedure, to see whether model choice changes the
conclusion that Grade A-C beats a model-selected portfolio.

SVM (also used in 우수사례1) is skipped: SVR training is roughly O(n^2)-O(n^3),
infeasible at ~750k training rows.
"""
import json
import time

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from . import config
from .data import load_dataset
from .features import feature_matrix
from .metrics import portfolio_summary, top_k_selection
from .returns import load_risk_free_table

K_GRID = np.arange(0.05, 1.01, 0.05)
MIN_DOLLAR_ACCEPT_RATE = 0.05


def onehot_align(X_tr, X_val, X_test, cat_cols):
    def dummies(X):
        return pd.get_dummies(X, columns=cat_cols, dummy_na=True)

    d_tr = dummies(X_tr)
    d_val = dummies(X_val).reindex(columns=d_tr.columns, fill_value=0)
    d_test = dummies(X_test).reindex(columns=d_tr.columns, fill_value=0)
    return d_tr, d_val, d_test


def impute_numeric(*frames):
    return [f.fillna(-1) for f in frames]


def fit_lightgbm(X_tr, y_tr, X_val, y_val, cat_cols):
    params = {
        "objective": "regression_l1", "metric": "l1", "learning_rate": 0.03,
        "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
        "verbosity": -1, "seed": config.RANDOM_STATE,
    }
    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_set, free_raw_data=False)
    booster = lgb.train(params, train_set, num_boost_round=3000, valid_sets=[val_set],
                         callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    return lambda X: booster.predict(X, num_iteration=booster.best_iteration)


def fit_xgboost(X_tr, y_tr, X_val, y_val, cat_cols, return_model=False):
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype("category")
        X_val[c] = X_val[c].astype("category")
    model = xgb.XGBRegressor(
        n_estimators=3000, learning_rate=0.03, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", enable_categorical=True,
        eval_metric="mae", early_stopping_rounds=100,
        random_state=config.RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    predict_fn = lambda X: model.predict(X)
    if return_model:
        return predict_fn, model
    return predict_fn


def fit_catboost(X_tr, y_tr, X_val, y_val, cat_cols):
    # CatBoost rejects NaN inside categorical columns -- needs an explicit string
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype(str)
        X_val[c] = X_val[c].astype(str)
    model = catboost.CatBoostRegressor(
        iterations=3000, learning_rate=0.03, depth=8,
        loss_function="MAE", eval_metric="MAE", cat_features=cat_cols,
        random_seed=config.RANDOM_STATE, early_stopping_rounds=100, verbose=False,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

    def predict(X):
        X = X.copy()
        for c in cat_cols:
            X[c] = X[c].astype(str)
        return model.predict(X)

    return predict


def fit_random_forest(X_tr, y_tr):
    model = RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=50,
        n_jobs=-1, random_state=config.RANDOM_STATE,
    )
    model.fit(X_tr, y_tr)
    return lambda X: model.predict(X)


def fit_linear(X_tr, y_tr):
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    return lambda X: model.predict(X)


def evaluate(name, predict_fn, X_val, df_val, X_test, df_test):
    pred_val = np.asarray(predict_fn(X_val))
    pred_test = np.asarray(predict_fn(X_test))
    corr_val = np.corrcoef(pred_val, df_val["ER"])[0, 1]
    corr_test = np.corrcoef(pred_test, df_test["ER"])[0, 1]

    df_val = df_val.copy()
    df_val["ER_hat"] = pred_val
    df_test = df_test.copy()
    df_test["ER_hat"] = pred_test

    rows = []
    for k in K_GRID:
        sel = top_k_selection(df_val, "ER_hat", k)
        s = portfolio_summary(sel)
        s["k_frac"] = round(k, 2)
        s["dollar_accept_rate"] = sel["funded_amnt"].sum() / df_val["funded_amnt"].sum()
        rows.append(s)
    val_sweep = pd.DataFrame(rows)
    eligible = val_sweep[val_sweep["dollar_accept_rate"] >= MIN_DOLLAR_ACCEPT_RATE]
    best_k = float(eligible.loc[eligible["sortino"].idxmax(), "k_frac"])

    sel_test = top_k_selection(df_test, "ER_hat", best_k)
    test_summary = portfolio_summary(sel_test)

    return {
        "model": name,
        "corr_val": round(corr_val, 4),
        "corr_test": round(corr_test, 4),
        "best_k_val": best_k,
        "test_dollar_accept_rate": round(sel_test["funded_amnt"].sum() / df_test["funded_amnt"].sum(), 4),
        "test_sharpe": round(test_summary["sharpe"], 4),
        "test_sortino": round(test_summary["sortino"], 4),
    }


def main():
    print("loading data...")
    rf_table = load_risk_free_table()
    df_train_full = load_dataset(config.TRAIN_CSV, rf_table, matured_only=True)
    df_test = load_dataset(config.TEST_CSV, rf_table, matured_only=True)
    print(f"  train={len(df_train_full):,}  test={len(df_test):,}")

    idx_tr, idx_val = train_test_split(df_train_full.index, test_size=0.15, random_state=config.RANDOM_STATE)
    df_tr, df_val = df_train_full.loc[idx_tr], df_train_full.loc[idx_val]

    X_tr_native, X_val_native, X_test_native = feature_matrix(df_tr), feature_matrix(df_val), feature_matrix(df_test)
    cat_cols = [c for c in X_tr_native.columns if str(X_tr_native[c].dtype) == "category"]
    y_tr, y_val = df_tr["ER"], df_val["ER"]

    print("one-hot encoding for linear/RF...")
    X_tr_oh, X_val_oh, X_test_oh = onehot_align(X_tr_native, X_val_native, X_test_native, cat_cols)
    X_tr_oh, X_val_oh, X_test_oh = impute_numeric(X_tr_oh, X_val_oh, X_test_oh)

    results = []

    print("\n=== baselines (model-independent) ===")
    base_all = portfolio_summary(df_test)
    base_grade = portfolio_summary(df_test[df_test["grade"].isin(["A", "B", "C"])])
    print(f"invest-all   sharpe={base_all['sharpe']:.3f} sortino={base_all['sortino']:.3f}")
    print(f"grade A-C    sharpe={base_grade['sharpe']:.3f} sortino={base_grade['sortino']:.3f}")

    print("\n=== Linear Regression ===")
    t0 = time.time()
    pred_fn = fit_linear(X_tr_oh, y_tr)
    results.append(evaluate("Linear Regression", pred_fn, X_val_oh, df_val, X_test_oh, df_test))
    print(results[-1], f"({time.time()-t0:.0f}s)")

    print("\n=== Random Forest ===")
    t0 = time.time()
    pred_fn = fit_random_forest(X_tr_oh, y_tr)
    results.append(evaluate("Random Forest", pred_fn, X_val_oh, df_val, X_test_oh, df_test))
    print(results[-1], f"({time.time()-t0:.0f}s)")

    print("\n=== XGBoost ===")
    t0 = time.time()
    X_tr_xgb, X_val_xgb, X_test_xgb = X_tr_native.copy(), X_val_native.copy(), X_test_native.copy()
    pred_fn = fit_xgboost(X_tr_xgb, y_tr, X_val_xgb, y_val, cat_cols)
    for c in cat_cols:
        X_test_xgb[c] = X_test_xgb[c].astype("category")
    results.append(evaluate("XGBoost", pred_fn, X_val_xgb, df_val, X_test_xgb, df_test))
    print(results[-1], f"({time.time()-t0:.0f}s)")

    print("\n=== CatBoost ===")
    t0 = time.time()
    pred_fn = fit_catboost(X_tr_native, y_tr, X_val_native, y_val, cat_cols)
    results.append(evaluate("CatBoost", pred_fn, X_val_native, df_val, X_test_native, df_test))
    print(results[-1], f"({time.time()-t0:.0f}s)")

    print("\n=== LightGBM (reference) ===")
    t0 = time.time()
    pred_fn = fit_lightgbm(X_tr_native, y_tr, X_val_native, y_val, cat_cols)
    results.append(evaluate("LightGBM", pred_fn, X_val_native, df_val, X_test_native, df_test))
    print(results[-1], f"({time.time()-t0:.0f}s)")

    out = pd.DataFrame(results)
    out["baseline_invest_all_sortino"] = base_all["sortino"]
    out["baseline_grade_abc_sortino"] = base_grade["sortino"]
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    out.to_csv(config.OUTPUTS_DIR / "model_comparison.csv", index=False)
    print("\n=== final comparison ===")
    print(out.to_string(index=False))
    print(f"\nsaved to {config.OUTPUTS_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
