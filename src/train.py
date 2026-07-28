import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .data import load_dataset
from .features import feature_matrix
from .metrics import portfolio_summary, top_k_selection
from .returns import load_risk_free_table

# guard against picking a K so small the Sortino estimate is just noise from a
# handful of loans/cohorts
MIN_DOLLAR_ACCEPT_RATE = 0.05
K_GRID = np.arange(0.05, 1.01, 0.05)


def main():
    print("loading risk-free table...")
    rf_table = load_risk_free_table()

    print("loading + preparing training data (matured loans only)...")
    df = load_dataset(config.TRAIN_CSV, rf_table, matured_only=True)
    print(f"  {len(df):,} matured loans")

    idx_tr, idx_val = train_test_split(
        df.index, test_size=0.15, random_state=config.RANDOM_STATE
    )
    df_tr, df_val = df.loc[idx_tr], df.loc[idx_val]

    X_tr, X_val = feature_matrix(df_tr), feature_matrix(df_val)
    y_tr, y_val = df_tr["ER"], df_val["ER"]

    cat_cols = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]

    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_set, free_raw_data=False)

    params = {
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

    print("training LightGBM regressor (target = ER, cumulative excess return)...")
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=3000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    pred_val = booster.predict(X_val, num_iteration=booster.best_iteration)
    mae = np.mean(np.abs(pred_val - y_val))
    corr = np.corrcoef(pred_val, y_val)[0, 1]
    print(f"\nval MAE(ER)={mae:.4f}  corr(pred,actual)={corr:.4f}  best_iter={booster.best_iteration}")

    imp = pd.Series(
        booster.feature_importance(importance_type="gain"), index=booster.feature_name()
    ).sort_values(ascending=False).head(15)
    print("\ntop 15 features (gain):")
    print(imp.to_string())

    # --- tune the top-K% selection threshold on the validation cohorts,
    # maximizing Sortino directly, so the test set stays untouched until the
    # very last, one-shot evaluation ---
    df_val = df_val.copy()
    df_val["ER_hat"] = pred_val

    print("\ntuning K (top-K%% per cohort) on validation to maximize Sortino...")
    rows = []
    for k in K_GRID:
        sel = top_k_selection(df_val, "ER_hat", k)
        s = portfolio_summary(sel)
        s["k_frac"] = round(k, 2)
        s["dollar_accept_rate"] = sel["funded_amnt"].sum() / df_val["funded_amnt"].sum()
        rows.append(s)
    sweep = pd.DataFrame(rows)[["k_frac", "dollar_accept_rate", "n_loans", "sharpe", "sortino"]]
    print(sweep.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    eligible = sweep[sweep["dollar_accept_rate"] >= MIN_DOLLAR_ACCEPT_RATE]
    best_row = eligible.loc[eligible["sortino"].idxmax()]
    best_k = float(best_row["k_frac"])
    print(f"\nselected K*={best_k:.2f} (val sortino={best_row['sortino']:.4f}, "
          f"dollar accept rate={best_row['dollar_accept_rate']:.1%})")

    config.MODELS_DIR.mkdir(exist_ok=True)
    booster.save_model(str(config.MODELS_DIR / "lgbm_er_model.txt"))
    joblib.dump(list(X_tr.columns), config.MODELS_DIR / "feature_columns.pkl")
    with open(config.MODELS_DIR / "best_k.json", "w") as f:
        json.dump({"best_k": best_k}, f)
    print(f"\nsaved model + tuned K to {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
