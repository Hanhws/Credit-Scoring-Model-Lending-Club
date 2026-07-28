"""Explain WHY the ER_hat>0 rule accepts/rejects loans, instead of treating the
resulting ~89% acceptance rate as an unexplained number. Breaks the decision
down by grade/sub_grade and by key credit variables (dti, fico, revol_util,
int_rate), and uses SHAP on the era-safe XGBoost model to show which features
actually drive a loan's predicted excess return above or below zero.
"""
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from . import config
from .data import load_dataset
from .features import feature_matrix
from .returns import load_risk_free_table


def dollar_rate_by(df, group_col, weight_col="funded_amnt", accept_col="accepted"):
    def agg(g):
        return pd.Series({
            "대출건수": len(g),
            "총투자금액": g[weight_col].sum(),
            "달러인수율": g.loc[g[accept_col], weight_col].sum() / g[weight_col].sum(),
        })
    return df.groupby(group_col).apply(agg, include_groups=False)


def qbucket(s, q=5):
    return pd.qcut(s, q, duplicates="drop")


def main():
    print("loading data + scoring with the era-safe XGBoost model...")
    rf_table = load_risk_free_table()
    df = load_dataset(config.TEST_CSV, rf_table, matured_only=True)

    booster = xgb.XGBRegressor()
    booster.load_model(str(config.MODELS_DIR / "xgb_er_model_safe.json"))
    X = feature_matrix(df, safe_only=True)
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    for c in cat_cols:
        X[c] = X[c].astype("category")
    df["ER_hat"] = booster.predict(X)
    df["accepted"] = df["ER_hat"] > 0

    config.OUTPUTS_DIR.mkdir(exist_ok=True)

    print("\n=== grade별 인수율 ===")
    by_grade = dollar_rate_by(df, "grade")
    print(by_grade.round(4))
    by_grade.round(4).to_csv(config.OUTPUTS_DIR / "acceptance_by_grade.csv", encoding="utf-8-sig")

    print("\n=== sub_grade별 인수율 ===")
    by_subgrade = dollar_rate_by(df, "sub_grade")
    by_subgrade.round(4).to_csv(config.OUTPUTS_DIR / "acceptance_by_subgrade.csv", encoding="utf-8-sig")

    print("\n=== 신용변수 구간별 인수율 ===")
    rows = []
    for col, label in [("dti", "DTI"), ("fico_avg", "FICO"), ("revol_util", "신용카드 사용률"),
                        ("int_rate", "금리")]:
        df[f"{col}_bucket"] = qbucket(df[col], 5)
        res = dollar_rate_by(df, f"{col}_bucket")
        res["변수"] = label
        res = res.reset_index().rename(columns={f"{col}_bucket": "구간"})
        rows.append(res)
    by_feature = pd.concat(rows, ignore_index=True)
    print(by_feature.round(4).to_string(index=False))
    by_feature.round(4).to_csv(config.OUTPUTS_DIR / "acceptance_by_feature.csv", index=False, encoding="utf-8-sig")

    print("\ncomputing SHAP values (sample of 20,000 test loans)...")
    sample = df.sample(n=min(20000, len(df)), random_state=config.RANDOM_STATE)
    X_sample = feature_matrix(sample, safe_only=True)
    for c in cat_cols:
        X_sample[c] = X_sample[c].astype("category")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X_sample.columns)
    mean_abs_shap = mean_abs_shap.sort_values(ascending=False)
    print("\n=== SHAP 변수 중요도 (|ER_hat|에 대한 평균 기여도, 상위 15개) ===")
    print(mean_abs_shap.head(15).round(5))
    mean_abs_shap.rename("mean_abs_shap").round(5).to_csv(config.OUTPUTS_DIR / "shap_importance.csv",
                                                            encoding="utf-8-sig")

    # per-loan shap values for the top features, for scatter/dependence plots
    top_feats = mean_abs_shap.head(8).index.tolist()
    shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)[top_feats]
    shap_df = shap_df.add_suffix("_shap")
    detail = pd.concat([X_sample[top_feats].reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)
    detail.to_csv(config.OUTPUTS_DIR / "shap_detail_top_features.csv", index=False, encoding="utf-8-sig")

    print(f"\nsaved outputs to {config.OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
