"""Final recommended model: XGBoost, with a parameter-free selection rule
(invest wherever predicted excess return > 0) instead of a grid-searched top-K%
per cohort. See discussion in the guide doc for why: grid-searching K over a
metric and reporting whichever K "won" has no story behind it and is a mild
form of look-ahead bias even under cross-validation. Testing an economically
motivated rule (would a rational lender fund this loan over buying a
treasury?) removes the free parameter entirely, and it also outperforms the
searched K in practice, while investing far more capital.

Uses config.SAFE_FEATURE_COLS (feature_matrix(..., safe_only=True)), not the
full feature list. ~50 of the original ~87 features are 100% missing for
pre-2012 vintages and ~0% missing from 2016 on (LC only started collecting
those credit-bureau fields around 2015-2016) -- a tree model can read "is this
NaN" as a proxy for loan age/era, which would let calendar-time information
leak in through missingness patterns without any date feature ever being
used. Retraining on the "safe" subset (features whose missingness doesn't
swing by more than 30pp between 2007-2010 and 2016-2018 vintages) reproduces
almost the same correlation (0.193 vs 0.221) and the same cyclical
acceptance-rate behavior (2008-09 ~81%, 2011-13 ~97%), which is the evidence
that the behavior is real borrower-level credit signal, not an era-via-
missingness shortcut.
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from . import config
from .compare_models import fit_xgboost
from .data import load_dataset
from .features import feature_matrix
from .metrics import cohort_returns, portfolio_summary, threshold_selection, top_k_selection
from .returns import load_risk_free_table


def main():
    print("loading data...")
    rf_table = load_risk_free_table()
    df_train_full = load_dataset(config.TRAIN_CSV, rf_table, matured_only=True)
    df_test = load_dataset(config.TEST_CSV, rf_table, matured_only=True)

    idx_tr, idx_val = train_test_split(df_train_full.index, test_size=0.15, random_state=config.RANDOM_STATE)
    df_tr, df_val = df_train_full.loc[idx_tr], df_train_full.loc[idx_val]

    X_tr, X_val, X_test = (feature_matrix(df_tr, safe_only=True), feature_matrix(df_val, safe_only=True),
                            feature_matrix(df_test, safe_only=True))
    cat_cols = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]
    y_tr, y_val = df_tr["ER"], df_val["ER"]
    print(f"using {X_tr.shape[1]} era-safe features (of {len(config.FEATURE_COLS)} total)")

    print("training XGBoost...")
    pred_fn, booster = fit_xgboost(X_tr, y_tr, X_val, y_val, cat_cols, return_model=True)
    for c in cat_cols:
        X_test[c] = X_test[c].astype("category")
    df_test = df_test.copy()
    df_test["ER_hat"] = pred_fn(X_test)
    print("corr(pred,actual) on test:", np.corrcoef(df_test["ER_hat"], df_test["ER"])[0, 1])

    config.MODELS_DIR.mkdir(exist_ok=True)
    booster.save_model(str(config.MODELS_DIR / "xgb_er_model_safe.json"))

    print("\n=== rule comparison on test set ===")
    rows = []

    def add(name, sub):
        s = portfolio_summary(sub)
        rows.append({
            "전략": name,
            "대출건수": s["n_loans"],
            "달러인수율": round(sub["funded_amnt"].sum() / df_test["funded_amnt"].sum(), 4),
            "Sharpe": round(s["sharpe"], 4),
            "Sortino": round(s["sortino"], 4),
        })
        print(name, s)

    add("전량 투자 (베이스라인)", df_test)
    add("Grade A-C만 (베이스라인)", df_test[df_test["grade"].isin(["A", "B", "C"])])
    add("XGBoost top-45% (그리드서치 K)", top_k_selection(df_test, "ER_hat", 0.45))
    add("XGBoost ER_hat>0 (파라미터 없는 규칙)", threshold_selection(df_test, "ER_hat"))

    out = pd.DataFrame(rows)
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    out.to_csv(config.OUTPUTS_DIR / "final_rule_comparison.csv", index=False, encoding="utf-8-sig")
    print("\n", out.to_string(index=False))

    # monthly acceptance rate under the threshold rule -- shows the rule
    # dialing origination volume up/down with the cycle rather than a fixed quota
    df_test["accepted"] = df_test["ER_hat"] > 0
    monthly = df_test.groupby("issue_month").apply(
        lambda g: pd.Series({
            "달러인수율": g.loc[g["accepted"], "funded_amnt"].sum() / g["funded_amnt"].sum(),
        }),
        include_groups=False,
    )
    monthly.index.name = "발행월"
    monthly.to_csv(config.OUTPUTS_DIR / "threshold_rule_monthly_acceptance.csv")
    print(f"\n2008-2009 평균 인수율: {monthly.loc['2008-01':'2009-12', '달러인수율'].mean():.3f}")
    print(f"2011-2013 평균 인수율: {monthly.loc['2011-01':'2013-12', '달러인수율'].mean():.3f}")

    # cohort ER series for the threshold-rule portfolio, for the chart
    sel = threshold_selection(df_test, "ER_hat")
    cr = cohort_returns(sel)["ER"].rename("XGBoost ER_hat>0")
    cr.index.name = "발행월"
    cr.to_csv(config.OUTPUTS_DIR / "threshold_rule_cohort_er.csv")

    print(f"\nsaved outputs to {config.OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
