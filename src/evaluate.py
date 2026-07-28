import json

import lightgbm as lgb
import pandas as pd

from . import config
from .data import load_dataset
from .features import feature_matrix
from .metrics import portfolio_summary, top_k_selection
from .returns import load_risk_free_table


def main():
    print("loading risk-free table + test data (matured loans only)...")
    rf_table = load_risk_free_table()
    df = load_dataset(config.TEST_CSV, rf_table, matured_only=True)
    print(f"  {len(df):,} matured test loans")

    booster = lgb.Booster(model_file=str(config.MODELS_DIR / "lgbm_er_model.txt"))
    X = feature_matrix(df)
    df["ER_hat"] = booster.predict(X, num_iteration=booster.best_iteration)

    with open(config.MODELS_DIR / "best_k.json") as f:
        best_k = json.load(f)["best_k"]
    print(f"using K*={best_k:.2f} tuned on the training validation split (test set untouched until now)")

    print("\n=== Baseline: invest in every matured test loan (status quo) ===")
    baseline = portfolio_summary(df)
    print_summary(baseline)

    print("\n=== Baseline: safe-grade only (grade A/B/C) ===")
    grade_abc = df[df["grade"].isin(["A", "B", "C"])]
    grade_summary = portfolio_summary(grade_abc)
    print_summary(grade_summary)
    grade_accept_rate = grade_abc["funded_amnt"].sum() / df["funded_amnt"].sum()
    print(f"  (dollar acceptance rate = {grade_accept_rate:.1%})")

    print(f"\n=== Model-selected portfolio: top-{best_k:.0%} per cohort (K* tuned on validation) ===")
    final_sel = top_k_selection(df, "ER_hat", best_k)
    final_summary = portfolio_summary(final_sel)
    print_summary(final_summary)
    print(f"  (dollar acceptance rate = {final_sel['funded_amnt'].sum() / df['funded_amnt'].sum():.1%})")

    print("\n=== Reference sweep on test set (NOT used for tuning, informational only) ===")
    rows = []
    for k in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        sel = top_k_selection(df, "ER_hat", k)
        s = portfolio_summary(sel)
        s["k_frac"] = k
        s["dollar_accept_rate"] = sel["funded_amnt"].sum() / df["funded_amnt"].sum()
        rows.append(s)
    sweep = pd.DataFrame(rows)[
        ["k_frac", "dollar_accept_rate", "n_loans", "mean_excess_return", "sharpe", "sortino"]
    ]
    print(sweep.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    sweep.to_csv(config.OUTPUTS_DIR / "sweep_results.csv", index=False)
    df[["id", "issue_month", "grade", "term_months", "funded_amnt", "total_pymnt",
        "Rf", "ER", "ER_hat"]].to_csv(config.OUTPUTS_DIR / "test_predictions.csv", index=False)
    print(f"\nsaved sweep + per-loan predictions to {config.OUTPUTS_DIR}")


def print_summary(s):
    print(f"  n_loans={s['n_loans']:,}  n_cohorts={s['n_cohorts']}  "
          f"funded=${s['funded_amnt']:,.0f}  mean_ER={s['mean_excess_return']:.4f}  "
          f"sharpe={s['sharpe']:.3f}  sortino={s['sortino']:.3f}")


if __name__ == "__main__":
    main()
