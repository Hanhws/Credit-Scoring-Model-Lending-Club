import json

import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from . import config
from .data import load_dataset
from .features import feature_matrix
from .metrics import cohort_returns, portfolio_summary, top_k_selection
from .returns import load_risk_free_table

# validated categorical palette (see dataviz skill references/palette.md), light mode
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "AppleGothic",
    "axes.unicode_minus": False,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})


def build_data():
    rf_table = load_risk_free_table()
    df = load_dataset(config.TEST_CSV, rf_table, matured_only=True)
    booster = lgb.Booster(model_file=str(config.MODELS_DIR / "lgbm_er_model.txt"))
    X = feature_matrix(df)
    df["ER_hat"] = booster.predict(X, num_iteration=booster.best_iteration)
    with open(config.MODELS_DIR / "best_k.json") as f:
        best_k = json.load(f)["best_k"]
    return df, best_k


def cohort_timeseries(df, best_k):
    strategies = {
        "전량 투자 (현행)": df,
        "Grade A-C만": df[df["grade"].isin(["A", "B", "C"])],
        f"모델 top-{best_k:.0%}": top_k_selection(df, "ER_hat", best_k),
    }
    series = {}
    for name, sub in strategies.items():
        cr = cohort_returns(sub)
        series[name] = cr["ER"]
    return pd.DataFrame(series), strategies


def plot_cohort_timeseries(ts, path):
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    colors = {ts.columns[0]: BLUE, ts.columns[1]: ORANGE, ts.columns[2]: AQUA}
    for col in ts.columns:
        s = ts[col].rolling(6, min_periods=3).mean()
        ax.plot(s.index, s.values, linewidth=2, color=colors[col], label=col, solid_capstyle="round")

    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.text(ts.index.min(), 0, "  무위험금리(Rf) 수준", color=MUTED, fontsize=9, va="bottom")

    ax.set_title("코호트(발행월)별 초과수익률 ER — 6개월 이동평균", color=INK, fontsize=13, loc="left", pad=14)
    ax.set_ylabel("초과수익률 (ER, 연환산)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_strategy_bars(df, best_k, path):
    strategies = {
        "전량 투자\n(현행)": df,
        "Grade A-C만": df[df["grade"].isin(["A", "B", "C"])],
        f"모델\ntop-{best_k:.0%}": top_k_selection(df, "ER_hat", best_k),
    }
    rows = []
    for name, sub in strategies.items():
        s = portfolio_summary(sub)
        rows.append({"strategy": name, "Sharpe": s["sharpe"], "Sortino": s["sortino"]})
    tbl = pd.DataFrame(rows).set_index("strategy")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    x = np.arange(len(tbl))
    width = 0.32
    bars1 = ax.bar(x - width / 2, tbl["Sharpe"], width, color=BLUE, label="Sharpe")
    bars2 = ax.bar(x + width / 2, tbl["Sortino"], width, color=AQUA, label="Sortino")

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 4 if h >= 0 else -14),
                        ha="center", fontsize=9, color=INK_SECONDARY)

    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(tbl.index, fontsize=10)
    ax.set_title("전략별 위험조정수익률 비교 (테스트셋)", color=INK, fontsize=13, loc="left", pad=14)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_k_sweep(df, path):
    rows = []
    for k in np.arange(0.05, 1.01, 0.05):
        sel = top_k_selection(df, "ER_hat", k)
        s = portfolio_summary(sel)
        rows.append({"k": k, "sortino": s["sortino"],
                      "dollar_accept_rate": sel["funded_amnt"].sum() / df["funded_amnt"].sum()})
    sweep = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(sweep["dollar_accept_rate"], sweep["sortino"], linewidth=2, color=BLUE, marker="o", markersize=5)
    best_idx = sweep["sortino"].idxmax()
    ax.scatter([sweep.loc[best_idx, "dollar_accept_rate"]], [sweep.loc[best_idx, "sortino"]],
               s=90, color=ORANGE, zorder=5, label=f"최고 Sortino (K={sweep.loc[best_idx,'k']:.0%})")

    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xlabel("투자 인수율 (달러 기준)")
    ax.set_ylabel("Sortino ratio")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("선별 강도(K)에 따른 Sortino 비율 — 테스트셋", color=INK, fontsize=13, loc="left", pad=14)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_cv_instability(path):
    folds = pd.read_csv(config.OUTPUTS_DIR / "cv_fold_curves.csv")
    summary = pd.read_csv(config.OUTPUTS_DIR / "cv_k_sweep.csv")

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    for fold_i, g in folds.groupby("fold"):
        g = g.sort_values("dollar_accept_rate")
        ax.plot(g["dollar_accept_rate"], g["sortino"], linewidth=1.2, color=MUTED,
                alpha=0.8, label="개별 fold (5개)" if fold_i == 0 else None)

    mean = summary["cv_mean_sortino"]
    std = summary["cv_std_sortino"]
    x = summary["dollar_accept_rate"]
    ax.fill_between(x, mean - std, mean + std, color=BLUE, alpha=0.12, linewidth=0)
    ax.plot(x, mean, linewidth=2.5, color=BLUE, marker="o", markersize=5, label="5-fold 평균 (±1 표준편차)")

    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xlabel("투자 인수율 (달러 기준)")
    ax.set_ylabel("Sortino ratio")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("K값별 Sortino의 fold간 불안정성 (5-fold 교차검증)", color=INK, fontsize=13, loc="left", pad=14)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


BASE_INVEST_ALL_SHARPE, BASE_INVEST_ALL_SORTINO = 0.791, 1.526
BASE_GRADE_ABC_SHARPE, BASE_GRADE_ABC_SORTINO = 0.971, 2.193


def plot_model_comparison_bars(path):
    mc = pd.read_csv(config.OUTPUTS_DIR / "model_comparison.csv")

    names = ["전량 투자\n(베이스라인)", "Grade A-C\n(베이스라인)"]
    sharpe = [BASE_INVEST_ALL_SHARPE, BASE_GRADE_ABC_SHARPE]
    sortino = [BASE_INVEST_ALL_SORTINO, BASE_GRADE_ABC_SORTINO]
    for _, row in mc.iterrows():
        names.append(f"{row['model']}\n(top-{row['best_k_val']:.0%})")
        sharpe.append(row["test_sharpe"])
        sortino.append(row["test_sortino"])

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    x = np.arange(len(names))
    width = 0.32
    bars1 = ax.bar(x - width / 2, sharpe, width, color=BLUE, label="Sharpe")
    bars2 = ax.bar(x + width / 2, sortino, width, color=AQUA, label="Sortino")
    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 4 if h >= 0 else -14),
                        ha="center", fontsize=9, color=INK_SECONDARY)

    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_title("베이스라인 vs 모델별 위험조정수익률 비교 (테스트셋, 각 모델 자체 최적 K)",
                 color=INK, fontsize=13, loc="left", pad=14)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_xgb_cv_instability(path):
    folds = pd.read_csv(config.OUTPUTS_DIR / "cv_fold_curves_xgb.csv")
    summary = pd.read_csv(config.OUTPUTS_DIR / "cv_k_sweep_xgb.csv")
    fold_cols = [c for c in folds.columns if c.startswith("fold") and not c.endswith("_dollar")]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    for i, c in enumerate(fold_cols):
        dollar_col = f"{c}_dollar"
        ax.plot(folds[dollar_col], folds[c], linewidth=1.2, color=MUTED, alpha=0.8,
                label="개별 fold (5개)" if i == 0 else None)

    mean = summary["cv_mean_sortino"]
    std = summary["cv_std_sortino"]
    x = summary["dollar_accept_rate"]
    ax.fill_between(x, mean - std, mean + std, color=BLUE, alpha=0.12, linewidth=0)
    ax.plot(x, mean, linewidth=2.5, color=BLUE, marker="o", markersize=5, label="5-fold 평균 (±1 표준편차)")
    ax.axhline(BASE_GRADE_ABC_SORTINO, color=ORANGE, linewidth=1.5, linestyle="--",
               label=f"Grade A-C 베이스라인 ({BASE_GRADE_ABC_SORTINO:.2f})")

    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xlabel("투자 인수율 (달러 기준)")
    ax.set_ylabel("Sortino ratio")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("XGBoost: K값별 Sortino의 fold간 안정성 (5-fold 교차검증)", color=INK, fontsize=13, loc="left", pad=14)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_threshold_rule_acceptance(path):
    monthly = pd.read_csv(config.OUTPUTS_DIR / "threshold_rule_monthly_acceptance.csv",
                           parse_dates=["발행월"], index_col="발행월")

    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    ax.plot(monthly.index, monthly["달러인수율"], linewidth=1.6, color=BLUE)
    ax.fill_between(monthly.index, 0, monthly["달러인수율"], color=BLUE, alpha=0.08)

    ax.axvspan(pd.Timestamp("2008-01-01"), pd.Timestamp("2009-12-31"), color=ORANGE, alpha=0.10)
    ax.text(pd.Timestamp("2008-06-01"), 0.05, "금융위기\n(평균 81%)", color=INK_SECONDARY, fontsize=9, ha="center")
    ax.axvspan(pd.Timestamp("2011-01-01"), pd.Timestamp("2013-12-31"), color=AQUA, alpha=0.10)
    ax.text(pd.Timestamp("2012-06-01"), 0.05, "호황기\n(평균 97%)", color=INK_SECONDARY, fontsize=9, ha="center")

    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel("달러 인수율 (ER_hat>0 규칙)")
    ax.set_title("XGBoost 'ER_hat>0' 규칙의 월별 투자 인수율 — 경기에 따라 자동 조절",
                 color=INK, fontsize=13, loc="left", pad=14)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_final_rule_bars(path):
    fc = pd.read_csv(config.OUTPUTS_DIR / "final_rule_comparison.csv")
    names = [n.replace(" (", "\n(") for n in fc["전략"]]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    x = np.arange(len(names))
    width = 0.32
    bars1 = ax.bar(x - width / 2, fc["Sharpe"], width, color=BLUE, label="Sharpe")
    bars2 = ax.bar(x + width / 2, fc["Sortino"], width, color=AQUA, label="Sortino")
    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=9, color=INK_SECONDARY)

    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_title("최종 비교: 그리드서치 K vs 파라미터 없는 규칙 (테스트셋)",
                 color=INK, fontsize=13, loc="left", pad=14)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_acceptance_by_grade(path):
    by_grade = pd.read_csv(config.OUTPUTS_DIR / "acceptance_by_grade.csv", index_col=0)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.bar(by_grade.index, by_grade["달러인수율"], color=BLUE, width=0.6)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.0%}", (b.get_x() + b.get_width() / 2, h), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9.5, color=INK_SECONDARY)

    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("LC 자체 등급 (grade)")
    ax.set_ylabel("달러 인수율 (ER_hat>0 규칙)")
    ax.set_title("등급별 인수율 — LC 자체 등급과 방향이 일치", color=INK, fontsize=13, loc="left", pad=14)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_acceptance_by_feature(path):
    by_feat = pd.read_csv(config.OUTPUTS_DIR / "acceptance_by_feature.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    labels = ["DTI", "FICO", "신용카드 사용률", "금리"]
    colors = [BLUE, AQUA, ORANGE, BLUE]
    for ax, label, color in zip(axes.flat, labels, colors):
        sub = by_feat[by_feat["변수"] == label].reset_index(drop=True)
        x = np.arange(len(sub))
        bars = ax.bar(x, sub["달러인수율"], color=color, width=0.6)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.0%}", (b.get_x() + b.get_width() / 2, h), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8.5, color=INK_SECONDARY)
        ax.set_xticks(x)
        ax.set_xticklabels(["Q1(최저)", "Q2", "Q3", "Q4", "Q5(최고)"], fontsize=8.5)
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_title(label, color=INK, fontsize=12, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle("신용변수 5분위 구간별 인수율 (ER_hat>0 규칙)", color=INK, fontsize=13.5, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_shap_importance(path):
    imp = pd.read_csv(config.OUTPUTS_DIR / "shap_importance.csv", index_col=0).iloc[:, 0]
    imp = imp.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set_xlabel("평균 |SHAP| (예측 초과수익률에 대한 기여도)")
    ax.set_title("SHAP 변수 중요도 — 무엇이 accept/reject를 가르는가", color=INK, fontsize=13, loc="left", pad=14)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_accept_reject_profile(path):
    prof = pd.read_csv(config.OUTPUTS_DIR / "accept_reject_profile.csv")
    prof = prof.dropna(subset=["차이_퍼센트"]).sort_values("차이_퍼센트")

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    colors = [ORANGE if v < 0 else BLUE for v in prof["차이_퍼센트"]]
    bars = ax.barh(prof["변수"], prof["차이_퍼센트"], color=colors)
    for b, v in zip(bars, prof["차이_퍼센트"]):
        ax.annotate(f"{v:+.0f}%", (v, b.get_y() + b.get_height() / 2), textcoords="offset points",
                    xytext=(6 if v >= 0 else -6, 0), va="center",
                    ha="left" if v >= 0 else "right", fontsize=9.5, color=INK_SECONDARY)

    ax.set_xlim(-27, 37)
    ax.axvline(0, color=BASELINE, linewidth=1.2)
    ax.set_xlabel("거절 그룹 중앙값이 승인 그룹 중앙값보다 몇 % 높은가/낮은가")
    ax.set_title("승인 vs 거절 대출의 프로필 차이 (중앙값 기준)", color=INK, fontsize=13, loc="left", pad=14)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_accept_reject_grade_dist(path):
    gdist = pd.read_csv(config.OUTPUTS_DIR / "accept_reject_grade_dist.csv", index_col=0)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    x = np.arange(len(gdist))
    width = 0.32
    ax.bar(x - width / 2, gdist["승인"], width, color=BLUE, label="승인 대출 중 비중")
    ax.bar(x + width / 2, gdist["거절"], width, color=ORANGE, label="거절 대출 중 비중")
    ax.set_xticks(x)
    ax.set_xticklabels(gdist.index)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("grade")
    ax.set_title("승인/거절 그룹 각각의 등급 구성비 — 거절 그룹은 D~G에 집중", color=INK, fontsize=13, loc="left", pad=14)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    print("scoring test set + building cohort series...")
    df, best_k = build_data()

    ts, _ = cohort_timeseries(df, best_k)
    plot_cohort_timeseries(ts, config.OUTPUTS_DIR / "cohort_excess_return.png")
    print(f"saved {config.OUTPUTS_DIR / 'cohort_excess_return.png'}")

    plot_strategy_bars(df, best_k, config.OUTPUTS_DIR / "strategy_comparison.png")
    print(f"saved {config.OUTPUTS_DIR / 'strategy_comparison.png'}")

    plot_k_sweep(df, config.OUTPUTS_DIR / "k_sweep.png")
    print(f"saved {config.OUTPUTS_DIR / 'k_sweep.png'}")

    if (config.OUTPUTS_DIR / "cv_fold_curves.csv").exists():
        plot_cv_instability(config.OUTPUTS_DIR / "cv_instability.png")
        print(f"saved {config.OUTPUTS_DIR / 'cv_instability.png'}")


if __name__ == "__main__":
    main()
