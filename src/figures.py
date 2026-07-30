"""Every figure that carries a step of the final argument, and nothing else.

Each function answers one question a reader will ask, in the order the report asks
it. Anything that does not change the conclusion was left out rather than kept as
decoration.

Design constraints held throughout
----------------------------------
Colour is used for identity only, and only the three categorical slots that
validate as a set: blue, orange, aqua. Grid and axes recede; values are labelled
directly on the marks rather than left to the reader to trace against an axis. No
figure uses two y-axes -- when two quantities of different scale belong in one
story they become side-by-side panels, because a dual axis lets the author choose
the crossing point and therefore the conclusion.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

from . import config

FIG_DIR = config.ROOT / "시각화"
RESULTS_DIR = config.ROOT / "결과"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.dpi": 160,
    # the Korean UI face has no U+2212, and matplotlib reaches for it by default
    "axes.unicode_minus": False,
})
PCT = FuncFormatter(lambda v, _: f"{v * 100:.0f}%")
# small ranges: 1.05% and 1.33% must not both render as "1%"
PCT1 = FuncFormatter(lambda v, _: f"{v * 100:.1f}%")


def _label(s):
    """Strategy names for figures. The combining circumflex in ER-hat has no glyph
    in the Korean UI face, so it is spelled out rather than dropped silently."""
    return (s.replace("ER̂", "예측수익")
             .replace("ER̂", "예측수익")
             .replace("̂", ""))


def _clean(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)
    return ax


def _save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")
    return path


# ---------------------------------------------------------------------------

def fig_split(blocks):
    """The three time blocks: what the model learns from, stops on, and is judged by."""
    fig, ax = plt.subplots(figsize=(9, 2.4))
    colors = {"train": BLUE, "valid": AQUA, "test": ORANGE}
    roles = {"train": "모델 학습", "valid": "학습 중단 시점 결정", "test": "규칙 결정"}
    # the third block CHOOSES the rule; calling it "test" invites confusion with
    # the held-out test FILE, so the display name says what it does
    display = {"train": "train", "valid": "valid", "test": "선택(test)"}
    for i, r in blocks.iterrows():
        start, end = pd.Timestamp(r["시작"]), pd.Timestamp(r["종료"])
        ax.barh(0, (end - start).days, left=start, height=0.42,
                color=colors[r["블록"]], zorder=3)
        mid = start + (end - start) / 2
        ax.text(mid, 0, f"{display[r['블록']]}\n{r['빈티지수']}개 빈티지",
                ha="center", va="center", color="white", fontsize=10, weight="bold",
                zorder=4)
        ax.text(mid, -0.34, f"{roles[r['블록']]}\n{int(r['대출건수']):,}건",
                ha="center", va="top", color=INK2, fontsize=9)
    ax.set_ylim(-0.75, 0.32)
    ax.set_yticks([])
    ax.grid(False)
    _clean(ax, ("top", "right", "left"))
    ax.set_title("그림 1. 학습 데이터를 시간 순서로 셋으로 나눈다", loc="left", pad=12)
    ax.set_xlabel("대출 발행 시점")
    return _save(fig, "fig1_데이터분할.png")


def fig_leak(corr_honest, corr_leak, sharpe_honest, sharpe_leak):
    """Two panels, because the two quantities have nothing to do with each other's
    scale: what shuffling does to prediction, and what it does to the headline."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (vals, title, fmt) in zip(axes, [
        ([corr_honest, corr_leak], "예측 정확도 (실제 수익과의 상관)", "{:.3f}"),
        ([sharpe_honest, sharpe_leak], "그 결과로 보고되는 Sharpe", "{:.2f}"),
    ]):
        bars = ax.bar(["시간 순서로\n나눔 (정직)", "무작위로 섞어\n나눔 (부풀려짐)"], vals,
                      color=[BLUE, ORANGE], width=0.55, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"  {fmt.format(v)}",
                    ha="center", va="bottom", color=INK, fontsize=12, weight="bold")
        ax.set_title(title, loc="left", fontsize=11.5, pad=8)
        ax.set_ylim(0, max(vals) * 1.28)
        _clean(ax)
    fig.suptitle("그림 2. 데이터를 무작위로 섞으면 성적이 부풀려진다",
                 x=0.02, ha="left", fontsize=13, weight="bold", y=1.08)
    fig.text(0.02, 0.98, "같은 최종 규칙(전략 ②)에 데이터 분할만 바꿔 적용 — 최종 사다리의 ②와 ⑥에 해당",
             ha="left", color=INK2, fontsize=9.5)
    fig.tight_layout()
    return _save(fig, "fig2_시대누출.png")


def fig_grade(grade_tbl):
    """Risk against return for each grade cut, plus the Sharpe each one implies."""
    t = grade_tbl[grade_tbl["전략"].str.contains("등급")].copy()
    t["컷"] = t["전략"].str.extract(r"등급 (\S+)")
    t["전략"] = t["전략"].map(_label)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))

    ax = axes[0]
    ax.plot(t["변동성"], t["평균초과수익"], "-", color=BASELINE, lw=1.5, zorder=2)
    ax.scatter(t["변동성"], t["평균초과수익"], s=90, color=BLUE, zorder=3,
               edgecolor=SURFACE, linewidth=2)
    # A-D through A-G sit almost on top of each other, so their labels fan out
    offsets = {"A-D": (-8, 9), "A-E": (12, 7), "A-F": (14, -5), "A-G": (12, -17)}
    for _, r in t.iterrows():
        ax.annotate(r["컷"], (r["변동성"], r["평균초과수익"]),
                    textcoords="offset points",
                    xytext=offsets.get(r["컷"], (9, -4)),
                    color=INK2, fontsize=10)
    ax.xaxis.set_major_formatter(PCT1); ax.yaxis.set_major_formatter(PCT1)
    ax.margins(x=0.16, y=0.16)
    ax.set_xlabel("위험 (빈티지별 수익의 표준편차)")
    ax.set_ylabel("수익 (빈티지 평균 초과수익)")
    ax.set_title("등급을 조이면 위험이 먼저 줄어든다", loc="left", fontsize=11.5, pad=8)
    _clean(ax)

    ax = axes[1]
    best = t["Sharpe"].idxmax()
    colors = [ORANGE if i == best else BLUE for i in t.index]
    bars = ax.barh(t["컷"], t["Sharpe"], color=colors, height=0.6, zorder=3)
    for b, v in zip(bars, t["Sharpe"]):
        ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:.2f}",
                va="center", color=INK, fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlim(0, t["Sharpe"].max() * 1.22)
    ax.set_xlabel("Sharpe ratio")
    ax.set_title("등급 컷별 Sharpe", loc="left", fontsize=11.5, pad=8)
    _clean(ax)

    fig.suptitle("그림 3. 어느 등급까지 빌려줄 것인가", x=0.02, ha="left",
                 fontsize=13, weight="bold", y=1.04)
    fig.text(0.02, -0.04,
             "개발 단계 숫자 — train 파일의 선택 블록에서 잰 값. 최종 평가(다른 대출)와 다른 데이터라 Sharpe가 조금 다르다.",
             ha="left", color=MUTED, fontsize=9)
    fig.tight_layout()
    return _save(fig, "fig3_등급컷.png")


def fig_lambda(lam):
    """The price of each extra step of tightening, and where it jumps."""
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    v = lam["lambda*"].to_numpy(dtype=float)
    knee = int(np.argmax(v))
    colors = [CRITICAL if i == knee else BLUE for i in range(len(v))]
    bars = ax.bar(lam["단계"], v, color=colors, width=0.58, zorder=3)
    for b, x in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, x,
                f"{x:.0f}", ha="center",
                va="bottom" if x >= 0 else "top",
                color=INK, fontsize=10.5, weight="bold")
    ax.axhline(0, color=BASELINE, lw=1.2, zorder=2)
    ax.set_ylabel("무차별 위험회피도 λ*")
    ax.set_title("그림 4. 한 단계 더 조이는 값이 어디서 갑자기 비싸지는가",
                 loc="left", pad=12)
    ax.text(0.99, 0.04,
            "값이 낮거나 음수면 조이는 게 거의 공짜\n마지막 단계에서 값이 튀면 거기가 멈출 곳",
            transform=ax.transAxes, ha="right", va="bottom",
            color=INK2, fontsize=9.5)
    _clean(ax)
    fig.tight_layout()
    return _save(fig, "fig4_무차별람다.png")


def fig_pool(pool_dev, pool_test):
    """Small multiples of the applicant mix across the whole span it moved over.

    Deliberately not one chart with two axes. DTI and the interest rate live on
    unrelated scales, and a dual axis would let the crossing point -- and so the
    apparent story -- be placed anywhere the author liked.

    The development and evaluation pools are drawn as one continuous line with the
    handover marked, because the point is that the mix drifts continuously; the
    rule only ever reads the part to the left of wherever it is standing.
    """
    pool = pd.concat([pool_dev, pool_test]).sort_index()
    pool = pool[~pool.index.duplicated(keep="last")]
    boundary = pd.DatetimeIndex(pool_test.index).min()

    cols = [("dti", "부채 대비 소득 비율 (DTI)", "높을수록 빚 부담이 큼"),
            ("int_rate", "적용 금리 (%)", "위험에 대한 보상"),
            ("fico_avg", "평균 신용점수 (FICO)", "높을수록 신용이 좋음")]
    cols = [c for c in cols if c[0] in pool.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.3))
    x = pd.DatetimeIndex(pool.index)
    for ax, (col, title, note) in zip(np.atleast_1d(axes), cols):
        y = pool[col].to_numpy(dtype=float)
        ax.plot(x, y, color=BLUE, lw=1.8, zorder=3)
        ax.axvline(boundary, color=BASELINE, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.annotate(f"{y[0]:.1f}", (x[0], y[0]), textcoords="offset points",
                    xytext=(0, -16), color=INK, fontsize=10, weight="bold", ha="left")
        ax.annotate(f"{y[-1]:.1f}", (x[-1], y[-1]), textcoords="offset points",
                    xytext=(0, 9), color=INK, fontsize=10, weight="bold", ha="right")
        ax.set_title(title, loc="left", fontsize=11, pad=6)
        ax.set_xlabel(note, fontsize=9.5, color=MUTED)
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.margins(y=0.18)
        _clean(ax)
    np.atleast_1d(axes)[0].text(
        boundary, np.atleast_1d(axes)[0].get_ylim()[1], " 평가 구간 →",
        color=MUTED, fontsize=9, va="top")
    fig.suptitle("그림 5. 신청서만 봐도 보인다 — 빚 부담은 10년간 꾸준히 올랐다",
                 x=0.02, ha="left", fontsize=13, weight="bold", y=1.05)
    fig.tight_layout()
    return _save(fig, "fig5_신청자풀.png")


def fig_dial(dial):
    """Why the timing rule was rejected: it tightened early and stayed open late.

    Two stacked panels on a shared time axis rather than one chart with two scales.
    A dual axis would let the two lines be slid until they appeared to agree; on a
    shared axis the reader sees for themselves that the dial's low points and the
    bad vintages do not line up.
    """
    d = dial.dropna(subset=["다이얼"])
    x = pd.DatetimeIndex(d.index)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 4.9), sharex=True,
                             gridspec_kw={"hspace": 0.35})

    ax = axes[0]
    y = d["다이얼"].to_numpy(dtype=float)
    ax.fill_between(x, 0, y, color=BLUE, alpha=0.12, zorder=2)
    ax.plot(x, y, color=BLUE, lw=2, zorder=3)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("규칙이 집행한 비율")
    ax.set_title("규칙이 조인 시점", loc="left", fontsize=11.5, pad=6)
    _clean(ax)

    ax = axes[1]
    if "실제_초과수익" in d.columns:
        r = d["실제_초과수익"].to_numpy(dtype=float)
        ax.axhline(0, color=BASELINE, lw=1.3, zorder=2)
        ax.fill_between(x, 0, r, where=r >= 0, color=AQUA, alpha=0.18, zorder=2)
        ax.fill_between(x, 0, r, where=r < 0, color=ORANGE, alpha=0.18, zorder=2)
        ax.plot(x, r, color=ORANGE, lw=2, zorder=3)
        ax.yaxis.set_major_formatter(PCT)
        ax.set_ylabel("그 달 대출의 실제 성적")
    ax.set_title("실제로 나빴던 시점", loc="left", fontsize=11.5, pad=6)
    _clean(ax)

    axes[0].annotate("좋은 달에 조였다", (x[2], y[2]),
                     textcoords="offset points", xytext=(14, -30),
                     color=INK, fontsize=10, weight="bold",
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    mid = len(x) // 2 + 4
    axes[0].annotate("무너지는 동안 전액 집행", (x[mid], y[mid]),
                     textcoords="offset points", xytext=(-30, -36),
                     ha="center", color=INK, fontsize=10, weight="bold",
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

    fig.suptitle("그림 6. 규칙이 조인 시점과 실제로 나빴던 시점이 어긋난다",
                 x=0.02, ha="left", fontsize=13, weight="bold", y=1.0)
    return _save(fig, "fig6_타이밍기각.png")


def fig_series(series):
    """The two return series the whole Sharpe comparison is computed from."""
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    x = pd.DatetimeIndex(series.index)
    for col, color in [("전량 투자", ORANGE), ("최종 규칙", BLUE)]:
        y = series[col].to_numpy(dtype=float)
        ax.plot(x, y, color=color, lw=2, zorder=3, label=col)
        ax.annotate(col, (x[-1], y[-1]), textcoords="offset points",
                    xytext=(8, 0), color=color, fontsize=10.5, weight="bold",
                    va="center")
    ax.axhline(0, color=BASELINE, lw=1.4, zorder=2)
    # mark where the credit cycle turns: the first vintage both strategies lose on
    both_neg = np.where((series["전량 투자"].to_numpy() < 0)
                        & (series["최종 규칙"].to_numpy() < 0))[0]
    if len(both_neg):
        x0 = x[both_neg[0]]
        ax.axvline(x0, color=BASELINE, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.text(x0, ax.get_ylim()[1], " 여기부터 두 전략 모두 손실 →",
                color=MUTED, fontsize=9, va="top")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("그 달 대출의 초과수익")
    ax.set_title("그림 7. 같은 시기, 두 전략이 실제로 겪은 성적", loc="left", pad=12)
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    ax.margins(x=0.12)
    _clean(ax)
    fig.tight_layout()
    return _save(fig, "fig7_빈티지수익계열.png")


def fig_ladder(final):
    """The headline, with the interval around it and the downside beside it.

    Rows are ordered so each strategy sits next to the control that judges it --
    ⑤ (matched-rate random) directly above ② (the final rule), ④ (time-constant)
    next to ③ (timing) -- instead of in circled-number order, so the figure makes
    the comparison the text makes without the reader hopping rows."""
    t = final.copy()
    order = {c: i for i, c in enumerate("①⑤②③④⑥")}
    t = t.sort_values("전략", key=lambda s: s.str[0].map(order)).reset_index(drop=True)
    t["짧은이름"] = t["전략"].str.replace(r"\s*\(.*\)", "", regex=True).map(_label)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    y = np.arange(len(t))[::-1]
    ax.hlines(y, t["CI하한"], t["CI상한"], color=BASELINE, lw=3, zorder=2)
    ax.scatter(t["Sharpe"], y, s=100, color=BLUE, zorder=4,
               edgecolor=SURFACE, linewidth=2)
    for yi, r in zip(y, t.itertuples()):
        ax.text(r.Sharpe, yi + 0.28, f"{r.Sharpe:.2f}", ha="center",
                color=INK, fontsize=10.5, weight="bold")
    ax.set_yticks(y, t["짧은이름"], fontsize=10)
    ax.set_ylim(-0.7, len(t) - 1 + 0.85)
    ax.tick_params(left=False)
    ax.axvline(0, color=BASELINE, lw=1.2)
    ax.set_xlabel("Sharpe ratio (막대는 95% 신뢰구간)")
    ax.set_title("Sharpe — 점추정과 불확실성", loc="left", fontsize=11.5, pad=8)
    _clean(ax, ("top", "right", "left"))

    ax = axes[1]
    worst = t["최악빈티지"].to_numpy(dtype=float)
    bars = ax.barh(y, worst, color=ORANGE, height=0.55, zorder=3)
    for b, v in zip(bars, worst):
        ax.text(v, b.get_y() + b.get_height() / 2, f"{v * 100:.1f}%  ",
                va="center", ha="right", color=INK, fontsize=10)
    ax.set_yticks(y, [""] * len(y))
    ax.set_ylim(-0.7, len(t) - 1 + 0.85)
    ax.tick_params(left=False)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("가장 나빴던 달의 손실")
    ax.set_title("최악의 경우", loc="left", fontsize=11.5, pad=8)
    _clean(ax, ("top", "right", "left"))

    fig.suptitle("그림 8. 최종 결과 — 규칙을 얼린 뒤 평가 데이터에 한 번 적용",
                 x=0.02, ha="left", fontsize=13, weight="bold", y=1.04)
    fig.tight_layout()
    return _save(fig, "fig8_최종사다리.png")


def fig_deltas(deltas):
    """The four differences the conclusion rests on, each with an interval.

    A point estimate of "+0.508" is a claim; the interval and the p-value are the
    evidence for it. Two of these clear zero comfortably and one does not, and the
    figure is arranged so that fact is the first thing visible.
    """
    t = deltas.copy()
    y = np.arange(len(t))[::-1]
    sig = t["ΔSharpe_p"].to_numpy(dtype=float) < 0.05

    fig, ax = plt.subplots(figsize=(9.6, 0.72 * len(t) + 1.9))
    ax.axvline(0, color=BASELINE, lw=1.4, zorder=2)
    for yi, r, s_ in zip(y, t.itertuples(), sig):
        color = BLUE if s_ else MUTED
        ax.hlines(yi, r.ΔSharpe_CI하한, r.ΔSharpe_CI상한, color=color, lw=3,
                  alpha=0.35, zorder=3)
        ax.scatter([r.ΔSharpe], [yi], s=105, color=color, zorder=4,
                   edgecolor=SURFACE, linewidth=2)
        # a bootstrap p of 0/5000 is "<0.001", never "0.000" -- p-values are not zero
        p_txt = "p<0.001" if r.ΔSharpe_p < 0.001 else f"p={r.ΔSharpe_p:.3f}"
        note = f"{r.ΔSharpe:+.2f}   {p_txt}" + ("" if s_ else "  (유의하지 않음)")
        ax.text(r.ΔSharpe, yi + 0.30, note, ha="center", color=INK,
                fontsize=10.5, weight="bold" if s_ else "normal")
    ax.set_yticks(y, t["비교"], fontsize=10)
    ax.set_ylim(-0.7, len(t) - 1 + 0.9)
    ax.tick_params(left=False)
    ax.set_xlabel("Sharpe 차이 (막대는 95% 부트스트랩 신뢰구간)")
    ax.set_title("그림 9. 각 기여가 0과 구별되는가", loc="left", pad=12)
    _clean(ax, ("top", "right", "left"))
    fig.tight_layout()
    return _save(fig, "fig9_차이검정.png")


def main():
    print("그림 생성...")
    rd = RESULTS_DIR
    blocks = pd.read_csv(rd / "분할_블록.csv")
    grade = pd.read_csv(rd / "개발_등급컷.csv")
    lam = pd.read_csv(rd / "개발_무차별lambda.csv")
    final = pd.read_csv(rd / "최종_사다리.csv")
    dial = pd.read_csv(rd / "최종_타이밍다이얼.csv", index_col=0, parse_dates=True)
    series = pd.read_csv(rd / "최종_빈티지수익계열.csv", index_col=0, parse_dates=True)
    pool = pd.read_csv(rd / "최종_신청자풀구성.csv", index_col=0, parse_dates=True)
    pool_dev = pd.read_csv(rd / "개발_신청자풀구성.csv", index_col=0, parse_dates=True)
    summary = pd.read_csv(rd / "최종_모델요약.csv", index_col=0).iloc[:, 0]
    deltas = pd.read_csv(rd / "최종_차이검정.csv")

    honest = float(summary["corr_시간분할"])
    leaky = float(summary["corr_무작위분할"])
    # ⑥ is the leaky-model twin of ② (grade cut x gate, no timing), so the honest
    # bar it is compared against must be ② -- not ③, whose timing dial was rejected
    s_final = float(final.loc[final["전략"].str.startswith("②"), "Sharpe"].iloc[0])
    s_leak = float(final.loc[final["전략"].str.startswith("⑥"), "Sharpe"].iloc[0])

    fig_split(blocks)
    fig_leak(honest, leaky, s_final, s_leak)
    fig_grade(grade)
    fig_lambda(lam)
    fig_pool(pool_dev, pool)
    fig_dial(dial)
    fig_series(series)
    fig_ladder(final)
    fig_deltas(deltas)
    print(f"완료 -> {FIG_DIR}")


if __name__ == "__main__":
    main()
