"""Develop the rule on the training file, then open the test file once.

The discipline this file enforces
--------------------------------
Every number in this project that turned out to be wrong was wrong for the same
reason: something was decided using data that also measured it. A grid search
over K picked the K that looked best on the evaluation set. A shuffled split let
the model learn the era it was pricing. A recalibration read outcomes that would
not have been known yet.

So the work is split into two stages with a wall between them.

  develop()   sees only lending_club_2020_train.csv, cut into three
              time-ordered blocks by split.py. The model is fitted on `train`,
              stopped on `valid`, and every choice -- which grades to lend to,
              whether the risk-free gate earns its place, how the timing dial is
              set -- is made by looking at `test`. Nothing here reports a
              headline number; it only decides.

  finalize()  opens lending_club_2020_test.csv with the rule already frozen,
              applies it once, and reports. No parameter is chosen at this stage,
              so there is nothing left for the evaluation set to flatter.

The frozen rule is three parts, in the order they were established:

  1. grade cut       lend only to grades A through B
  2. risk-free gate  and only where predicted excess return clears Treasuries

A third part -- a timing dial that funded less of a vintage when the applicant pool
looked worse -- was built, tested, and REJECTED. Its forecast correlates 0.03 with
what the vintages went on to earn; it tightened through the best months of the
window and stayed fully open through the collapse. It is reported as a negative
finding rather than dropped from the record, because the reason it fails is the
substance: a level signal read from application forms does not travel.

Term
----
36-month loans only. A 60-month loan written after 2015-12 has not matured by
the 2021-01 snapshot, so the 60-month series simply stops in 2016 and never
experiences the 2017-18 deterioration that the whole question turns on. Mixing
the two produces a series whose composition changes partway through; reporting
60-month separately produces one that ends before the interesting part. Neither
supports a conclusion, and saying so is more useful than averaging them.
"""
import numpy as np
import pandas as pd

from . import config, inference, metrics, model, rules, split, timing
from .data import load_dataset
from .returns import load_risk_free_table

TERM = 36

# Every configuration this project actually ran, counted so the deflated Sharpe can
# be charged for all of it. Searching until something looks significant is exactly
# what DSR exists to penalise, and a count that omits the failed searches is not a
# correction but a decoration.
#    7  grade cuts            A-G .. A alone
#    2  terms                 36-month, 60-month
#    3  level recalibrations  none, matured-vintage, twelve-month seasoning
#    7  evaluation windows    2009..2015 start dates swept for autocorrelation
#    3  model splits          time cutoff, shuffled, annual walk-forward
#    1  bucket allocator      max-Sharpe over grade buckets (corner solution)
#   51  frontier cells        grade cut x quota, with the risk-free gate
#    4  cash-flow timings     charge-off / recovery lag pairs
#   19  K sweep               top-K quota grid
#    1  timing re-specification  the dial's reference level, corrected after the
#                               first version returned a null (see timing.py)
TRIALS_PRE_FREEZE = 98

# Diagnostics run AFTER the rule was frozen. They reuse the evaluation data, so
# they are searches too and have to be charged to the same budget -- leaving them
# out would understate the benchmark the observed Sharpe has to clear.
#    4  algorithm comparison  LightGBM / CatBoost / Ridge / XGBoost (2.2.4절)
#    1  random-trim control   approval-rate-matched random thinning (2.3.2절)
TRIALS_POST_FREEZE = 5

TRIALS_RUN = TRIALS_PRE_FREEZE + TRIALS_POST_FREEZE  # 103

# The knee rule picks the cut just before lambda* jumps clear of the earlier steps.
# The multiplier was chosen AFTER seeing the pattern, which is a researcher degree
# of freedom, so the report shows what the other plausible multipliers would pick.
KNEE_MULTIPLIER = 2.0
KNEE_SENSITIVITY = (1.5, 2.0, 3.0)

GRADE_CUTS = {"A-G": list("ABCDEFG"), "A-F": list("ABCDEF"), "A-E": list("ABCDE"),
              "A-D": list("ABCD"), "A-C": list("ABC"), "A-B": list("AB"), "A": ["A"]}


# ---------------------------------------------------------------------------
# shared measurement
# ---------------------------------------------------------------------------

def measure(df, w, rf_table, label, base_er=None):
    """One strategy, scored on the vintage excess-return series it produces."""
    vt = metrics.trim_thin_vintages(
        metrics.vintage_table(df, weights=w, rf_table=rf_table))
    er = vt["excess"].to_numpy(dtype=float)
    lo, hi, _ = inference.sharpe_bootstrap_ci(er)
    n_eff, rho = inference.effective_sample_size(er)
    dd = float(np.sqrt(np.mean(np.minimum(er, 0.0) ** 2)))
    row = {
        "전략": label,
        "집행률": rules.dollar_accept_rate(df, w),
        "승인률변동성": float(vt["accept_rate"].std(ddof=1)),
        "평균초과수익": float(er.mean()),
        "변동성": float(er.std(ddof=1)),
        "Sharpe": inference.sharpe(er),
        "CI하한": lo, "CI상한": hi,
        "Sortino": float(er.mean() / dd) if dd > 0 else np.nan,
        "손실빈티지비율": float((er < 0).mean()),
        "최악빈티지": float(er.min()),
        "빈티지수": len(er),
        "유효표본수": float(n_eff),
        "자기상관": float(rho),
        "PSR": inference.probabilistic_sharpe_ratio(er, n_eff=max(n_eff, 2.0)),
    }
    if base_er is not None and len(base_er) == len(er):
        t = inference.paired_mean_test(er, base_er)
        row["Δ평균초과수익"] = t["mean_diff"]
        row["대응HAC_p"] = t["p_value"]
    return row, er


def knee_cut(lam, order, multiplier):
    """Stop at the cut just before the price of tightening jumps clear of the
    earlier steps. `multiplier` sets what counts as a jump, relative to the median
    absolute price so far."""
    v = lam["lambda*"].to_numpy(dtype=float)
    return order[int(np.argmax(v > multiplier * np.median(np.abs(v))))]


def indifference_lambda(er_loose, er_tight):
    """The risk aversion at which two choices are equally good.

    Mean-variance utility is U = mu - (lambda/2) * sigma^2. Setting the utilities
    equal and solving gives lambda* = 2 * (mu_loose - mu_tight) / (var_loose -
    var_tight): tighten only if your own lambda exceeds it. Reading where lambda*
    jumps is how the grade cut gets chosen without anyone naming a lambda.
    """
    dmu = er_loose.mean() - er_tight.mean()
    dvar = er_loose.var(ddof=1) - er_tight.var(ddof=1)
    return float(2 * dmu / dvar) if dvar != 0 else np.nan


def delta_tests(sers, pairs):
    """Test the differences the conclusion actually rests on.

    Two things go in the table that the earlier draft left out.

    The first is a test on the SHARPE difference. The paired HAC test measures the
    mean excess return gap and says nothing about the ratio, yet the ratio is what
    is being claimed; a stationary bootstrap that resamples both series on the SAME
    block indices preserves the pairing and gives the difference an interval.

    The second is the difference series' OWN autocorrelation and effective sample
    size. Differencing removes the common credit cycle, which is the reason the
    paired test has any power at all -- but "removes" is not "eliminates", and with
    26 observations a Newey-West standard error is itself poorly determined. The
    honest thing is to publish how much independent information the difference
    carries rather than to assert that pairing solves the problem.
    """
    rows = []
    for a, b in pairs:
        if a not in sers or b not in sers:
            continue
        ea, eb = np.asarray(sers[a]), np.asarray(sers[b])
        if len(ea) != len(eb):
            continue
        d = inference.sharpe_diff_bootstrap(ea, eb)
        t = inference.paired_mean_test(ea, eb)
        n_eff_d, rho_d = inference.effective_sample_size(ea - eb)
        rows.append({
            "비교": f"{a}  vs  {b}",
            "ΔSharpe": d["sharpe_diff"],
            "ΔSharpe_CI하한": d["ci_low"], "ΔSharpe_CI상한": d["ci_high"],
            "ΔSharpe_p": d["p_value"],
            "Δ평균초과수익": t["mean_diff"], "대응HAC_p": t["p_value"],
            "차분_자기상관": float(rho_d), "차분_유효표본수": float(n_eff_d),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# stage 1 -- decide, using the training file only
# ---------------------------------------------------------------------------

def load_blocks(rf_table):
    df = load_dataset(config.TRAIN_CSV, rf_table, matured_only=True)
    df = df[df["term_months"] == TERM].reset_index(drop=True)
    return split.split(df), split.summary(df), df


def develop(rf_table, verbose=True):
    """Fit the model, then settle every rule on the internal test block."""
    (tr, va, te), blocks, df_all = load_blocks(rf_table)
    if verbose:
        print(blocks.to_string(index=False))

    art = model.fit(tr, va)
    te = te.copy()
    te["ER_hat"] = model.predict(*art, te)
    corr_honest = float(np.corrcoef(te["ER_hat"], te["ER"])[0, 1])

    # the diagnostic: what a shuffled split would have reported instead
    shuffled = pd.concat([tr, va, te], ignore_index=True)
    art_leak = model.fit(shuffled)
    leak_pred = model.predict(*art_leak, te)
    corr_leak = float(np.corrcoef(leak_pred, te["ER"])[0, 1])
    te["ER_hat_leak"] = leak_pred

    gate = te["ER_hat"].to_numpy(dtype=float) > 0.0
    base_row, base_er = measure(te, rules.w_all(te), rf_table, "전량 투자")

    # --- choice 1 and 2: how far down the grade ladder, with the gate on ---
    rows, sers = [base_row], {"A-G(전량)": base_er}
    for name, keep in GRADE_CUTS.items():
        w = rules.w_mask(te, te["grade"].isin(keep).to_numpy() & gate)
        if w.sum() <= 0:
            continue
        r, er = measure(te, w, rf_table, f"등급 {name} × ER̂>0", base_er)
        rows.append(r); sers[name] = er
    grade_tbl = pd.DataFrame(rows)

    order = [k for k in GRADE_CUTS if k in sers]
    lam = pd.DataFrame([
        {"단계": f"{a} → {b}",
         "수익포기": sers[a].mean() - sers[b].mean(),
         "lambda*": indifference_lambda(sers[a], sers[b])}
        for a, b in zip(order, order[1:])
    ])
    cut = knee_cut(lam, order, KNEE_MULTIPLIER)
    knee_tbl = pd.DataFrame([
        {"배수": m, "선택된 등급컷": knee_cut(lam, order, m)}
        for m in KNEE_SENSITIVITY
    ])

    # --- choice 3: the timing dial, fitted on train+valid outcomes only ---
    hist = pd.concat([tr, va], ignore_index=True)
    hist_vt = metrics.trim_thin_vintages(
        metrics.vintage_table(hist, weights=None, rf_table=rf_table),
        min_pool_share=0.0002)
    hist = hist[hist["issue_month"].isin(hist_vt.index)]
    dial_fit = {"pool": timing.pool_table(hist), "realised": hist_vt["excess"]}

    frozen = {"grade_cut": cut, "grades": GRADE_CUTS[cut], "gate": True,
              "model": art, "model_leaky": art_leak, "dial_fit": dial_fit,
              "corr_honest": corr_honest, "corr_leaky": corr_leak}
    if verbose:
        print(f"\n예측 상관  시간분할(정직) {corr_honest:.4f}   무작위분할(누출) {corr_leak:.4f}")
        print(f"선택된 등급 컷: {cut}")
    return frozen, {"blocks": blocks, "grade": grade_tbl, "lambda": lam,
                    "knee": knee_tbl, "internal_test": te,
                    "trial_sharpes": grade_tbl["Sharpe"].to_numpy(dtype=float)}


# ---------------------------------------------------------------------------
# stage 2 -- report, using the test file once
# ---------------------------------------------------------------------------

def apply_frozen(df, frozen, rf_table):
    """The frozen rule, and the controls that isolate what each part contributed."""
    df = df.copy()
    df["ER_hat"] = model.predict(*frozen["model"], df)
    gate = df["ER_hat"].to_numpy(dtype=float) > 0.0
    in_grade = df["grade"].isin(frozen["grades"]).to_numpy()

    pool = timing.pool_table(df)
    # the reference level comes from the training period, not from the evaluation
    # window's own first months -- see timing.dial_with_history
    fc, dial = timing.dial_with_history(frozen["dial_fit"], pool, TERM)

    rows, sers = [], {}
    base_row, base_er = measure(df, rules.w_all(df), rf_table, "① 전량 투자")
    rows.append(base_row); sers["①전량"] = base_er

    w_grade = rules.w_mask(df, in_grade & gate)
    r2, _er = measure(df, w_grade, rf_table,
                   f"② 등급 {frozen['grade_cut']} × ER̂>0", base_er)
    rows.append(r2); sers["②등급"] = _er

    w_final = timing.weights(df, "ER_hat", dial, in_grade & gate)
    r_fin, er_fin = measure(df, w_final, rf_table, "③ ② + 풀구성 타이밍 (기각)", base_er)
    rows.append(r_fin); sers["③타이밍"] = er_fin

    # same average acceptance as ③, held constant in time: the control that proves
    # any timing gain came from lending less IN BAD YEARS, not from lending less
    flat = pd.Series(r_fin["집행률"] / max(r2["집행률"], 1e-9), index=dial.index)
    r, _er = measure(df, timing.weights(df, "ER_hat", flat, in_grade & gate),
                     rf_table, "④ 동일 평균승인률·시간불변 (타이밍 대조군)", base_er)
    rows.append(r); sers["④시간불변"] = _er

    # matched-rate random control: prices the risk change that comes for free from
    # simply funding fewer loans. Matched to ② -- the rule the report declares
    # final -- not to the rejected ③, so the text's "same acceptance rate" is true
    rand, rand_ers = [], []
    for seed in range(20):
        r_, e_ = measure(df, rules.w_random_at_rate(df, r2["집행률"], seed),
                         rf_table, "⑤ 동일 승인률 무작위", base_er)
        rand.append(r_); rand_ers.append(e_)
    rnd = pd.DataFrame(rand).mean(numeric_only=True).to_dict()
    rnd["전략"] = "⑤ 동일 승인률 무작위 (평균 of 20)"
    rows.append(rnd)
    # the control's series for the paired test is the average across seeds, so the
    # comparison is against the typical random book rather than a lucky draw
    sers["⑤무작위"] = np.mean(np.vstack(rand_ers), axis=0)

    leak = model.predict(*frozen["model_leaky"], df)
    r, _er = measure(df, rules.w_mask(df, in_grade & (leak > 0)), rf_table,
                     "⑥ 무작위분할 모델 (도달불가 상한)", base_er)
    rows.append(r); sers["⑥누출상한"] = _er

    vt = metrics.trim_thin_vintages(
        metrics.vintage_table(df, weights=None, rf_table=rf_table))
    # the series the report plots as "최종 규칙" must be the rule the report
    # declares final -- ② (grade cut x gate), not ③, whose timing dial was rejected
    series = pd.DataFrame({"전량 투자": base_er, "최종 규칙": sers["②등급"],
                           "타이밍 포함 (기각)": er_fin}, index=vt.index)
    return (pd.DataFrame(rows), pd.DataFrame({"예측": fc, "다이얼": dial}),
            series, pool, sers)


RESULTS_DIR = config.ROOT / "results"
MAIN_COLS = ["전략", "집행률", "승인률변동성", "평균초과수익", "변동성", "Sharpe",
             "CI하한", "CI상한", "Sortino", "손실빈티지비율", "최악빈티지"]
STAT_COLS = ["전략", "빈티지수", "유효표본수", "자기상관", "PSR",
             "Δ평균초과수익", "대응HAC_p"]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    fmt = lambda x: f"{x:0.4f}"
    rf_table = load_risk_free_table()

    print("=" * 96)
    print("1단계 — 개발: lending_club_2020_train.csv 만 사용")
    print("=" * 96)
    frozen, dev = develop(rf_table)

    print("\n무릎 규칙 배수 민감도 (사후에 정한 기준이라 함께 공개)")
    print(dev["knee"].to_string(index=False))

    print("\n등급 컷 후보 (내부 test 블록에서 측정)")
    print(dev["grade"][MAIN_COLS].to_string(index=False, float_format=fmt))
    print("\n한 단계 더 조일 때의 무차별 위험회피도")
    print(dev["lambda"].to_string(index=False, float_format=fmt))

    dev["blocks"].to_csv(RESULTS_DIR / "분할_블록.csv", index=False, encoding="utf-8-sig")
    dev["grade"].to_csv(RESULTS_DIR / "개발_등급컷.csv", index=False, encoding="utf-8-sig")
    dev["lambda"].to_csv(RESULTS_DIR / "개발_무차별lambda.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 96)
    print("2단계 — 최종 평가: 규칙 동결 후 lending_club_2020_test.csv 를 한 번만 사용")
    print("=" * 96)
    _, boundary = split.vintage_boundaries(
        load_dataset(config.TRAIN_CSV, rf_table, matured_only=True))
    te = load_dataset(config.TEST_CSV, rf_table, matured_only=True)
    te = te[(te["term_months"] == TERM) & (te["issue_month"] >= boundary)]
    te = te.reset_index(drop=True)
    print(f"평가 대상 {len(te):,}건, "
          f"{te['issue_month'].min():%Y-%m}..{te['issue_month'].max():%Y-%m}, "
          f"빈티지 {te['issue_month'].nunique()}개")

    final, dial, series, pool, sers = apply_frozen(te, frozen, rf_table)

    # the comparisons the conclusion rests on, each with an interval on the Sharpe
    # difference and the independent information the difference series carries
    deltas = delta_tests(sers, [
        ("②등급", "①전량"),      # 최종 규칙 전체의 기여
        ("③타이밍", "①전량"),
        ("②등급", "⑤무작위"),    # 선택의 순수 기여 (승인률 통제)
        ("③타이밍", "④시간불변"),  # 타이밍의 순수 기여 (평균 승인률 통제)
        ("②등급", "⑥누출상한"),  # 정직한 규칙이 상한에서 얼마나 떨어져 있나
    ])
    # the bootstrap's mean block length is itself a choice, and with 26
    # observations a 12-month mean block leaves roughly two independent blocks --
    # so publish how the headline interval moves when the block length moves
    block_rows = []
    for mb in (6, 12, 18):
        d = inference.sharpe_diff_bootstrap(sers["②등급"], sers["①전량"], mean_block=mb)
        lo, hi, _ = inference.sharpe_bootstrap_ci(sers["②등급"], mean_block=mb)
        block_rows.append({"평균블록개월": mb,
                           "ΔSharpe": d["sharpe_diff"],
                           "ΔCI하한": d["ci_low"], "ΔCI상한": d["ci_high"],
                           "Δp": d["p_value"],
                           "②Sharpe_CI하한": lo, "②Sharpe_CI상한": hi})
    block_sens = pd.DataFrame(block_rows)

    dsr = inference.deflated_sharpe_ratio(
        sers["②등급"], n_trials=TRIALS_RUN,
        trial_sharpes=dev["trial_sharpes"],
        n_eff=max(inference.effective_sample_size(sers["②등급"])[0], 2.0))
    print("\n최종 사다리 — 수익과 위험")
    print(final[MAIN_COLS].to_string(index=False, float_format=fmt))
    print("\n최종 사다리 — 통계적 유의성")
    print(final[[c for c in STAT_COLS if c in final.columns]]
          .to_string(index=False, float_format=fmt))
    print("\n차이에 대한 검정 — 결론이 실제로 기대는 비교들")
    print(deltas.to_string(index=False, float_format=fmt))

    print("\n부트스트랩 블록 길이 민감도 (② vs ①, ②의 Sharpe 구간)")
    print(block_sens.to_string(index=False, float_format=fmt))

    print(f"\nDeflated Sharpe (이 프로젝트가 실제로 돌린 시도 {dsr['n_trials']}회 반영)")
    for k, v in dsr.items():
        print(f"  {k}: {v}")

    print("\n무릎 규칙 배수 민감도")
    print(dev["knee"].to_string(index=False))

    print("\n타이밍 다이얼")
    print(dial.dropna().iloc[::3].to_string(float_format=fmt))

    final.to_csv(RESULTS_DIR / "최종_사다리.csv", index=False, encoding="utf-8-sig")
    dial.join(series["전량 투자"].rename("실제_초과수익")).to_csv(
        RESULTS_DIR / "최종_타이밍다이얼.csv", encoding="utf-8-sig")
    deltas.to_csv(RESULTS_DIR / "최종_차이검정.csv", index=False, encoding="utf-8-sig")
    block_sens.to_csv(RESULTS_DIR / "최종_블록민감도.csv", index=False, encoding="utf-8-sig")
    pd.Series(dsr).to_csv(RESULTS_DIR / "최종_DeflatedSharpe.csv", encoding="utf-8-sig")
    dev["knee"].to_csv(RESULTS_DIR / "개발_무릎민감도.csv", index=False, encoding="utf-8-sig")
    series.to_csv(RESULTS_DIR / "최종_빈티지수익계열.csv", encoding="utf-8-sig")
    pool.to_csv(RESULTS_DIR / "최종_신청자풀구성.csv", encoding="utf-8-sig")
    # the pool the dial learned from, so the figure can show the whole span the
    # applicant mix actually moved over rather than just the evaluation window
    frozen["dial_fit"]["pool"].to_csv(
        RESULTS_DIR / "개발_신청자풀구성.csv", encoding="utf-8-sig")
    pd.Series({"corr_시간분할": frozen["corr_honest"],
               "corr_무작위분할": frozen["corr_leaky"],
               "등급컷": frozen["grade_cut"]}).to_csv(
        RESULTS_DIR / "최종_모델요약.csv", encoding="utf-8-sig")
    print(f"\n저장 완료 -> {RESULTS_DIR}")
    return frozen, dev, final, dial


if __name__ == "__main__":
    main()
