"""부록 실험. ④의 +0.047이 모형의 판별력인가, 아니면 단순히 7%를 덜어낸 효과인가.

④ = 등급 A-B × 게이트 × (매달 ER̂ 상위 93.7%만 집행)
비교군 = 등급 A-B × 게이트 × (매달 '무작위' 93.7%만 집행)

둘의 집행률을 맞춘 뒤 Sharpe를 비교한다. 차이가 없으면 +0.047은 모형이 아니라
규모 축소/구성 변화의 부산물이고, 차이가 있으면 모형 순위에 실제 정보가 있다.
"""
import numpy as np, pandas as pd
from src import config, model, pipeline, rules, split, timing
from src.data import load_dataset

TERM = pipeline.TERM
rf_table = pipeline.load_risk_free_table()

print("1단계 — 개발(모형 학습 + 규칙 동결)...", flush=True)
frozen, _ = pipeline.develop(rf_table, verbose=False)

print("2단계 — 평가 파일 적재...", flush=True)
_, boundary = split.vintage_boundaries(
    load_dataset(config.TRAIN_CSV, rf_table, matured_only=True))
te = load_dataset(config.TEST_CSV, rf_table, matured_only=True)
te = te[(te["term_months"] == TERM) & (te["issue_month"] >= boundary)].reset_index(drop=True)

te["ER_hat"] = model.predict(*frozen["model"], te)
gate = te["ER_hat"].to_numpy(dtype=float) > 0.0
in_grade = te["grade"].isin(frozen["grades"]).to_numpy()
base_mask = in_grade & gate

_, base_er = pipeline.measure(te, rules.w_all(te), rf_table, "① 전량")
r2, _ = pipeline.measure(te, rules.w_mask(te, base_mask), rf_table, "②", base_er)
print(f"\n② 등급 A-B × 게이트      집행률 {r2['집행률']*100:.2f}%  Sharpe {r2['Sharpe']:.4f}")

QUOTA = 0.9369  # pipeline이 ④에 쓰는 상수와 동일
flat = pd.Series(QUOTA, index=sorted(te["issue_month"].unique()))

# ④ 재현: ER̂ 순위로 상위 93.7%
w4 = timing.weights(te, "ER_hat", flat, base_mask)
r4, _ = pipeline.measure(te, w4, rf_table, "④", base_er)
print(f"④ ER̂ 순위로 상위 {QUOTA*100:.1f}%   집행률 {r4['집행률']*100:.2f}%  Sharpe {r4['Sharpe']:.4f}")

# 비교군: 동일 규칙이되 순위를 무작위로
funded = te["funded_amnt"].to_numpy(dtype=float)
rows = []
for seed in range(20):
    rng = np.random.default_rng(seed)
    te["_rand"] = rng.random(len(te))
    w = timing.weights(te, "_rand", flat, base_mask)
    r, _ = pipeline.measure(te, w, rf_table, f"무작위 trim seed{seed}", base_er)
    rows.append({"seed": seed, "집행률": r["집행률"], "Sharpe": r["Sharpe"],
                 "평균초과수익": r["평균초과수익"]})
rnd = pd.DataFrame(rows)

print(f"\n무작위 상위 {QUOTA*100:.1f}% (시드 20개)")
print(f"  집행률 평균 {rnd['집행률'].mean()*100:.2f}%")
print(f"  Sharpe  평균 {rnd['Sharpe'].mean():.4f}  (최소 {rnd['Sharpe'].min():.4f} ~ 최대 {rnd['Sharpe'].max():.4f}, 표준편차 {rnd['Sharpe'].std():.4f})")

print("\n" + "=" * 70)
print(f"② -> ④ (ER̂ 순위 솎아내기) : {r4['Sharpe']-r2['Sharpe']:+.4f}")
print(f"② -> 무작위 솎아내기        : {rnd['Sharpe'].mean()-r2['Sharpe']:+.4f}")
print(f"모형 순위의 순수 기여       : {r4['Sharpe']-rnd['Sharpe'].mean():+.4f}")
z = (r4["Sharpe"] - rnd["Sharpe"].mean()) / rnd["Sharpe"].std()
print(f"  (무작위 시드 분포 대비 {z:+.2f} 표준편차)")
print("=" * 70)

rnd.to_csv("결과/부록_무작위솎아내기_대조군.csv", index=False, encoding="utf-8-sig")
print("\n저장: 결과/부록_무작위솎아내기_대조군.csv")
