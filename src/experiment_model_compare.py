"""부록 실험. 알고리즘 선택이 결론을 바꾸는가.

우수사례들은 여러 알고리즘을 나열해 최고 성능 하나를 고른다. 본 연구는 XGBoost
하나만 썼으므로, 그 선택이 결과를 좌우했는지 확인해 둘 필요가 있다.

규율:
  - 전부 **개발용 파일 안에서만** 수행한다. 평가 파일은 열지 않는다.
  - 학습은 train 블록, 조기중단은 valid 블록, 측정은 quiz 블록.
    (본편의 XGBoost와 완전히 동일한 조건)
  - 지표는 MSE가 아니라 본 보고서가 실제로 쓰는 것 — 예측 상관과 Sharpe.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge

from src import config, metrics, model, pipeline, rules
from src.features import feature_matrix

rf_table = pipeline.load_risk_free_table()
(tr, va, te), blocks, _ = pipeline.load_blocks(rf_table)
print(blocks.to_string(index=False), "\n")

X_tr = feature_matrix(tr, safe_only=True)
cols = list(X_tr.columns)
cat_cols = [c for c in cols if str(X_tr[c].dtype) == "category"]


def _align(df):
    X = feature_matrix(df, safe_only=True)[cols]
    for c in cat_cols:
        X[c] = X[c].astype("category").cat.set_categories(X_tr[c].cat.categories)
    return X


X_va, X_te = _align(va), _align(te)
num = lambda X: X.drop(columns=cat_cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)


def fit_xgb():
    m, c, cats = model.fit(tr, va)
    return model.predict(m, c, cats, te)


def fit_lgbm():
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=63,
                          random_state=config.RANDOM_STATE, verbose=-1)
    m.fit(X_tr, tr["ER"], eval_set=[(X_va, va["ER"])],
          callbacks=[lgb.early_stopping(50, verbose=False)],
          categorical_feature=cat_cols)
    return np.asarray(m.predict(X_te), dtype=float)


def fit_cat():
    from catboost import CatBoostRegressor, Pool
    f = lambda X: X.assign(**{c: X[c].astype(str) for c in cat_cols})
    m = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=6,
                          random_seed=config.RANDOM_STATE, verbose=False)
    m.fit(Pool(f(X_tr), tr["ER"], cat_features=cat_cols),
          eval_set=Pool(f(X_va), va["ER"], cat_features=cat_cols),
          early_stopping_rounds=50, verbose=False)
    return np.asarray(m.predict(f(X_te)), dtype=float)


def fit_ridge():
    A, B = num(X_tr), num(X_te)
    mu, sd = A.mean(), A.std().replace(0, 1.0)
    m = Ridge(alpha=1.0).fit((A - mu) / sd, tr["ER"])
    return np.asarray(m.predict((B - mu) / sd), dtype=float)


CANDIDATES = {"XGBoost (본편)": fit_xgb, "LightGBM": fit_lgbm,
              "CatBoost": fit_cat, "Ridge 선형": fit_ridge}

_, base_er = pipeline.measure(te, rules.w_all(te), rf_table, "전량")
QUOTA = 0.9369
rows = []
for name, fn in CANDIDATES.items():
    print(f"학습: {name} ...", flush=True)
    te = te.copy()
    te["ER_hat"] = fn()
    corr = float(np.corrcoef(te["ER_hat"], te["ER"])[0, 1])

    gate = te["ER_hat"].to_numpy(dtype=float) > 0.0
    in_grade = te["grade"].isin(["A", "B"]).to_numpy()
    r_gate, _ = pipeline.measure(te, rules.w_mask(te, in_grade & gate), rf_table, name, base_er)

    from src import timing
    flat = pd.Series(QUOTA, index=sorted(te["issue_month"].unique()))
    r_trim, _ = pipeline.measure(te, timing.weights(te, "ER_hat", flat, in_grade & gate),
                                 rf_table, name, base_er)
    rows.append({"모델": name, "예측상관": corr, "게이트통과율": float(gate.mean()),
                 "Sharpe_등급×게이트": r_gate["Sharpe"], "집행률_게이트": r_gate["집행률"],
                 "Sharpe_순위컷": r_trim["Sharpe"], "집행률_순위컷": r_trim["집행률"]})

out = pd.DataFrame(rows)
print("\n" + "=" * 96)
print("quiz 블록(개발 데이터)에서 측정 — 평가 파일은 열지 않음")
print("=" * 96)
print(out.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))
print("\n범위: 예측상관 {:.4f}~{:.4f} · 등급×게이트 Sharpe {:.4f}~{:.4f} · 순위컷 Sharpe {:.4f}~{:.4f}".format(
    out["예측상관"].min(), out["예측상관"].max(),
    out["Sharpe_등급×게이트"].min(), out["Sharpe_등급×게이트"].max(),
    out["Sharpe_순위컷"].min(), out["Sharpe_순위컷"].max()))
out.to_csv("results/부록_모델비교.csv", index=False, encoding="utf-8-sig")
print("\n저장: results/부록_모델비교.csv")
