"""The one model. XGBoost regression on cumulative excess return.

This replaces the five separate training scripts that used to exist (LightGBM,
XGBoost, CatBoost, RandomForest, Linear, each with its own K sweep). Comparing
model families was worth doing once and it did not change any conclusion: the
gap between families was far smaller than the gap between an honest and a
dishonest evaluation split, which is what this file is organised around instead.

Two fits, deliberately
----------------------
random    trained on a random subset of loans spanning 2007-2018 and scored on
          other loans from the same years. Loan ids never overlap, so it looks
          out-of-sample, but the model has seen the 2016 credit environment
          before it is asked to price a 2016 loan. Reported ONLY as the
          optimistic bound, to show the size of the gap.

time      trained on loans issued before 2015 and scored on 2015+ vintages.
          This is what a lender could actually have deployed in January 2015.
          Every headline number comes from this fit.

The gap between the two is the cost of not knowing the future, and on this
dataset it is the single largest term in the whole analysis -- larger than model
choice, feature set, or selection rule.

Features
--------
config.SAFE_FEATURE_COLS, not the full list. Around 50 of LC's credit-bureau
fields are 100% missing before 2012 and ~0% missing after 2016, because LC only
began collecting them in 2015-16. A tree can read "is this field NaN" as a
near-perfect clock, which smuggles calendar time -- and the era's base default
rate -- into a model that was never given a date column. The safe subset keeps
only features whose missingness moves less than 30pp between the 2007-2010 and
2016-2018 vintages.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from . import config
from .features import feature_matrix

TIME_CUTOFF = pd.Timestamp("2015-01-01")

PARAMS = dict(
    n_estimators=3000, learning_rate=0.03, max_depth=7,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    tree_method="hist", enable_categorical=True,
    eval_metric="mae", early_stopping_rounds=100,
    random_state=config.RANDOM_STATE, n_jobs=-1, verbosity=0,
)


def fit(df_fit, df_valid=None, safe_only=True):
    """Fit on df_fit, using df_valid to decide when to stop adding trees.

    Boosting needs a series it is not being fitted on to choose the number of
    rounds; if that series comes out of the training block, the stopping point is
    tuned on data the model has already seen and the model runs longer than it
    should. `df_valid` is the dedicated validation block from split.py, which is
    LATER in calendar time than the training block, so the stopping rule is also
    only ever informed by the past.

    When df_valid is omitted the early-stopping set is carved randomly out of
    df_fit. That path exists only for the diagnostic that measures what a
    shuffled split does to the numbers -- it is never the production fit.
    """
    if df_valid is not None and len(df_valid):
        df_a, df_b = df_fit, df_valid
    else:
        idx_tr, idx_es = train_test_split(
            df_fit.index, test_size=0.15, random_state=config.RANDOM_STATE
        )
        df_a, df_b = df_fit.loc[idx_tr], df_fit.loc[idx_es]

    X_tr = feature_matrix(df_a, safe_only=safe_only)
    X_es = feature_matrix(df_b, safe_only=safe_only)[X_tr.columns]
    cat_cols = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]
    for c in cat_cols:
        X_es[c] = X_es[c].astype("category").cat.set_categories(X_tr[c].cat.categories)

    model = xgb.XGBRegressor(**PARAMS)
    model.fit(X_tr, df_a["ER"], eval_set=[(X_es, df_b["ER"])], verbose=False)
    return model, list(X_tr.columns), {c: X_tr[c].cat.categories for c in cat_cols}


def predict(model, cols, cats, df, safe_only=True):
    X = feature_matrix(df, safe_only=safe_only)[cols]
    for c, categories in cats.items():
        X[c] = X[c].astype("category").cat.set_categories(categories)
    return np.asarray(model.predict(X), dtype=float)


def fit_predict(df_train, df_score, split="time", safe_only=True):
    """split='time'   -> train on pre-2015 originations only (honest)
       split='random' -> train on everything (era-leaking, optimistic bound)"""
    if split == "time":
        df_fit = df_train[df_train["issue_month"] < TIME_CUTOFF]
    elif split == "random":
        df_fit = df_train
    else:
        raise ValueError(f"unknown split {split!r}")
    model, cols, cats = fit(df_fit, safe_only=safe_only)
    preds = predict(model, cols, cats, df_score, safe_only=safe_only)
    return preds, {
        "split": split,
        "n_fit": len(df_fit),
        "fit_months": f"{df_fit['issue_month'].min():%Y-%m}..{df_fit['issue_month'].max():%Y-%m}",
        "corr": float(np.corrcoef(preds, df_score["ER"])[0, 1]),
    }


def fit_predict_walkforward(df_train, df_score, start_year, end_year=None,
                            safe_only=True, verbose=True):
    """Refit once a year on everything whose outcome was KNOWN by that January,
    and score that year's originations.

    Two things separate this from `fit_predict(split='time')`.

    The first is that the evaluation window stops being a single block. One fit
    scored on 2015-2018 produces 37 vintages of a monotonically deteriorating
    credit box, which is one observation of the credit cycle no matter how many
    loans it contains. Refitting annually lets the evaluation start years
    earlier, and the governing quantity -- the effective sample size, roughly
    N(1-rho)/(1+rho) -- responds to a window that CHANGES DIRECTION far more than
    to one that is merely longer.

    The second is a leak that the single time split does not close. Fitting on
    "originations before 2015" quietly assumes a lender in January 2015 knew what
    its 2013 loans would eventually return. It did not: a 36-month loan written
    in 2013 is still paying. The target here is realised lifetime excess return,
    so a loan is only usable for training once it has MATURED, and `matures_at <=
    asof` is the honest cut. It is an expensive one -- the January 2013 fit gets
    5,396 loans where the old split would have handed it 280,000 -- and that cost
    is itself a finding about how little a lender in this market actually knew.
    """
    end_year = end_year or int(df_score["issue_month"].max().year)
    preds = np.full(len(df_score), np.nan)
    issue = df_score["issue_month"].to_numpy()
    info = []

    for year in range(start_year, end_year + 1):
        asof = pd.Timestamp(f"{year}-01-01")
        nxt = pd.Timestamp(f"{year + 1}-01-01")
        target = (issue >= asof) & (issue < nxt)
        if not target.any():
            continue
        df_fit = df_train[df_train["matures_at"] <= asof]
        if len(df_fit) < 500:
            if verbose:
                print(f"  {year}: only {len(df_fit)} matured loans to learn from -- skipped")
            continue
        m, cols, cats = fit(df_fit, safe_only=safe_only)
        preds[target] = predict(m, cols, cats, df_score[target], safe_only=safe_only)
        rec = {"year": year, "n_fit": len(df_fit), "n_scored": int(target.sum()),
               "fit_through": f"{df_fit['issue_month'].max():%Y-%m}",
               "corr": float(np.corrcoef(preds[target],
                                         df_score.loc[target, "ER"])[0, 1])}
        info.append(rec)
        if verbose:
            print(f"  {year}: fit {rec['n_fit']:,} matured (through {rec['fit_through']})"
                  f" -> scored {rec['n_scored']:,}, corr {rec['corr']:.3f}")
    return preds, pd.DataFrame(info)


def evaluation_universe(df_score, split="time"):
    """The vintages a given split is allowed to be scored on. A time-split model
    may only be judged on originations it could not have seen."""
    if split == "time":
        return df_score[df_score["issue_month"] >= TIME_CUTOFF]
    return df_score
