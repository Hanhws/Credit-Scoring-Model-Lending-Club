"""Statistical inference for cohort-level Sharpe ratios.

Everything here exists because the cohort ER series is NOT an i.i.d. sample.
Adjacent origination months share the same macro regime, and 3-5y loans issued
in different months have overlapping repayment windows, so the lag-1
autocorrelation of the cohort ER series runs around 0.78. Treating the ~128
cohorts as 128 independent draws understates the standard error by roughly a
factor of 3 (effective sample size is closer to 16), which is the difference
between "beats the Grade A-C baseline" and "cannot be distinguished from it".

Four tools, deliberately kept separate:

1. Newey-West (HAC) standard errors for the mean of a series -- used for the
   PAIRED test (strategy ER minus baseline ER, cohort by cohort). Pairing is
   what makes this test powerful: both strategies face the same macro shocks in
   the same month, so differencing removes the common cycle and what remains is
   the strategy effect.

2. Stationary bootstrap (Politis & Romano 1994) for confidence intervals on the
   Sharpe ratio itself. Sharpe is a nonlinear function of two moments, so its
   sampling distribution is not well approximated by a delta-method normal at
   this sample size; resampling whole blocks preserves the autocorrelation
   structure that makes the naive interval too narrow.

3. Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012): P(true SR > a
   benchmark SR), given the sample's skew and kurtosis. Loan returns are
   left-skewed and fat-tailed, so the normal-theory t-test on the Sharpe is
   optimistic; PSR puts the third and fourth moments into the standard error.

4. Deflated Sharpe Ratio: PSR benchmarked against the Sharpe you would EXPECT
   to see as the maximum over N independent trials of a strategy with zero true
   skill. Every K value swept, every model family compared, and every feature
   set tried is a trial, and reporting the best of them is a selection-biased
   estimate. DSR is the number that survives that correction -- without it, a
   headline Sharpe picked as the max of a sweep means very little.
"""
import numpy as np
from scipy import stats


def newey_west_lags(n):
    """Standard automatic bandwidth, floor(4*(n/100)^(2/9)) (Newey-West 1994)."""
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_se(x, lags=None):
    """HAC standard error of the sample mean of x."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if lags is None:
        lags = newey_west_lags(n)
    e = x - x.mean()
    s = float(e @ e) / n
    for j in range(1, lags + 1):
        gamma = float(e[j:] @ e[:-j]) / n
        s += 2.0 * (1.0 - j / (lags + 1.0)) * gamma
    s = max(s, 1e-18)  # Bartlett kernel guarantees non-negativity in theory only
    return np.sqrt(s / n), lags


def effective_sample_size(x):
    """AR(1)-based effective n: n*(1-r1)/(1+r1). A blunt instrument, but it is
    the number that makes the autocorrelation problem legible at a glance."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    r1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    return n * (1 - r1) / (1 + r1), r1


def sharpe(er):
    er = np.asarray(er, dtype=float)
    sd = er.std(ddof=1)
    return np.nan if sd == 0 else er.mean() / sd


def weighted_sharpe(er, weights):
    """Sharpe where each cohort is weighted by dollars invested rather than
    counted once. The equal-weighted version lets a $50k month from 2007 carry
    the same influence as a $269M month from 2015 -- a 5,400x gap in the data."""
    er = np.asarray(er, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    mean = float(w @ er)
    # unbiased variance for reliability weights
    denom = 1.0 - float(w @ w)
    if denom <= 0:
        return np.nan
    var = float(w @ (er - mean) ** 2) / denom
    return np.nan if var <= 0 else mean / np.sqrt(var)


def _stationary_bootstrap_idx(n, n_boot, mean_block, rng):
    """Index matrix (n_boot, n) of stationary-bootstrap resamples: geometric
    block lengths with mean `mean_block`, wrapping at the end of the series."""
    p = 1.0 / mean_block
    jump = rng.random((n_boot, n)) < p
    jump[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    t = np.broadcast_to(np.arange(n), (n_boot, n))
    block_start = np.maximum.accumulate(np.where(jump, t, 0), axis=1)
    base = np.take_along_axis(starts, block_start, axis=1)
    return (base + (t - block_start)) % n


def sharpe_bootstrap_ci(er, n_boot=5000, mean_block=12, seed=42, alpha=0.05):
    """Percentile CI for the Sharpe ratio under the stationary bootstrap.
    mean_block defaults to 12 cohorts (~one year) so that a resample keeps a
    full seasonal/macro block intact rather than shuffling months apart."""
    er = np.asarray(er, dtype=float)
    n = len(er)
    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_idx(n, n_boot, mean_block, rng)
    samples = er[idx]
    means = samples.mean(axis=1)
    sds = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sds > 0, means / sds, np.nan)
    sr = sr[np.isfinite(sr)]
    lo, hi = np.percentile(sr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(np.std(sr, ddof=1))


def paired_mean_test(er_strategy, er_baseline, lags=None):
    """Paired HAC t-test on the per-cohort excess-return difference.

    H0: the strategy's mean cohort ER equals the baseline's. Because the two
    series are observed on the SAME cohorts, differencing first removes the
    macro cycle they share -- without pairing, the common cycle dominates both
    variances and the test has almost no power.
    """
    d = np.asarray(er_strategy, dtype=float) - np.asarray(er_baseline, dtype=float)
    n = len(d)
    se, used_lags = newey_west_se(d, lags)
    t = d.mean() / se
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
    return {
        "mean_diff": float(d.mean()),
        "hac_se": float(se),
        "t_stat": float(t),
        "p_value": float(p),
        "lags": used_lags,
        "n": n,
    }


def sharpe_diff_bootstrap(er_a, er_b, n_boot=5000, mean_block=12, seed=42, alpha=0.05):
    """Bootstrap CI and two-sided p-value for Sharpe(a) - Sharpe(b), resampling
    both series with the SAME block indices so the pairing across cohorts is
    preserved."""
    a = np.asarray(er_a, dtype=float)
    b = np.asarray(er_b, dtype=float)
    n = len(a)
    rng = np.random.default_rng(seed)
    idx = _stationary_bootstrap_idx(n, n_boot, mean_block, rng)

    def _sr(mat):
        m = mat.mean(axis=1)
        s = mat.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(s > 0, m / s, np.nan)

    diff = _sr(a[idx]) - _sr(b[idx])
    diff = diff[np.isfinite(diff)]
    observed = sharpe(a) - sharpe(b)
    lo, hi = np.percentile(diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # p-value by inverting the percentile interval around the null of zero
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    return {
        "sharpe_diff": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
    }


# ---------------------------------------------------------------------------
# Multiple-testing / non-normality corrections
# ---------------------------------------------------------------------------

EULER_MASCHERONI = 0.5772156649015329


def lo_sharpe_se(er, adjust_autocorr=True):
    """Lo (2002) asymptotic standard error of the Sharpe ratio,
    SE = sqrt((1 + SR^2 / 2) / n), under i.i.d. normal returns.

    With adjust_autocorr, n is replaced by the AR(1) effective sample size.
    The overlapping-maturity problem means the raw n is a fiction: 3-5 year
    loans issued in adjacent months repay into the same calendar quarters, so
    consecutive observations carry largely the same macro shock. Using raw n
    here would report an interval roughly 3x too narrow.
    """
    er = np.asarray(er, dtype=float)
    n = float(len(er))
    sr = sharpe(er)
    if adjust_autocorr:
        n_eff, _ = effective_sample_size(er)
        n = max(n_eff, 2.0)
    return float(np.sqrt((1.0 + 0.5 * sr ** 2) / n)), float(n)


def moments(er):
    """Sample Sharpe plus the higher moments PSR/DSR need. Kurtosis is
    NON-excess (normal == 3), which is the convention in the Bailey &
    Lopez de Prado formulas."""
    er = np.asarray(er, dtype=float)
    return {
        "sharpe": sharpe(er),
        "skew": float(stats.skew(er, bias=False)),
        "kurtosis": float(stats.kurtosis(er, fisher=False, bias=False)),
        "n": len(er),
    }


def probabilistic_sharpe_ratio(er, sr_benchmark=0.0, n_eff=None):
    """P(true SR > sr_benchmark), correcting for skew and kurtosis.

    The denominator is the standard error of the Sharpe estimator under
    non-normal returns. Negative skew (a long left tail -- exactly what a loan
    book has, since the upside is capped at the coupon and the downside is the
    whole principal) INFLATES that standard error, so a left-skewed strategy
    needs a higher sample Sharpe to reach the same confidence.

    n_eff overrides the sample size, so the autocorrelation haircut from
    effective_sample_size can be carried through instead of pretending the
    cohort/month series is i.i.d.
    """
    m = moments(er)
    sr, skew, kurt = m["sharpe"], m["skew"], m["kurtosis"]
    n = float(n_eff if n_eff is not None else m["n"])
    var = 1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr ** 2
    if var <= 0 or n <= 1:
        return np.nan
    z = (sr - sr_benchmark) * np.sqrt(n - 1.0) / np.sqrt(var)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials, sr_variance):
    """E[max SR] over n_trials independent strategies whose true SR is zero and
    whose estimated SRs have variance sr_variance -- the false-discovery
    yardstick from Bailey & Lopez de Prado (2014).

    This is the number a sweep produces from pure noise. Sweeping K over 20
    values and 5 model families is 100 trials; the max of 100 zero-skill draws
    is roughly 2.5 standard deviations up, so a Sharpe below that is not
    evidence of anything.
    """
    n_trials = max(int(n_trials), 1)
    if n_trials == 1:
        return 0.0
    g = EULER_MASCHERONI
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe_ratio(er, n_trials, sr_variance=None, trial_sharpes=None, n_eff=None):
    """PSR against the expected-maximum-of-noise benchmark.

    sr_variance should be the variance of the Sharpe ratios ACROSS the trials
    actually run (pass trial_sharpes and it is computed for you). That is the
    honest input: a sweep whose Sharpes barely move across K has little room
    for selection bias, while one that swings wildly has a lot.
    """
    if sr_variance is None:
        if trial_sharpes is None:
            raise ValueError("pass sr_variance or trial_sharpes")
        sr_variance = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1))
    sr_star = expected_max_sharpe(n_trials, sr_variance)
    return {
        "n_trials": int(n_trials),
        "sr_variance_across_trials": float(sr_variance),
        "expected_max_sharpe_under_null": sr_star,
        "observed_sharpe": sharpe(er),
        "dsr": probabilistic_sharpe_ratio(er, sr_benchmark=sr_star, n_eff=n_eff),
    }


def annualize_sharpe(sr_periodic, periods_per_year=12):
    """sqrt-of-time scaling. Valid only for a serially UNCORRELATED periodic
    series -- the monthly book return is autocorrelated (amortising loans hand
    back a smooth coupon stream), so the annualised figure is an upper bound and
    should always be reported next to the bootstrap interval, never alone."""
    return float(sr_periodic * np.sqrt(periods_per_year))


def desmooth(returns, rho=None):
    """Geltner (1991) first-order unsmoothing: r*_t = (r_t - rho*r_{t-1}) / (1 - rho).

    A book of amortising loans held at amortised cost has a mechanically smooth
    monthly return: the coupon arrives on schedule, and a charge-off is
    recognised on an accounting trigger (120 days past due) rather than when the
    market's view of the borrower changed. Nothing marks to market, so the
    monthly series has an autocorrelation near 0.9 and a volatility far below
    the economic risk being carried. Reporting a Sharpe off that series is the
    "volatility laundering" private credit is routinely accused of, and it is
    the single easiest way for this analysis to flatter itself.

    The standard correction, from the real-estate/appraisal-lag literature,
    treats the observed return as a moving average of the true one and inverts
    it. The mean is preserved and the variance is inflated by roughly
    (1+rho)/(1-rho), so the desmoothed Sharpe is the conservative figure and the
    raw one is an upper bound. Report both; never report only the raw one.
    """
    r = np.asarray(returns, dtype=float)
    if rho is None:
        rho = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    rho = float(np.clip(rho, 0.0, 0.95))
    if rho == 0:
        return r.copy(), 0.0
    out = (r[1:] - rho * r[:-1]) / (1.0 - rho)
    return out, rho
