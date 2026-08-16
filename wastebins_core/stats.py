"""
Statistical reporting for the experiment suite.
===============================================

Reviewer 2 asked for confidence intervals, significance testing and sensitivity
analysis, none of which the first submission provided.  This module supplies
them in the form appropriate to the data, which is *paired* and decidedly
non-normal: every policy is evaluated on the same set of scenario snapshots, so
comparisons are within-snapshot and the differences are skewed and heavy-tailed.

Consequently the defaults here are:

* **BCa bootstrap intervals** rather than normal-theory intervals.  The
  bias-corrected and accelerated interval (Efron 1987) is second-order accurate
  and does not assume symmetry, which matters for ratios such as "percentage
  distance saved".
* **Wilcoxon signed-rank** rather than a paired t-test, with an exact
  permutation fallback for small samples.
* **Cliff's delta** and **paired Cohen's d** for effect size, because a p-value
  on 25 snapshots says whether an effect exists, not whether it matters.
* **Holm-Bonferroni** correction, since comparing one proposed method against
  five baselines across several metrics is a multiple-comparison problem.

Everything returns plain dictionaries so results serialise straight into the
results JSON that the figures and the manuscript tables are built from.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                    # SciPy is available but not required
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except Exception:                       # pragma: no cover
    _scipy_stats = None
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Point estimates and intervals
# ---------------------------------------------------------------------------
def _percentile_of_score(samples: np.ndarray, value: float) -> float:
    return float(np.mean(samples < value))


def bootstrap_ci(data: Sequence[float], statistic: Callable[[np.ndarray], float] = np.mean,
                 confidence: float = 0.95, n_resamples: int = 10_000,
                 seed: int = 42, method: str = "bca") -> Dict:
    """
    Bootstrap confidence interval for a statistic of one sample.

    ``method`` is ``"bca"`` (default), ``"percentile"`` or ``"basic"``.  BCa
    corrects for both bias and skewness in the bootstrap distribution; it falls
    back to the percentile interval when the acceleration term is undefined
    (which happens when every jackknife replicate is identical).
    """
    x = np.asarray(list(data), dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return {"estimate": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0,
                "method": method, "confidence": confidence}
    if n == 1:
        v = float(statistic(x))
        return {"estimate": v, "ci_low": v, "ci_high": v, "n": 1,
                "method": "degenerate", "confidence": confidence}

    rng = np.random.default_rng(seed)
    theta_hat = float(statistic(x))
    idx = rng.integers(0, n, size=(int(n_resamples), n))
    replicates = np.array([statistic(x[i]) for i in idx], dtype=float)
    replicates = replicates[np.isfinite(replicates)]
    if replicates.size == 0:
        return {"estimate": theta_hat, "ci_low": theta_hat, "ci_high": theta_hat,
                "n": n, "method": "degenerate", "confidence": confidence}

    alpha = 1.0 - confidence
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    used = method

    if method == "bca":
        prop = _percentile_of_score(replicates, theta_hat)
        if 0.0 < prop < 1.0:
            z0 = _norm_ppf(prop)
            # jackknife acceleration
            jack = np.array([statistic(np.delete(x, i)) for i in range(n)], dtype=float)
            jack_mean = jack.mean()
            num = float(np.sum((jack_mean - jack) ** 3))
            den = 6.0 * (float(np.sum((jack_mean - jack) ** 2)) ** 1.5)
            a = num / den if abs(den) > 1e-12 else 0.0
            z_lo, z_hi = _norm_ppf(lo_q), _norm_ppf(hi_q)
            lo_q = _norm_cdf(z0 + (z0 + z_lo) / max(1e-12, (1 - a * (z0 + z_lo))))
            hi_q = _norm_cdf(z0 + (z0 + z_hi) / max(1e-12, (1 - a * (z0 + z_hi))))
            lo_q = min(max(lo_q, 1e-6), 1 - 1e-6)
            hi_q = min(max(hi_q, 1e-6), 1 - 1e-6)
            if hi_q <= lo_q:
                lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
                used = "percentile(bca-degenerate)"
        else:
            used = "percentile(bca-degenerate)"

    ci_low = float(np.quantile(replicates, lo_q))
    ci_high = float(np.quantile(replicates, hi_q))
    if method == "basic":
        ci_low, ci_high = 2 * theta_hat - ci_high, 2 * theta_hat - ci_low

    return {
        "estimate": round(theta_hat, 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
        "n": int(n),
        "method": used,
        "confidence": confidence,
        "n_resamples": int(n_resamples),
        "std_error": round(float(np.std(replicates, ddof=1)), 6),
    }


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation; ~1e-9 accurate)."""
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------
def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> Dict:
    """
    Cliff's delta: P(a > b) - P(a < b).

    A non-parametric effect size that needs no distributional assumption and is
    interpretable directly as dominance.  Magnitude thresholds follow Romano et
    al. (2006).
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.size == 0 or y.size == 0:
        return {"delta": 0.0, "magnitude": "none"}
    diff = np.sign(x[:, None] - y[None, :])
    delta = float(diff.mean())
    magnitude = ("negligible" if abs(delta) < 0.147 else
                 "small" if abs(delta) < 0.33 else
                 "medium" if abs(delta) < 0.474 else "large")
    return {"delta": round(delta, 5), "magnitude": magnitude}


def paired_cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    d = np.asarray(list(a), dtype=float) - np.asarray(list(b), dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return 0.0
    sd = float(np.std(d, ddof=1))
    return float(np.mean(d) / sd) if sd > 1e-12 else 0.0


def _exact_sign_permutation_p(differences: np.ndarray) -> float:
    """Exact two-sided p-value by enumerating sign flips (small samples only)."""
    d = differences[np.isfinite(differences) & (np.abs(differences) > 1e-15)]
    n = d.size
    if n == 0:
        return 1.0
    observed = abs(float(d.sum()))
    count = 0
    total = 1 << n
    for mask in range(total):
        signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
        if abs(float(np.dot(signs, np.abs(d)))) >= observed - 1e-12:
            count += 1
    return min(1.0, count / total)


def paired_test(a: Sequence[float], b: Sequence[float],
                alternative: str = "two-sided") -> Dict:
    """
    Wilcoxon signed-rank test on paired observations.

    Falls back to an exact sign-permutation test when the sample is too small
    for the signed-rank normal approximation to be trusted, and reports which
    test was actually used so the result is not silently over-claimed.
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    d = x - y
    finite = np.isfinite(d)
    d = d[finite]

    if d.size == 0 or np.allclose(d, 0.0):
        return {"test": "none", "p_value": 1.0, "n_pairs": int(d.size),
                "statistic": 0.0, "note": "all differences are zero"}

    if _HAVE_SCIPY and d.size >= 6:
        try:
            result = _scipy_stats.wilcoxon(x[finite], y[finite],
                                           alternative=alternative,
                                           zero_method="wilcox")
            return {"test": "wilcoxon_signed_rank",
                    "statistic": round(float(result.statistic), 5),
                    "p_value": float(result.pvalue),
                    "n_pairs": int(d.size),
                    "alternative": alternative}
        except Exception:
            pass

    if d.size <= 20:
        return {"test": "exact_sign_permutation",
                "statistic": round(float(d.sum()), 5),
                "p_value": _exact_sign_permutation_p(d),
                "n_pairs": int(d.size),
                "alternative": "two-sided"}

    # Normal approximation to the signed-rank statistic.
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    w_plus = float(np.sum(ranks[d > 0]))
    nn = d.size
    mean_w = nn * (nn + 1) / 4.0
    sd_w = math.sqrt(nn * (nn + 1) * (2 * nn + 1) / 24.0)
    z = (w_plus - mean_w) / sd_w if sd_w > 1e-12 else 0.0
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return {"test": "wilcoxon_normal_approx", "statistic": round(w_plus, 5),
            "p_value": float(min(1.0, p)), "n_pairs": int(nn), "z": round(z, 4)}


def compare_paired(proposed: Sequence[float], baseline: Sequence[float],
                   label: str = "", lower_is_better: bool = True,
                   confidence: float = 0.95, seed: int = 42,
                   n_resamples: int = 10_000) -> Dict:
    """
    Full paired comparison: difference, relative improvement with a bootstrap
    interval, significance test, and two effect sizes.

    The relative improvement is bootstrapped *as a ratio of means* rather than
    as the mean of per-pair ratios, because per-pair ratios explode whenever a
    baseline value approaches zero.
    """
    p = np.asarray(list(proposed), dtype=float)
    b = np.asarray(list(baseline), dtype=float)
    n = min(p.size, b.size)
    p, b = p[:n], b[:n]

    differences = b - p if lower_is_better else p - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_resamples), n)) if n > 0 else None

    if idx is not None and n > 0:
        num = (b[idx].mean(axis=1) - p[idx].mean(axis=1)) if lower_is_better else \
              (p[idx].mean(axis=1) - b[idx].mean(axis=1))
        den = b[idx].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = 100.0 * num / den
        ratio = ratio[np.isfinite(ratio)]
        alpha = 1 - confidence
        rel_lo = float(np.quantile(ratio, alpha / 2)) if ratio.size else 0.0
        rel_hi = float(np.quantile(ratio, 1 - alpha / 2)) if ratio.size else 0.0
    else:
        rel_lo = rel_hi = 0.0

    mean_b = float(np.mean(b)) if b.size else 0.0
    mean_p = float(np.mean(p)) if p.size else 0.0
    rel = (100.0 * (mean_b - mean_p) / mean_b) if (lower_is_better and abs(mean_b) > 1e-12) \
        else ((100.0 * (mean_p - mean_b) / mean_b) if abs(mean_b) > 1e-12 else 0.0)

    return {
        "label": label,
        "n_pairs": int(n),
        "mean_proposed": round(mean_p, 5),
        "mean_baseline": round(mean_b, 5),
        "mean_difference": bootstrap_ci(differences, confidence=confidence,
                                        n_resamples=n_resamples, seed=seed),
        "relative_improvement_pct": round(rel, 4),
        "relative_improvement_ci": [round(rel_lo, 4), round(rel_hi, 4)],
        "test": paired_test(p, b),
        "cliffs_delta": cliffs_delta(p, b),
        "paired_cohens_d": round(paired_cohens_d(p, b), 5),
        "lower_is_better": lower_is_better,
    }


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------
def holm_bonferroni(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict]:
    """
    Holm-Bonferroni step-down correction.

    Uniformly more powerful than plain Bonferroni at the same family-wise error
    rate, and the right default when one proposed method is compared against
    several baselines on the same data.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Dict] = {}
    max_so_far = 0.0
    for rank, (name, p) in enumerate(items):
        adjusted = min(1.0, (m - rank) * float(p))
        max_so_far = max(max_so_far, adjusted)   # enforce monotonicity
        out[name] = {
            "p_raw": float(p),
            "p_adjusted": round(max_so_far, 8),
            "significant": bool(max_so_far < alpha),
            "rank": rank + 1,
        }
    return out


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------
def sensitivity_sweep(values: Sequence[float], outcomes: Sequence[float],
                      parameter: str = "alpha") -> Dict:
    """
    Summarise how an outcome responds to a swept parameter.

    Reports the elasticity at the midpoint, the argmin/argmax, and the width of
    the *plateau* within 5% of the best value, because for a tuning parameter
    the practically important question is how wide the good region is, not only
    where the optimum sits.
    """
    v = np.asarray(list(values), dtype=float)
    y = np.asarray(list(outcomes), dtype=float)
    if v.size < 2 or v.size != y.size:
        return {"parameter": parameter, "n": int(v.size)}

    order = np.argsort(v)
    v, y = v[order], y[order]
    best_i = int(np.argmin(y))
    worst_i = int(np.argmax(y))

    tolerance = abs(y[best_i]) * 0.05
    plateau = v[np.abs(y - y[best_i]) <= tolerance]

    mid = v.size // 2
    if 0 < mid < v.size - 1 and abs(v[mid]) > 1e-12 and abs(y[mid]) > 1e-12:
        dydx = (y[mid + 1] - y[mid - 1]) / max(v[mid + 1] - v[mid - 1], 1e-12)
        elasticity = float(dydx * v[mid] / y[mid])
    else:
        elasticity = 0.0

    return {
        "parameter": parameter,
        "n": int(v.size),
        "best_value": round(float(v[best_i]), 6),
        "best_outcome": round(float(y[best_i]), 6),
        "worst_value": round(float(v[worst_i]), 6),
        "worst_outcome": round(float(y[worst_i]), 6),
        "range_pct": round(float(100.0 * (y.max() - y.min()) / abs(y.min()))
                           if abs(y.min()) > 1e-12 else 0.0, 4),
        "elasticity_at_midpoint": round(elasticity, 5),
        "plateau_within_5pct": [round(float(plateau.min()), 6),
                                round(float(plateau.max()), 6)] if plateau.size else None,
        "sweep": [{"value": round(float(a), 6), "outcome": round(float(c), 6)}
                  for a, c in zip(v, y)],
    }


def summarise_metric(samples: Sequence[float], name: str = "",
                     confidence: float = 0.95, seed: int = 42) -> Dict:
    """Mean with a bootstrap interval plus robust order statistics."""
    x = np.asarray(list(samples), dtype=float)
    x = x[np.isfinite(x)]
    ci = bootstrap_ci(x, confidence=confidence, seed=seed)
    return {
        "metric": name,
        "n": int(x.size),
        "mean": round(float(np.mean(x)), 5) if x.size else 0.0,
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "median": round(float(np.median(x)), 5) if x.size else 0.0,
        "iqr": [round(float(np.quantile(x, 0.25)), 5),
                round(float(np.quantile(x, 0.75)), 5)] if x.size else [0.0, 0.0],
        "std": round(float(np.std(x, ddof=1)), 5) if x.size > 1 else 0.0,
        "min": round(float(np.min(x)), 5) if x.size else 0.0,
        "max": round(float(np.max(x)), 5) if x.size else 0.0,
    }


def describe_protocol() -> Dict:
    return {
        "intervals": "BCa bootstrap, 10,000 resamples, 95% confidence",
        "significance": "Wilcoxon signed-rank on paired snapshots; exact sign "
                        "permutation for small samples",
        "effect_size": "Cliff's delta (non-parametric) and paired Cohen's d",
        "multiple_comparisons": "Holm-Bonferroni across baselines",
        "pairing": "every policy is evaluated on the identical set of scenario "
                   "snapshots, so all comparisons are within-snapshot",
        "rationale": "the per-snapshot differences are skewed and heavy-tailed, "
                     "so normal-theory intervals and t-tests would misstate both "
                     "the interval width and the p-value",
    }
