"""Homework 2 - statistical tests and loss functions shared by the backtesting scripts (Points 2, 3, 4).
All functions take numpy arrays; VaR/ES are return quantiles (negative numbers), hit I_t = 1{r_t < VaR_t}."""
import numpy as np
from scipy import stats
from scipy.special import xlogy                              # x*log(y) with 0*log(0) = 0 (no hits / no transitions)


# ------------------------------------------------------------------ VaR coverage / independence
def kupiec_uc(I, a):
    """Kupiec (1995) unconditional coverage: LR_uc = -2 log[L(a)/L(pi_hat)] ~ chi2(1)."""
    n, n1 = len(I), int(I.sum()); n0 = n - n1; ph = n1 / n   # days, hits, non-hits, observed hit rate
    lr = -2 * (xlogy(n0, 1 - a) + xlogy(n1, a) - xlogy(n0, 1 - ph) - xlogy(n1, ph))   # likelihood ratio
    return lr, stats.chi2.sf(lr, 1)


def christoffersen_ind(I):
    """Christoffersen (1998) independence: first-order Markov chain vs i.i.d. hits, LR_ind ~ chi2(1)."""
    a, b = I[:-1], I[1:]                                     # hit yesterday, hit today
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())   # transitions from no hit
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())   # transitions from a hit
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)       # log-likelihood under independence
    l1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)   # under the Markov chain
    lr = max(-2 * (l0 - l1), 0.0)
    return lr, stats.chi2.sf(lr, 1), n11                     # n11 = consecutive hits


def christoffersen_cc(I, a):
    """Conditional coverage LR_cc = LR_uc + LR_ind ~ chi2(2)."""
    lr = kupiec_uc(I, a)[0] + christoffersen_ind(I)[0]
    return lr, stats.chi2.sf(lr, 2)


def ljung_box(x, m=5):
    """Ljung-Box Q(m) on the series x (demeaned) ~ chi2(m)."""
    x = x - x.mean(); n = len(x); d = (x**2).sum()
    if d == 0:                                               # constant series: test not defined
        return np.nan, np.nan
    rho = np.array([(x[k:] * x[:-k]).sum() / d for k in range(1, m + 1)])   # autocorrelations, lags 1..m
    q = n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, m + 1)))
    return q, stats.chi2.sf(q, m)


def berkowitz_ind(I, a, m=5):
    """Independence test of Berkowitz, Christoffersen and Pelletier (2011): Ljung-Box on the demeaned hit
    sequence I_t - a up to lag m. Under H0 the hits are serially uncorrelated."""
    return ljung_box(I - a, m)


def dq_test(I, var, a, lags=4):
    """Engle-Manganelli (2004) Dynamic Quantile test: regress Hit_t = I_t - a on a constant, Hit_{t-1..t-lags}
    and VaR_t; DQ = b' X'X b / (a(1-a)) ~ chi2(lags + 2) under H0 (all coefficients zero)."""
    H = I - a                                                # Hit_t
    T = len(H)
    X = np.column_stack([np.ones(T - lags)] + [H[lags - j - 1:T - j - 1] for j in range(lags)] + [var[lags:]])   # regressors
    y = H[lags:]
    b = np.linalg.pinv(X.T @ X) @ X.T @ y                    # OLS (pinv: robust to collinearity when there are few hits)
    dq = b @ X.T @ X @ b / (a * (1 - a))
    return dq, stats.chi2.sf(dq, X.shape[1])


# ------------------------------------------------------------------ loss functions
def tick_loss(r, var, a):
    """Quantile ('tick') loss of Gonzalez-Rivera, Lee and Mishra (2004): (a - 1{r < VaR}) (r - VaR).
    Strictly consistent for the quantile, so it ranks VaR forecasts correctly in expectation."""
    return (a - (r < var)) * (r - var)


def lopez_loss(r, var, a=None):
    """Lopez (1999) magnitude loss: 1 + (r - VaR)^2 on violation days, 0 otherwise."""
    return np.where(r < var, 1 + (r - var) ** 2, 0.0)


def fz0_loss(r, var, es, a):
    """FZ0 loss of Patton, Ziegel and Chen (2019, J. of Econometrics 211), from the Fissler-Ziegel (2016) class:
    L = -1/(a ES) 1{r <= VaR}(VaR - r) + VaR/ES + log(-ES) - 1, with VaR, ES < 0.
    Jointly strictly consistent for (VaR, ES): the only kind of loss that can rank ES forecasts."""
    return -1 / (a * es) * (r <= var) * (var - r) + var / es + np.log(-es) - 1


# ------------------------------------------------------------------ Diebold-Mariano
def nw_lrv(d, lags=None):
    """Newey-West long-run variance of d (Bartlett kernel)."""
    T = len(d); d = d - d.mean()
    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))       # usual automatic lag length
    v = d @ d / T                                            # variance
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (d[k:] @ d[:-k]) / T # weighted autocovariances
    return v


def diebold_mariano(l1, l2, lags=None):
    """DM test on d = l1 - l2 (H0: equal expected loss). stat < 0: model 1 better. Two-sided p-value, N(0,1)."""
    d = l1 - l2                                              # loss differential
    v = nw_lrv(d, lags)
    if v <= 0:                                               # identical losses
        return np.nan, np.nan
    s = d.mean() / np.sqrt(v / len(d))
    return s, 2 * stats.norm.sf(abs(s))


# ------------------------------------------------------------------ Model Confidence Set
def block_indices(T, B, block, rng):
    """Circular moving-block bootstrap: B index vectors of length T."""
    nb = int(np.ceil(T / block))                             # blocks per bootstrap sample
    starts = rng.integers(0, T, (B, nb))                     # random block starts
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % T   # consecutive days, wrapping at the end
    return idx.reshape(B, -1)[:, :T]


def mcs(L, names, alpha=0.10, B=1000, block=10, seed=42):
    """Model Confidence Set of Hansen, Lunde and Nason (2011) with the T_max statistic.
    L: (T, m) losses. Returns a dict name -> MCS p-value; models with p-value >= alpha are in the MCS."""
    L = np.asarray(L, float); T, m = L.shape
    rng = np.random.default_rng(seed)
    idx = block_indices(T, B, block, rng)                    # same bootstrap samples for every elimination step
    Lbar = L.mean(axis=0)                                    # average loss of each model
    Lbar_b = np.vstack([L[idx[i:i + 100]].mean(axis=1) for i in range(0, B, 100)])   # (B, m) bootstrap averages
    alive = list(range(m)); pvals = {}; pmax = 0.0           # models still in the set
    while len(alive) > 1:
        lb, lbb = Lbar[alive], Lbar_b[:, alive]
        d = lb - lb.mean()                                   # loss of each model relative to the average of the set
        db = lbb - lbb.mean(axis=1, keepdims=True)
        var = ((db - d) ** 2).mean(axis=0)                   # bootstrap variance of d
        var[var <= 0] = np.nan
        t = d / np.sqrt(var)                                 # t-statistics
        if np.all(np.isnan(t)):                              # remaining models have identical losses: none can be removed
            break
        tb = np.nanmax((db - d) / np.sqrt(var), axis=1)      # bootstrap distribution of T_max
        p = (tb > np.nanmax(t)).mean()                       # p-value of H0: equal predictive ability
        pmax = max(pmax, p)                                  # MCS p-values are non-decreasing
        worst = alive[int(np.nanargmax(t))]                  # model with the largest relative loss
        pvals[names[worst]] = pmax
        alive.remove(worst)                                  # eliminate it and repeat
    for j in alive:
        pvals[names[j]] = 1.0                                # surviving model(s)
    return pvals
