"""Homework 2 - AR(1)-GJR-GARCH(1,1) written with numpy/scipy only (no `arch` package), so every formula is visible.
Responsible: member B (reviewer: C).

    r_t      = c + phi * r_{t-1} + eps_t ,           eps_t = sigma_t * z_t
    sigma2_t = omega + (alpha + gamma * 1{eps_{t-1} < 0}) * eps_{t-1}^2 + beta * sigma2_{t-1}

z_t ~ i.i.d. with mean 0 and variance 1: Normal, standardised Student-t, or Hansen (1994) skewed-t.
The variance recursion is a linear filter in eps^2, so it is computed with scipy.signal.lfilter (C speed).
sigma2 of the first residual is set to the sample variance of the residuals (backcast).
Used by: var_garch.py (M09-M15), var_factors.py (factor GARCH, M21), var_10d.py (FHS over 10 days).
"""
import numpy as np                                           # arrays
from scipy import optimize, stats                            # optimiser, distributions
from scipy.signal import lfilter                             # fast linear recursion
from scipy.special import gammaln                            # log-gamma for the t / skewed-t densities

PARAM_NAMES = ["c", "phi", "omega", "alpha", "gamma", "beta"]   # order of the parameter vector theta
DIST_PARAMS = {"normal": [], "t": ["nu"], "skewt": ["nu", "lam"]}  # extra parameters of each innovation density


# ------------------------------------------------------------------ Hansen (1994) skewed-t, standardised (mean 0, var 1)
def _skt_abc(nu, lam):
    """Constants a, b, c of Hansen's skewed-t (nu = degrees of freedom, lam = skewness in (-1, 1))."""
    c = np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2)) / np.sqrt(np.pi * (nu - 2))   # normalising constant
    a = 4 * lam * c * (nu - 2) / (nu - 1)                    # shift that makes the mean 0
    b = np.sqrt(1 + 3 * lam**2 - a**2)                       # scale that makes the variance 1
    return a, b, c


def skewt_logpdf(z, nu, lam):
    """Log-density of the standardised skewed-t."""
    a, b, c = _skt_abc(nu, lam)
    s = np.where(z < -a / b, 1 - lam, 1 + lam)               # different scale on the left and right of the mode
    return np.log(b) + np.log(c) - (nu + 1) / 2 * np.log1p(((b * z + a) / s) ** 2 / (nu - 2))


def skewt_ppf(u, nu, lam):
    """Quantile function of the standardised skewed-t (inverse of the cdf, piecewise on the two sides)."""
    a, b, _ = _skt_abc(nu, lam)
    u = np.asarray(u, float)
    k = np.sqrt((nu - 2) / nu)                               # from a Student-t with nu df to unit variance
    lo = (1 - lam) / b * k * stats.t.ppf(u / (1 - lam), nu) - a / b                            # left branch
    hi = (1 + lam) / b * k * stats.t.ppf(0.5 + (u - (1 - lam) / 2) / (1 + lam), nu) - a / b   # right branch
    return np.where(u < (1 - lam) / 2, lo, hi)               # (1-lam)/2 = probability of the left branch


def std_t_ppf(u, nu):
    """Quantile of the Student-t rescaled to unit variance."""
    return stats.t.ppf(u, nu) * np.sqrt((nu - 2) / nu)


def std_t_cdf(z, nu):
    """Cdf of the unit-variance Student-t (used for the PIT in Point 4)."""
    return stats.t.cdf(z * np.sqrt(nu / (nu - 2)), nu)


def std_t_es(a, nu):
    """Expected shortfall (lower tail, negative number) of the unit-variance Student-t at level a."""
    q = stats.t.ppf(a, nu)                                   # a-quantile of the standard t
    return -stats.t.pdf(q, nu) / a * (nu + q**2) / (nu - 1) * np.sqrt((nu - 2) / nu)   # closed form, rescaled


# ------------------------------------------------------------------ filter and likelihood
def filter_garch(theta, r):
    """Residuals eps_t, variances sigma2_t (t = 1..T-1) and the one-step-ahead forecasts (mu_{T+1}, sigma2_{T+1})."""
    c, phi, omega, alpha, gamma, beta = theta[:6]            # mean and variance parameters
    eps = r[1:] - c - phi * r[:-1]                           # AR(1) residuals (the first day is lost)
    e2 = eps**2                                              # squared residuals
    x = omega + (alpha + gamma * (eps < 0)) * e2             # input of the filter: sigma2_t = x_{t-1} + beta * sigma2_{t-1}
    s2 = np.empty_like(eps)
    s2[0] = e2.mean()                                        # backcast: first variance = sample variance of the residuals
    s2[1:] = lfilter([1.0], [1.0, -beta], x[:-1], zi=[beta * s2[0]])[0]   # GJR-GARCH recursion for t = 2..T-1
    mu_next = c + phi * r[-1]                                # conditional mean forecast for T+1
    s2_next = x[-1] + beta * s2[-1]                          # conditional variance forecast for T+1
    return eps, s2, mu_next, s2_next


def loglik(theta, r, dist):
    """Log-likelihood of the AR(1)-GJR-GARCH(1,1) with the chosen innovation density."""
    eps, s2, _, _ = filter_garch(theta, r)
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):      # invalid variance path -> reject these parameters
        return -1e10
    z = eps / np.sqrt(s2)                                    # standardised residuals
    if dist == "normal":                                     # Gaussian log-density (QML)
        ll = -0.5 * (np.log(2 * np.pi) + np.log(s2) + z**2)
    elif dist == "t":                                        # unit-variance Student-t log-density
        nu = theta[6]
        ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
              - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(z**2 / (nu - 2)))
    else:                                                    # Hansen skewed-t log-density (Jacobian -0.5 log s2)
        ll = skewt_logpdf(z, theta[6], theta[7]) - 0.5 * np.log(s2)
    v = ll.sum()
    return v if np.isfinite(v) else -1e10


def start_values(r, dist):
    """Cold-start values: typical daily equity estimates."""
    v = r.var()
    th = [r.mean(), 0.0, 0.05 * v, 0.03, 0.08, 0.88]         # c, phi, omega, alpha, gamma, beta
    return np.array(th + {"normal": [], "t": [8.0], "skewt": [8.0, -0.05]}[dist])   # + nu (+ lam)


def bounds(r, dist):
    """Box constraints of the optimiser."""
    v = r.var()
    b = [(-abs(r.mean()) - 1, abs(r.mean()) + 1), (-0.99, 0.99), (1e-6 * v, 2 * v), (0.0, 0.5), (-0.5, 0.5), (0.0, 0.999)]
    return b + {"normal": [], "t": [(2.1, 100.0)], "skewt": [(2.1, 100.0), (-0.95, 0.95)]}[dist]   # nu > 2: finite variance


def fit(r, dist="normal", start=None):
    """MLE (Gaussian = QML) with SLSQP under the constraints alpha + gamma >= 0 and alpha + gamma/2 + beta <= 0.999.
    `start` = previous estimate (warm start); a cold start is tried too if the warm start fails."""
    r = np.asarray(r, float)
    cons = [{"type": "ineq", "fun": lambda th: th[3] + th[4]},                       # news impact >= 0 for negative shocks
            {"type": "ineq", "fun": lambda th: 0.999 - th[3] - th[4] / 2 - th[5]}]   # covariance stationarity
    f = lambda th: -loglik(th, r, dist) / len(r)             # objective: minus the average log-likelihood
    lo, hi = [b[0] for b in bounds(r, dist)], [b[1] for b in bounds(r, dist)]
    best = None
    for th0 in ([start] if start is not None else []) + [start_values(r, dist)]:   # warm start first, then cold start
        th0 = np.clip(th0, lo, hi)                           # starting point inside the bounds
        res = optimize.minimize(f, th0, method="SLSQP", bounds=bounds(r, dist), constraints=cons,
                                options={"maxiter": 300, "ftol": 1e-9})
        if best is None or res.fun < best.fun:               # keep the best optimum found
            best = res
        if res.success and start is not None:                # warm start converged: no need for the cold start
            break
    return best.x


def forecast(theta, r):
    """Fixed parameters, data up to T: standardised residuals z_1..z_{T-1} and (mu_{T+1}, sigma_{T+1})."""
    eps, s2, mu, s2n = filter_garch(theta, np.asarray(r, float))
    return eps / np.sqrt(s2), mu, np.sqrt(s2n)


def simulate_paths(theta, r, z_pool, h, n_sim, rng):
    """Filtered Historical Simulation of h-day cumulative log returns: z drawn with replacement from z_pool,
    mean and variance recursions updated along each path. Returns an (n_sim,) array of sums of h returns."""
    c, phi, omega, alpha, gamma, beta = theta[:6]
    eps, s2, _, _ = filter_garch(theta, np.asarray(r, float))    # filtered state at the end of the window
    r_prev = np.full(n_sim, r[-1]); e_prev = np.full(n_sim, eps[-1]); s2_prev = np.full(n_sim, s2[-1])   # same start for all paths
    total = np.zeros(n_sim)                                  # cumulative return of each path
    for _ in range(h):                                       # one step per day
        s2_t = omega + (alpha + gamma * (e_prev < 0)) * e_prev**2 + beta * s2_prev   # variance of the next day
        e_t = np.sqrt(s2_t) * rng.choice(z_pool, n_sim, replace=True)              # bootstrapped standardised shock
        r_t = c + phi * r_prev + e_t                         # simulated return
        total += r_t                                         # log returns add up over the horizon
        r_prev, e_prev, s2_prev = r_t, e_t, s2_t             # state for the next step
    return total
