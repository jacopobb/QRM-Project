"""Points 8-9 - violation counts, Kupiec test, Expected Shortfall."""
import numpy as np
from scipy.stats import chi2, norm

def violations(r_oos, var_ret):
    return (r_oos < var_ret).astype(int)

def kupiec(viol, alpha):
    n, x = len(viol), viol.sum(); pi = x / n
    ll0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    ll1 = (n - x) * np.log(1 - pi) + (x * np.log(pi) if x else 0)
    lr = -2 * (ll0 - ll1)
    return x, n * alpha, lr, 1 - chi2.cdf(lr, 1)    # violations, expected, LR, p-value

def es_gaussian(mu, sig, alpha):
    return mu - sig * norm.pdf(norm.ppf(alpha)) / alpha

def es_gpd(var_loss, sig, xi, beta, u, mu=0):
    """ES of loss for GPD tail: (VaR + beta - xi*u) / (1 - xi), then scaled back to returns by caller."""
    return (var_loss + beta - xi * u) / (1 - xi)
