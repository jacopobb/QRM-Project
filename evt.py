"""Points 5-7 - GEV and GPD on losses L = -Z, EVT-based VaR."""
import numpy as np
from scipy.stats import genextreme, genpareto
from config import *

def block_maxima(L, m=BLOCK):
    n = len(L) // m
    return L.values[-n * m:].reshape(n, m).max(axis=1)

def fit_gev(maxima):
    c, loc, scale = genextreme.fit(maxima)        # scipy: shape c = -xi
    return c, loc, scale

def extremal_index(L, u_q=0.95):
    """Intervals estimator (Ferro-Segers) - or use runs estimator; returns theta in (0,1]."""
    idx = np.flatnonzero(L.values > np.quantile(L, u_q))
    T = np.diff(idx)
    if len(T) == 0: return 1.0
    th = 2 * (T.sum() ** 2) / (len(T) * (T ** 2).sum()) if T.max() <= 2 else \
         2 * ((T - 1).sum() ** 2) / (len(T) * ((T - 1) * (T - 2)).sum())
    return float(min(1.0, th))

def var_gev(params, p, m=BLOCK, theta=1.0):
    """Quantile of the loss at level p: G(x) = F(x)^(m*theta)  =>  x = G^-1(p^(m*theta))."""
    c, loc, scale = params
    return genextreme.ppf(p ** (m * theta), c, loc, scale)

def fit_gpd(L, q=0.95):
    u = np.quantile(L, q); exc = L[L > u] - u
    xi, _, beta = genpareto.fit(exc, floc=0)
    return u, xi, beta, len(L), len(exc)

def var_gpd(u, xi, beta, n, nu, p):
    return u + beta / xi * (((n / nu) * (1 - p)) ** (-xi) - 1)

def to_return_var(mu, sig, q_loss):
    """Cases 3-5: back to returns with the GARCH forecasts. Loss VaR = -mu + sig*q; as return = -that."""
    return mu - sig * q_loss
# TODO: GEV QQ/PP plots + KS test, mean-excess plot, parameter-stability plot for threshold
