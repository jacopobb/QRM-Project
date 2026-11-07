"""Points 5-7 - GEV and GPD on losses L = -Z, EVT-based VaR."""
import numpy as np
import pandas as pd
from scipy.stats import genextreme, genpareto, norm, kstest
from settings import *
import matplotlib.pyplot as plt

TABLE_DIR, FIG_DIR = "outputs/tables", "outputs/figures"

def block_maxima(L, m=BLOCK):
    n = len(L) // m
    return L.values[-n * m:].reshape(n, m).max(axis=1)

def fit_gev(maxima):
    #shape,mu,sigma
    c, loc, scale = genextreme.fit(maxima)        # scipy: shape c = -xi
    return c, loc, scale

def gev_residuals(maxima, params):
    """Under a correct GEV, t = -log G(M) = (1 + xi*(M-mu)/sigma)^(-1/xi) is Exponential(1)."""
    return -genextreme.logcdf(maxima, *params)

def extremal_index(L, u_q=0.95, k=BLOCK):
    """Blocks estimator of the extremal index (slides 50-51):
    theta = (1/k) * log(1 - G(u)) / log(1 - N(u)),
    N(u) = fraction of observations above u, G(u) = fraction of blocks (size k) whose maximum is above u.
    u = u_q quantile of the losses. Same blocks as the GEV (oldest incomplete block discarded, slide 19)."""
    x = L.values; n = len(x) // k; x = x[-n * k:]
    u = np.quantile(x, u_q)
    N_u = (x > u).mean()
    G_u = (x.reshape(n, k).max(axis=1) > u).mean()
    if G_u >= 1 or N_u <= 0: return np.nan
    return float(np.log(1 - G_u) / (k * np.log(1 - N_u)))

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


def main():
    ds = pd.read_csv("data/Z_insample.csv", index_col=0, parse_dates=True).sort_index()
    #We need to take in cosideration the in sample resiuals and turn them into losses:
    
    ###############
    ### POINT 5 ###
    ###############

    L = -ds  # losses = -Z

    #we create a dictionary to store the GEV parameters for each asset
    gev_params, rows = {}, {}

    # we iter for each tiker and the last EWP to get the GEV parameters 
    # for each asset and the EWP

    for c in L.columns:
        maxima = block_maxima(L[c])
        gev_params[c] = fit_gev(maxima)
        rows[c] = {"n_blocks": len(maxima), "xi": -gev_params[c][0],        # xi = -c
                   "mu": gev_params[c][1], "sigma": gev_params[c][2]}
    tab_a = pd.DataFrame(rows).T
    print("\n=== E5a GEV parameters ===\n", tab_a.round(4).to_string()); tab_a.round(6).to_csv(f"{TABLE_DIR}/E5a_gev_parameters.csv")


    #now we have to detemine the quality of the fit for each asset e EWP:

    #under correct specification the residuals from the block maxima should
    # follow a exponential random variable with unit parameter.
    # we can see this by plotting the QQ plot so theoretical quantiles vs the observed
    fig, axes = plt.subplots(5, 4, figsize=(14, 14)); axes = axes.ravel()
    rows = {}
    for ax, c in zip(axes, L.columns):
        t = np.sort(gev_residuals(block_maxima(L[c]), gev_params[c]))
        n = len(t)
        theo = -np.log(1 - (np.arange(1, n + 1) - 0.5) / n)                # Exp(1) quantiles
        ax.scatter(theo, t, s=6); ax.plot([0, theo.max()], [0, theo.max()], "r-"); ax.set_title(c, fontsize=9)
        rows[c] = {"resid_mean": t.mean(), "resid_std": t.std(ddof=1), "KS_p": kstest(t, "expon").pvalue}
    for ax in axes[len(L.columns):]: ax.axis("off")
    fig.suptitle("Fig E5 - QQ-plot of GEV residuals vs Exp(1)"); fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/E5_gev_residual_qq.png", dpi=150)
    tab_b = pd.DataFrame(rows).T                       # KS p-value approximate: GEV parameters are estimated
    print("\n=== E5b fit quality (expected: mean 1, std 1, KS_p > 0.05) ===\n", tab_b.round(4).to_string()); tab_b.round(6).to_csv(f"{TABLE_DIR}/E5b_gev_fit_quality.csv")

    # 4) Extremal index theta of the daily losses (blocks estimator, slides 50-52; thresholds 90%, 95%, 99% as in slide 52)  -> TABLE E5c
    theta_tab = pd.DataFrame({f"theta_u{round(q * 100)}": {c: extremal_index(L[c].dropna(), q) for c in L.columns}
                              for q in (0.95, 0.99)})
    print("\n=== E5c extremal index ===\n", theta_tab.round(3).to_string()); theta_tab.round(6).to_csv(f"{TABLE_DIR}/E5c_extremal_index.csv")
    theta = theta_tab["theta_u95"].clip(upper=1.0)       # used in the VaR; theta lives in (0,1] (slide 17): raw estimate > 1 = no clustering

    # 5) VaR of Z (loss quantile, positive number) at p = 1 - alpha, without and with theta  -> TABLE E5d
    rows = []
    for c in L.columns:
        for a in ALPHAS:
            p = 1 - a
            rows.append({"series": c, "alpha": a, "theta": theta[c],
                         "VaR_GEV_no_theta": var_gev(gev_params[c], p),
                         "VaR_GEV_with_theta": var_gev(gev_params[c], p, theta=theta[c]),
                         "VaR_empirical": np.quantile(L[c].dropna(), p),    # sanity check
                         "VaR_Normal": norm.ppf(p)})
    tab_d = pd.DataFrame(rows).set_index(["series", "alpha"])
    print("\n=== E5d VaR of Z ===\n", tab_d.round(4).to_string()); tab_d.round(6).to_csv(f"{TABLE_DIR}/E5d_var_Z_gev.csv")

    ###############
    ### POINT 6 ###
    ###############

    # Applichiamo il metodo Peaks-Over-Threshold (POT) usando la GPD sulle perdite L
    gpd_params = {}
    rows_gpd = []

    for c in L.columns:
        # 1. Stima dei parametri GPD per ogni serie. 
        # Di default fit_gpd usa come soglia (u) il quantile al 95% (q=0.95).
        u, xi, beta, n, nu = fit_gpd(L[c].dropna(), q=0.95)
        gpd_params[c] = {"u": u, "xi": xi, "beta": beta, "n": n, "nu": nu}
        
        # 2. Calcolo del VaR per i livelli alpha (0.05 e 0.01)
        for a in ALPHAS:
            p = 1 - a
            var_g = var_gpd(u, xi, beta, n, nu, p)
            var_emp = np.quantile(L[c].dropna(), p)
            var_norm = norm.ppf(p)
            
            rows_gpd.append({
                "series": c, 
                "alpha": a, 
                "threshold_u": u, 
                "xi": xi, 
                "beta": beta,
                "n_exceedances": nu,
                "VaR_GPD": var_g,
                "VaR_empirical": var_emp,
                "VaR_Normal": var_norm
            })
            
    tab_gpd = pd.DataFrame(rows_gpd).set_index(["series", "alpha"])
    print("\n=== E6 VaR of Z (GPD) ===\n", tab_gpd.round(4).to_string())
    tab_gpd.round(6).to_csv(f"{TABLE_DIR}/E6_var_Z_gpd.csv")

    # --- PLOT E6: QQ-Plot delle eccedenze GPD vs Quantili Teorici GPD ---
    fig, axes = plt.subplots(5, 4, figsize=(14, 14)); axes = axes.ravel()
    for ax, c in zip(axes, L.columns):
        # Estraiamo i parametri calcolati
        u = gpd_params[c]["u"]
        xi = gpd_params[c]["xi"]
        beta = gpd_params[c]["beta"]
        
        # Filtriamo le eccedenze reali
        exc = L[c].dropna()[L[c].dropna() > u] - u
        exc_sorted = np.sort(exc)
        
        # Calcoliamo i quantili teorici della GPD
        # Formula: y = (beta/xi) * ((1-p)^(-xi) - 1)
        p = (np.arange(1, len(exc) + 1) - 0.5) / len(exc)
        theo_q = (beta / xi) * ((1 - p)**(-xi) - 1)
        
        ax.scatter(theo_q, exc_sorted, s=6)
        ax.plot([0, theo_q.max()], [0, theo_q.max()], "r-")
        ax.set_title(c, fontsize=9)
        
    for ax in axes[len(L.columns):]: ax.axis("off")
    fig.suptitle("Fig E6 - QQ-plot of GPD exceedances"); fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/E6_gpd_qq.png", dpi=150)
    
    ###############
    ### POINT 7 ###
    ###############
    # Casi 3-5: Trasformare il VaR dei residui (loss) in VaR dei rendimenti
    # tramite le previsioni GARCH mu e sigma_normal calcolate nel Punto 4.
    
    FC_DIR = "outputs/forecasts"
    
    for c in L.columns:
        fc_path = f"{FC_DIR}/{c}.csv"
        import os
        if not os.path.exists(fc_path):
            continue  # Salta se la previsione per questo asset/EWP non esiste ancora
            
        # Carichiamo le previsioni del modello GARCH per il titolo
        fc = pd.read_csv(fc_path, index_col=0, parse_dates=True)
        
        for a in ALPHAS:
            # Quantile EVT 1: GEV indipendenza (nessun theta) -> Punto 5
            q3 = tab_d.loc[(c, a), "VaR_GEV_no_theta"]
            
            # Quantile EVT 2: GEV con clustering (con theta) -> Punto 5
            q4 = tab_d.loc[(c, a), "VaR_GEV_with_theta"]
            
            # Quantile EVT 3: GPD indipendenza -> Punto 6
            q5 = tab_gpd.loc[(c, a), "VaR_GPD"]
            
            # Calcoliamo i nuovi VaR per i rendimenti (Case 3, 4, 5)
            # Nota: si usa sigma_normal perché i residui sono stati estratti dal GARCH gaussiano
            fc[f"VaR_case3_{a}"] = to_return_var(fc["mu"], fc["sigma_normal"], q3)
            fc[f"VaR_case4_{a}"] = to_return_var(fc["mu"], fc["sigma_normal"], q4)
            fc[f"VaR_case5_{a}"] = to_return_var(fc["mu"], fc["sigma_normal"], q5)
            
        # Sovrascriviamo il file CSV in outputs/forecasts aggiornandolo con le nuove colonne.
        # Questi file saranno pronti per essere letti dal punto 8 (backtesting) e punto 9 (ES).
        fc.to_csv(fc_path)
        
    print(f"\n=== E7 VaR sui rendimenti (Casi 3, 4 e 5) calcolato e aggiornato in {FC_DIR} ===")

    # --- TABELLA E PLOT E7: Confronto VaR Out-of-Sample per PORT ---
    
    # 1. Creiamo una tabella riassuntiva del VaR medio out-of-sample per ogni asset all'1%
    summary_e7 = []
    for c in L.columns:
        fc = pd.read_csv(f"{FC_DIR}/{c}.csv", index_col=0, parse_dates=True)
        summary_e7.append({
            "Series": c,
            "Avg_VaR_Normal_1%": fc["VaR_case1_0.01"].mean(),
            "Avg_VaR_GEV_1%": fc["VaR_case4_0.01"].mean(),
            "Avg_VaR_GPD_1%": fc["VaR_case5_0.01"].mean()
        })
    tab_e7 = pd.DataFrame(summary_e7).set_index("Series")
    print("\n=== E7 Average Out-of-Sample VaR (1%) ===")
    print(tab_e7.round(4).to_string())
    tab_e7.round(6).to_csv(f"{TABLE_DIR}/E7_avg_var_comparison.csv")

    # 2. Plot della serie storica del VaR per il portafoglio (PORT)
    c = "PORT"
    if os.path.exists(f"{FC_DIR}/{c}.csv"):
        fc_port = pd.read_csv(f"{FC_DIR}/{c}.csv", index_col=0, parse_dates=True)
        
        fig, ax = plt.subplots(figsize=(12, 5))
        # Plot dei rendimenti
        ax.plot(fc_port.index, fc_port["ret"], label="PORT Returns", color="lightgrey", lw=1.5)
        
        # Plot delle 3 curve VaR all'1% (le rendiamo negative per metterle sul grafico dei rendimenti)
        # Nota: nel file fc, i VaR per i ritorni sono già calcolati con il segno corretto dalla to_return_var
        ax.plot(fc_port.index, fc_port["VaR_case1_0.01"], label="VaR 1% Case 1 (Normal)", color="blue", lw=1.2, alpha=0.8)
        ax.plot(fc_port.index, fc_port["VaR_case4_0.01"], label="VaR 1% Case 4 (GEV with theta)", color="orange", lw=1.2, linestyle="--", alpha=0.8)
        ax.plot(fc_port.index, fc_port["VaR_case5_0.01"], label="VaR 1% Case 5 (GPD)", color="red", lw=1.2, linestyle="-.", alpha=0.8)
        
        ax.set_title("Fig E7 - Out-of-Sample PORT Returns vs 1% VaR Models")
        ax.set_ylabel("Returns / VaR (%)")
        ax.legend(fontsize=9, loc="lower left")
        fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/E7_VaR_OOS_PORT.png", dpi=150)
if __name__ == "__main__":
    main()
