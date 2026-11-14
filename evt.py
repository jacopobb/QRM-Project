"""Points 5-7 - GEV and GPD on losses L = -Z, EVT-based VaR."""
import numpy as np
import pandas as pd
from scipy.stats import genextreme, genpareto, norm, kstest
from statsmodels.tools.numdiff import approx_hess
from settings import *
import matplotlib.pyplot as plt
import os

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

def fit_gpd(L, u):
    """Slide 64-67: Adattamento GPD usando una soglia u scelta dal Mean Excess Plot."""
    exc = L[L > u] - u
    xi, _, beta = genpareto.fit(exc, floc=0)
    T = len(L)
    N_u = len(exc)
    return u, xi, beta, T, N_u

def LogLGPD(x, exc):                                     # prof's function, excesses passed as argument
    beta, xi = np.exp(x[0]), np.exp(x[1])
    return -(-len(exc) * np.log(beta) - (1 + 1/xi) * np.sum(np.log(1 + xi * exc / beta)))


def var_gpd(u, xi, beta, T, N_u, alpha):
    """Slide 72: q_alpha(F) = u + (beta/xi) * ((alpha / (N(u)/T))**(-xi) - 1)"""
    return u + (beta / xi) * ((alpha / (N_u / T)) ** (-xi) - 1)

def to_return_var(mu, sig, q_z):
    """Slide 75: q_alpha(F_x) = mu_{T+1} + sigma_{T+1} * q_alpha(F_z)"""
    return mu + sig * q_z

def main():
    ds = pd.read_csv("data/Z_insample.csv", index_col=0, parse_dates=True).sort_index()

    #We need to take in cosideration the in sample resiuals and turn them into losses:
    
    ###############
    ### POINT 5 ###
    ###############

    L = -ds  # losses = -Z

    #we create a dictionary to store the GEV parameters for each asset
    gev_params, rows = {}, {}

    # we iter for each tiker and the last PORT to get the GEV parameters 
    # for each asset and the PORT

    for c in L.columns:
        maxima = block_maxima(L[c])
        gev_params[c] = fit_gev(maxima)
        rows[c] = {"n_blocks": len(maxima), "xi": -gev_params[c][0],        # xi = -c
                   "mu": gev_params[c][1], "sigma": gev_params[c][2]}
    tab_a = pd.DataFrame(rows).T
    print("\n=== E5a GEV parameters ===\n", tab_a.round(4).to_string()); tab_a.round(6).to_csv(f"{TABLE_DIR}/E5a_gev_parameters.csv")


    #now we have to detemine the quality of the fit for each asset e PORT:
    #EWP
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
    fig.suptitle("Fig F6.1 - QQ-plot of GEV residuals vs Exp(1)"); fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/E5_gev_residual_qq.png", dpi=150)
    tab_b = pd.DataFrame(rows).T                       # KS p-value approximate: GEV parameters are estimated
    print("\n=== E5b fit quality (expected: mean 1, std 1, KS_p > 0.05) ===\n", tab_b.round(4).to_string()); tab_b.round(6).to_csv(f"{TABLE_DIR}/E5b_gev_fit_quality.csv")

    # 4) Extremal index theta of the daily losses (blocks estimator, slides 50-52; thresholds 90%, 95%, 99% as in slide 52)  -> TABLE E5c
    theta_tab = pd.DataFrame({f"theta_u{round(q * 100)}": {c: extremal_index(L[c].dropna(), q) for c in L.columns}
                              for q in (0.95, 0.99)})
    print("\n=== E5c extremal index ===\n", theta_tab.round(3).to_string()); theta_tab.round(6).to_csv(f"{TABLE_DIR}/E5c_extremal_index.csv")
    theta_tab = theta_tab.clip(upper=1.0)       # used in the VaR; theta lives in (0,1] (slide 17): raw estimate > 1 = no clustering

    # 5) VaR of Z (loss quantile, positive number) at p = 1 - alpha, without and with theta  -> TABLE E5d
    rows = []
    for c in L.columns:
        for a in ALPHAS:
            p = 1 - a
            rows.append({"series": c, "alpha": a, "theta": theta_tab.loc[c,f"theta_u{round(p*100)}"],
                         "VaR_GEV_no_theta": var_gev(gev_params[c], p),
                         "VaR_GEV_with_theta": var_gev(gev_params[c], p, theta=theta_tab.loc[c,f"theta_u{round(p*100)}"]),
                         "VaR_empirical": np.quantile(L[c].dropna(), p),    # sanity check
                         "VaR_Normal": norm.ppf(p)})
    tab_d = pd.DataFrame(rows).set_index(["series", "alpha"])
    print("\n=== E5d VaR of Z ===\n", tab_d.round(4).to_string()); tab_d.round(6).to_csv(f"{TABLE_DIR}/E5d_var_Z_gev.csv")

    # --- PLOT E5: Istogramma Massimi a Blocchi vs Curva GEV ---
    fig, axes = plt.subplots(5, 4, figsize=(14, 14)); axes = axes.ravel()
    for ax, c in zip(axes, L.columns):
        maxima = block_maxima(L[c])
        params = gev_params[c] # c, loc, scale (ricorda che c = -xi)
        
        # Istogramma empirico dei massimi
        ax.hist(maxima, bins=15, density=True, alpha=0.6, color='steelblue', edgecolor='black')
        
        # Curva teorica GEV
        x_vals = np.linspace(maxima.min(), maxima.max(), 100)
        # genextreme.pdf calcola la densità di probabilità (f_X(x) della slide 21)
        pdf_vals = genextreme.pdf(x_vals, *params)
        
        ax.plot(x_vals, pdf_vals, 'r-', lw=2, label=f'GEV ($\\xi$={-params[0]:.2f})')
        ax.set_title(c, fontsize=9)
        ax.legend(fontsize=8)
        
    for ax in axes[len(L.columns):]: ax.axis("off")
    fig.suptitle("Fig F6 - Histogram of Block Maxima with Fitted GEV Density")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/E5_gev_histograms.png", dpi=150)

    ###############
    ### POINT 6 ###
    ###############
    
    # 1. Definiamo il dizionario delle soglie manuali (i "gomiti" letti sull'asse X)
    manual_thresholds = {
        "Swisscom": 1, "Nestle": 1.1, "Richemont": 1.3, "UBS": 1.6,
        "Zurich": 1.1, "SwissRe": 0.9, "SwissLife": 1.5, "HelvetiaBaloise": 0.6,
        "Roche": 1, "Novartis": 1.6, "Lonza": 1.3, "ABB": 1.3,
        "Geberit": 1.4, "Givaudan": 1.5, "Holcim": 1.3, "Sika": 1.5,
        "PORT": 1.3, "PORT": 1.4
    }

    # --- PLOT: Mean Excess Plot per tutti gli asset (Scelta della soglia) ---
    fig, axes = plt.subplots(5, 4, figsize=(14, 14))
    axes = axes.ravel()
    
    for ax, c in zip(axes, L.columns):
        losses_c = L[c].dropna().values
        thresholds = np.linspace(np.percentile(losses_c, 50), np.percentile(losses_c, 99), 60)
        mean_excesses = []
        for th in thresholds:
            excesses = losses_c[losses_c > th] - th
            mean_excesses.append(excesses.mean() if len(excesses) > 0 else np.nan)
        ax.plot(thresholds, mean_excesses, 'b.-', markersize=4)

        u_val = manual_thresholds[c]                      # niente fallback silenzioso

        # limiti di validità: u deve stare sotto il quantile (1-alpha) delle perdite
        lim_5 = np.quantile(losses_c, 1 - max(ALPHAS))    # limite per il VaR 5% (il più stringente)
        lim_1 = np.quantile(losses_c, 1 - min(ALPHAS))    # limite per il VaR 1%
        ax.axvline(lim_5, color='black', linestyle=':', linewidth=1.3, label=f'limit 5% VaR ({lim_5:.2f})')
        ax.axvline(lim_1, color='grey',  linestyle=':', linewidth=1.3, label=f'limit 1% VaR ({lim_1:.2f})')

        ok = u_val < lim_5                                # soglia valida per tutti gli alpha?
        ax.axvline(u_val, color='green' if ok else 'red', linestyle='--', linewidth=1.5, label=f'u = {u_val}')
        ax.set_title(c + ("" if ok else "  (u > limit 5%)"), fontsize=9, color='black' if ok else 'red')
        ax.legend(fontsize=6, loc='upper left')
        
    for ax in axes[len(L.columns):]: 
        ax.axis("off")
        
    fig.suptitle("Fig E6.2 - Mean Excess Plots (Manual Threshold Selection)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/E6a_mean_excess_plots.png", dpi=150)

    # --- CALCOLO VaR GPD CON SOGLIE MANUALI ---
    gpd_params = {}
    rows_gpd = []

    for c in L.columns:
        # 1. Recupero soglia e stima dei parametri GPD
        u_val = manual_thresholds.get(c, np.percentile(L[c].dropna(), 95)) 
        u, xi, beta, T_len, N_u = fit_gpd(L[c].dropna(), u=u_val)
        y = np.log([beta, xi])
        exc = (L[c].dropna()[L[c].dropna() > u] - u).values 
        H = np.linalg.inv(approx_hess(y, lambda x: LogLGPD(x, excesses))) 
        G = np.diag(np.exp(y))
        se_beta, se_xi = np.sqrt(np.diag(G @ H @ G))
        z_beta, z_xi = beta / se_beta, xi / se_xi  


        gpd_params[c] = {"u": u, "xi": xi,"z_xi": z_xi, "z_beta": z_beta, "beta": beta, "T": T_len, "N_u": N_u}
        
        # 2. Calcolo del VaR per i livelli alpha richiesti
        for a in ALPHAS:
            var_g = var_gpd(u, xi, beta, T_len, N_u, alpha=a)
            var_emp = np.quantile(L[c].dropna(), 1 - a)
            var_norm = norm.ppf(1 - a)
            
            rows_gpd.append({
                "series": c, "alpha": a, "threshold_u": u, 
                "xi": xi, "beta": beta, "n_exceedances": N_u,
                "VaR_GPD": var_g, "VaR_empirical": var_emp, "VaR_Normal": var_norm
            })
            
    tab_gpd = pd.DataFrame(rows_gpd).set_index(["series", "alpha"])
    print("\n=== E6 VaR of Z (GPD) ===\n", tab_gpd.round(4).to_string())
    tab_gpd.round(6).to_csv(f"{TABLE_DIR}/E6_var_Z_gpd.csv")

    # --- PLOT E6: QQ-Plot delle eccedenze GPD vs Quantili Teorici GPD ---
    fig, axes = plt.subplots(5, 4, figsize=(14, 14)); axes = axes.ravel()
    for ax, c in zip(axes, L.columns):
        u = gpd_params[c]["u"]
        xi = gpd_params[c]["xi"]
        beta = gpd_params[c]["beta"]
        
        exc = L[c].dropna()[L[c].dropna() > u] - u
        exc_sorted = np.sort(exc)
        
        p = (np.arange(1, len(exc) + 1) - 0.5) / len(exc)
        theo_q = (beta / xi) * ((1 - p)**(-xi) - 1)
        
        ax.scatter(theo_q, exc_sorted, s=6)
        ax.plot([0, theo_q.max()], [0, theo_q.max()], "r-")
        ax.set_title(c, fontsize=9)
        
    for ax in axes[len(L.columns):]: ax.axis("off")
    fig.suptitle("Fig F6.4 - QQ-plot of GPD exceedances")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/E6_gpd_qq.png", dpi=150)
    
    # --- PLOT E6: Istogramma Eccedenze vs Curva GPD ---
    fig, axes = plt.subplots(5, 4, figsize=(14, 14)); axes = axes.ravel()
    for ax, c in zip(axes, L.columns):
        u = gpd_params[c]["u"]
        xi = gpd_params[c]["xi"]
        beta = gpd_params[c]["beta"]
        
        # Estraiamo le eccedenze (y_t = x_t - u) come da slide 63
        exc = L[c].dropna()[L[c].dropna() > u] - u
        
        # Istogramma empirico delle eccedenze
        ax.hist(exc, bins=20, density=True, alpha=0.6, color='darkorange', edgecolor='black')
        
        # Curva teorica GPD
        x_vals = np.linspace(0, exc.max(), 100)
        # scipy.stats.genpareto usa c=xi, loc=0, scale=beta per la funzione g(y)
        pdf_vals = genpareto.pdf(x_vals, c=xi, loc=0, scale=beta)
        
        ax.plot(x_vals, pdf_vals, 'r-', lw=2, label=f'GPD ($\\xi$={xi:.2f})')
        ax.set_title(c, fontsize=9)
        ax.legend(fontsize=8)
        
    for ax in axes[len(L.columns):]: ax.axis("off")
    fig.suptitle("Fig F6.3 - Histogram of Exceedances with Fitted GPD Density")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/E6_gpd_histograms.png", dpi=150)
    
    # --- TABELLA E6: Numero di eccedenze N(u) per ogni serie ---
    exceedances_data = []
    for c, params in gpd_params.items():
        exceedances_data.append({
            "Series": c,
            "Threshold_u": params["u"],
            "Total_Obs_T": params["T"],
            "Exceedances_N_u": params["N_u"],
            "%_Exceedances": round(params["N_u"] / params["T"] * 100, 2)
        })
        
    tab_exceedances = pd.DataFrame(exceedances_data).set_index("Series")
    print("\n=== Table E6: Number of Exceedances N(u) ===\n", tab_exceedances.to_string())
    tab_exceedances.to_csv(f"{TABLE_DIR}/E6_exceedances.csv")

    ###############
    ### POINT 7 ###
    ###############
    # Casi 3-5: Trasformare il VaR dei residui (loss) in VaR dei rendimenti
    # tramite le previsioni GARCH mu e sigma_normal calcolate nel Punto 4.
    
    FC_DIR = "outputs/forecasts"
    
    for c in L.columns:
        fc_path = f"{FC_DIR}/{c}.csv"
        if not os.path.exists(fc_path):
            continue  # Salta se la previsione per questo asset/PORT non esiste ancora
            
        # Carichiamo le previsioni del modello GARCH per il titolo
        fc = pd.read_csv(fc_path, index_col=0, parse_dates=True)
        
        for a in ALPHAS:
            # Essendo Z = -L, il quantile dei residui standardizzati z_t è il negativo del VaR delle perdite
            q_z3 = - tab_d.loc[(c, a), "VaR_GEV_no_theta"]
            q_z4 = - tab_d.loc[(c, a), "VaR_GEV_with_theta"]
            q_z5 = - tab_gpd.loc[(c, a), "VaR_GPD"]
            
            # Slide 75: Applichiamo la formula con il '+'
            fc[f"VaR_case3_{a}"] = to_return_var(fc["mu"], fc["sigma_normal"], q_z3)
            fc[f"VaR_case4_{a}"] = to_return_var(fc["mu"], fc["sigma_normal"], q_z4)
            fc[f"VaR_case5_{a}"] = to_return_var(fc["mu"], fc["sigma_normal"], q_z5)
            
        # Sovrascriviamo il file CSV in outputs/forecasts aggiornandolo con le nuove colonne.
        # Questi file saranno pronti per essere letti dal punto 8 (backtesting) e punto 9 (ES).
        fc.to_csv(fc_path)
        
    print(f"\n=== E7 VaR sui rendimenti (Casi 3, 4 e 5) calcolato e aggiornato in {FC_DIR} ===")

    # --- TABELLA E PLOT E7: Confronto VaR Out-of-Sample per PORT ---
    
    # 1. Creiamo una tabella riassuntiva del VaR medio out-of-sample per ogni asset all'1%
    summary_e7 = []
    for c in L.columns:
        fc = pd.read_csv(f"{FC_DIR}/{c}.csv", index_col=0, parse_dates=True)
        row = {"Series": c}
        for a in ALPHAS:
            s = f"{int(a*100)}%"
            row[f"Avg_VaR_case1_{s}"] = fc[f"VaR_case1_{a}"].mean()
            row[f"Avg_VaR_case3_{s}"] = fc[f"VaR_case3_{a}"].mean()
            row[f"Avg_VaR_case4_{s}"] = fc[f"VaR_case4_{a}"].mean()
            row[f"Avg_VaR_case5_{s}"] = fc[f"VaR_case5_{a}"].mean()
        summary_e7.append(row)

    tab_e7 = pd.DataFrame(summary_e7).set_index("Series")
    print("\n=== E7 Average Out-of-Sample VaR (1%) ===")
    print(tab_e7.round(4).to_string())
    tab_e7.round(6).to_csv(f"{TABLE_DIR}/E7_avg_var_comparison.csv")

    # A) PLOT E7 (Grid All Series): rendimenti OOS e VaR per TUTTE le serie, una figura per livello alpha
    def plot_var_grid(a):
        n = len(L.columns); nc = 4; nr = int(np.ceil(n / nc))
        # Aggiunto un po' di spazio extra in basso nel figsize (da nr*3 a nr*3.2) per la legenda
        fig, axes = plt.subplots(nr, nc, figsize=(nc * 5, nr * 3.2), sharex=True); axes = axes.ravel()
        for ax, c in zip(axes, L.columns):
            fc = pd.read_csv(f"{FC_DIR}/{c}.csv", index_col=0, parse_dates=True)
            ax.plot(fc.index, fc["ret"], color="lightgrey", lw=0.8, label="Returns")
            ax.plot(fc.index, fc[f"VaR_case1_{a}"], color="blue",       lw=1.0, label="Case 1 (Normal)")
            ax.plot(fc.index, fc[f"VaR_case3_{a}"], color="tab:green",  lw=1.8, label="Case 3 (GEV, no theta)")
            ax.plot(fc.index, fc[f"VaR_case4_{a}"], color="tab:orange", lw=1.0, ls="--", label="Case 4 (GEV, theta)")
            ax.plot(fc.index, fc[f"VaR_case5_{a}"], color="red",        lw=1.0, ls="-.", label="Case 5 (GPD)")
            ax.set_title(c, fontsize=9); ax.tick_params(labelsize=7)
            
        for ax in axes[n:]: ax.axis("off")
        
        # Spostiamo la legenda in basso al centro (sotto i grafici)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, 0.01))
        
        # Aggiustiamo il layout per lasciare spazio alla legenda in basso
        fig.suptitle(f"Fig E7a - OOS returns vs {int(a * 100)}% VaR, all series", y=0.98, fontsize=14)
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        fig.savefig(f"{FIG_DIR}/E7_VaR_OOS_all_{int(a * 100)}pct.png", dpi=150)

    for a in ALPHAS:
        plot_var_grid(a)

    # B) PLOT E7 (Solo EWP Focus): Grafico grande e dettagliato solo per il portafoglio rappresentativo
    for a in ALPHAS:
        target = "PORT"
        if os.path.exists(f"{FC_DIR}/{target}.csv"):
            fc_target = pd.read_csv(f"{FC_DIR}/{target}.csv", index_col=0, parse_dates=True)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            ax.plot(fc_target.index, fc_target["ret"], color="lightgrey", lw=1.5, label=f"{target} Returns")
            ax.plot(fc_target.index, fc_target[f"VaR_case1_{a}"], color="blue",       lw=1.2, label="Case 1 (Normal)")
            ax.plot(fc_target.index, fc_target[f"VaR_case3_{a}"], color="tab:green",  lw=2.0, label="Case 3 (GEV, no theta)")
            ax.plot(fc_target.index, fc_target[f"VaR_case4_{a}"], color="tab:orange", lw=1.5, ls="--", label="Case 4 (GEV, theta)")
            ax.plot(fc_target.index, fc_target[f"VaR_case5_{a}"], color="red",        lw=1.5, ls="-.", label="Case 5 (GPD)")
            
            ax.set_title(f"Fig E7b - Out-of-Sample {target} Returns vs {int(a * 100)}% VaR Models", fontsize=12)
            ax.set_ylabel("Returns / VaR (%)", fontsize=10)
            
            # Legenda in basso a sinistra dentro il grafico, per non coprire le linee
            ax.legend(fontsize=9, loc="lower left")
            
            fig.tight_layout()
            fig.savefig(f"{FIG_DIR}/E7_VaR_OOS_{target}_{int(a * 100)}pct.png", dpi=150)

    for a in ALPHAS:
        plot_var_grid(a)


if __name__ == "__main__":
    main()
