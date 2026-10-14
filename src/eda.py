"""STEP 3 - Exploratory data analysis. Each function makes ONE figure/table, so
you can call or debug them individually.  Figures -> outputs/figures, tables -> outputs/tables.

Run alone to test:   python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from . import clean as K
from . import load_data as L

VAR_LABEL = {"gas": "Gas cum. [MSCF]", "oil": "Oil cum. [STB]", "wat": "Water cum. [STB]"}
PARAM_LABEL = {"fault_tran": "Fault transmissibility", "poro_mult": "Porosity multiplier",
               "perm_mult": "Permeability multiplier", "aq_pv": "Aquifer pore volume"}


def _save(fig, name):
    C.ensure_dirs()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"{name}.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
def plot_param_distributions(U: pd.DataFrame):
    """Histograms of the 4 parameters, coloured by split (train/val/test)."""
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, p in zip(axs, C.PARAMS):
        for sp, col in zip(["train", "val", "test"], ["#4c78a8", "#f58518", "#54a24b"]):
            ax.hist(U.loc[U.split == sp, p], bins=12, alpha=0.6, label=sp, color=col)
        ax.set_title(PARAM_LABEL[p], fontsize=10)
    axs[0].legend()
    _save(fig, "01_param_distributions")


def plot_param_pairs(U: pd.DataFrame):
    """Pairwise scatter - checks the design (Latin hypercube -> parameters ~independent)."""
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    for i, a in enumerate(C.PARAMS):
        for j, b in enumerate(C.PARAMS):
            ax = axs[i, j]
            if i == j:
                ax.hist(U[a], bins=12, color="#999")
            else:
                ax.scatter(U[b], U[a], s=8, c=U.split.map({"train": "#4c78a8", "val": "#f58518", "test": "#54a24b"}))
            if i == 3: ax.set_xlabel(C.PARAMS[j], fontsize=8)
            if j == 0: ax.set_ylabel(C.PARAMS[i], fontsize=8)
    _save(fig, "02_param_pairs")
    return U[C.PARAMS].corr(method="spearman")


def plot_curves(cube, obs, dates, U, var="oil", color_by="poro_mult"):
    """All 100 simulated curves coloured by one parameter, observed data in black."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sc = ax.scatter([], [], c=[], cmap="viridis")  # for colorbar
    norm = plt.Normalize(U[color_by].min(), U[color_by].max())
    cmap = plt.get_cmap("viridis")
    for i in range(C.N_CASES):
        ax.plot(dates, cube[var][i], color=cmap(norm(U[color_by].iloc[i])), lw=0.8, alpha=0.8)
    ax.plot(dates, obs[var], "k.", ms=4, label="observed")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=PARAM_LABEL[color_by])
    ax.set_title(f"{VAR_LABEL[var]} - 100 cases coloured by {color_by}")
    ax.legend()
    _save(fig, f"03_curves_{var}_by_{color_by}")


def plot_observed_overlay(obs_table: pd.DataFrame):
    """Observed field rates + water injection (what the field actually did)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for c in obs_table.columns:
        ax.plot(obs_table.index, obs_table[c], label=c)
    ax.set_title("Observed field production (quarterly)")
    ax.legend(fontsize=8)
    _save(fig, "04_observed_rates")


def correlation_with_final(cube, U):
    """Spearman correlation of each parameter with the FINAL cumulative value (2008-01-01)."""
    rows = {v: {p: pd.Series(cube[v][:, -1], index=U.index).corr(U[p], method="spearman")
                for p in C.PARAMS} for v in C.VARS}
    T = pd.DataFrame(rows).T
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(T.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(4), C.PARAMS, rotation=20); ax.set_yticks(range(3), T.index)
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{T.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    ax.set_title("Spearman: parameter vs cumulative at 2008-01-01")
    _save(fig, "05_param_vs_final_corr")
    T.to_csv(C.TABLE_DIR / "param_vs_final_corr.csv")
    return T


def mismatch_table(cube, obs, U, weights=None):
    """Normalised RMSE of each case vs observed, per variable and weighted total."""
    weights = weights or C.MISMATCH_WEIGHTS
    out = pd.DataFrame(index=U.index)
    for v in C.VARS:
        scale = obs[v][-1]
        out[f"nrmse_{v}"] = np.sqrt(((cube[v] - obs[v]) ** 2).mean(axis=1)) / scale
    out["total"] = sum(weights[v] * out[f"nrmse_{v}"] for v in C.VARS) / sum(weights.values())
    out = out.join(U)
    out["rank"] = out["total"].rank().astype(int)
    return out.sort_values("total")


def plot_mismatch_vs_params(M):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
    for ax, p in zip(axs, C.PARAMS):
        ax.scatter(M[p], M["total"], s=10, c=M.split.map({"train": "#4c78a8", "val": "#f58518", "test": "#54a24b"}))
        rho = M[p].corr(M["total"], method="spearman")
        ax.set_title(f"{p}  (rho={rho:.2f})", fontsize=10)
        ax.set_yscale("log")
    axs[0].set_ylabel("weighted nRMSE vs observed")
    _save(fig, "06_mismatch_vs_params")


def check_gas_oil_relation(cube):
    """Is gas just a scaled copy of oil? (then gas adds no extra information)"""
    ratio = cube["gas"][:, 1:] / np.maximum(cube["oil"][:, 1:], 1e-9)
    return pd.Series({"gas/oil ratio: mean": ratio.mean(), "gas/oil ratio: std across cases+time": ratio.std(),
                      "corr(final gas, final oil)": np.corrcoef(cube["gas"][:, -1], cube["oil"][:, -1])[0, 1]})


def summary_stats(cube, U):
    """Final-value summary table (min/mean/max over the 100 cases)."""
    d = {v: pd.Series(cube[v][:, -1]).describe()[["min", "mean", "max", "std"]] for v in C.VARS}
    return pd.DataFrame(d)


# ---------------------------------------------------------------------------
def run_all():
    C.ensure_dirs()
    U = L.load_uncertainty()
    obs_table = L.load_observed_table()
    cube, obs, dates = K.load_cache("cum")

    plot_param_distributions(U)
    print("Parameter Spearman correlations (should be ~0 for a space-filling design):")
    print(plot_param_pairs(U).round(2), "\n")
    for var in C.VARS:
        plot_curves(cube, obs, dates, U, var, "poro_mult")
    plot_curves(cube, obs, dates, U, "wat", "perm_mult")
    plot_curves(cube, obs, dates, U, "wat", "aq_pv")
    plot_observed_overlay(obs_table)
    print("Final-value summary:\n", summary_stats(cube, U).round(0), "\n")
    print("Param vs final-value correlation:\n", correlation_with_final(cube, U).round(2), "\n")
    print("Gas vs oil relation:\n", check_gas_oil_relation(cube).round(4), "\n")
    M = mismatch_table(cube, obs, U)
    M.to_csv(C.TABLE_DIR / "mismatch_ranking.csv")
    plot_mismatch_vs_params(M)
    print("Top-10 best history matches:\n", M.head(10)[["total", "split"] + C.PARAMS].round(4))
    print("\nFigures ->", C.FIG_DIR.resolve())


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    run_all()
