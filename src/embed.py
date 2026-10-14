"""STEP 4 - "oil2vec": embed each case's history curves into a small latent vector.

Pipeline:  curves (gas,oil,wat x 41 times) --scale--> PCA --> z (n_comp numbers per case)
Then we ask the key question: HOW MUCH OF EACH PARAMETER CAN BE RECOVERED FROM z?
(cross-validated R2).  Parameters with R2 ~ 0 are NOT identifiable from history.

Run alone to test:   python -m src.embed
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import MinMaxScaler

from . import config as C
from . import clean as K
from . import load_data as L


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------
def build_features(cube: dict, obs: dict, vars_=None):
    """Stack variables side by side -> X (n_cases, n_vars*n_times).
    Each variable is divided by the OBSERVED final value so units are comparable
    (gas/oil/water all become ~1 at the end).  Returns X, x_obs (1, n_features)."""
    vars_ = vars_ or C.VARS
    X = np.hstack([cube[v] / obs[v][-1] for v in vars_])
    x_obs = np.hstack([obs[v] / obs[v][-1] for v in vars_])[None, :]
    return X, x_obs


def fit_pca(X, x_obs, n_components=6):
    pca = PCA(n_components=n_components, random_state=C.SEED).fit(X)
    return pca, pca.transform(X), pca.transform(x_obs)


def reconstruction_error(pca, X):
    """Relative error when X is compressed to n_components and decoded back."""
    Xr = pca.inverse_transform(pca.transform(X))
    return np.sqrt(((X - Xr) ** 2).mean()) / np.sqrt((X ** 2).mean())


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_scree(pca):
    ev = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(range(1, len(ev) + 1), ev, label="each")
    ax.plot(range(1, len(ev) + 1), np.cumsum(ev), "ko-", label="cumulative")
    ax.set_xlabel("PC"); ax.set_ylabel("explained variance"); ax.legend()
    ax.set_title("PCA scree (history curves)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "10_pca_scree.png", dpi=130); plt.close(fig)


def plot_latent(Z, z_obs, U, dims=(0, 1)):
    """Latent plane (PC1 vs PC2) coloured by each parameter; observed = black star."""
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, p in zip(axs, C.PARAMS):
        sc = ax.scatter(Z[:, dims[0]], Z[:, dims[1]], c=U[p], cmap="viridis", s=18)
        ax.scatter(z_obs[:, dims[0]], z_obs[:, dims[1]], marker="*", c="k", s=160, label="observed")
        ax.set_title(p); fig.colorbar(sc, ax=ax)
        ax.set_xlabel(f"PC{dims[0] + 1}"); ax.set_ylabel(f"PC{dims[1] + 1}")
    axs[0].legend()
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "11_latent_by_param.png", dpi=130); plt.close(fig)


# ---------------------------------------------------------------------------
# Identifiability: can z recover each parameter?  (the honest test of oil2vec)
# ---------------------------------------------------------------------------
def _gp():
    k = ConstantKernel(1.0) * Matern(length_scale=np.ones(1), nu=2.5) + WhiteKernel(1e-2)
    return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=2, random_state=C.SEED)


def identifiability(Z, U, n_comp_used=4):
    """5-fold CV R2 of predicting each parameter from the first n_comp_used latent dims."""
    Zs = MinMaxScaler().fit_transform(Z[:, :n_comp_used])
    rows = {}
    for p in C.PARAMS:
        gp = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * Matern(length_scale=np.ones(n_comp_used), nu=2.5) + WhiteKernel(1e-2),
            normalize_y=True, n_restarts_optimizer=2, random_state=C.SEED)
        pred = cross_val_predict(gp, Zs, U[p].values, cv=KFold(5, shuffle=True, random_state=C.SEED))
        y = U[p].values
        rows[p] = {"CV_R2": 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                   "CV_RMSE_pct_of_range": 100 * np.sqrt(((y - pred) ** 2).mean()) / (y.max() - y.min())}
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# k-means in latent space
# ---------------------------------------------------------------------------
def kmeans_scan(Z, ks=range(2, 9), n_comp_used=4):
    """Silhouette score per k: is there ANY natural cluster structure?"""
    rows = []
    for k in ks:
        lab = KMeans(k, n_init=10, random_state=C.SEED).fit_predict(Z[:, :n_comp_used])
        rows.append({"k": k, "silhouette": silhouette_score(Z[:, :n_comp_used], lab)})
    return pd.DataFrame(rows)


def pick_diverse(Z, score, k=10, n_comp_used=4):
    """Cluster candidates into k groups and return the index of the best-scoring
    member (lowest `score`) of each cluster -> k distinct, good cases."""
    lab = KMeans(k, n_init=10, random_state=C.SEED).fit_predict(Z[:, :n_comp_used])
    return [int(np.where(lab == c)[0][np.argmin(score[lab == c])]) for c in range(k)]


# ---------------------------------------------------------------------------
def run_all(n_components=6):
    C.ensure_dirs()
    U = L.load_uncertainty()
    cube, obs, dates = K.load_cache("cum")
    X, x_obs = build_features(cube, obs)
    pca, Z, z_obs = fit_pca(X, x_obs, n_components)

    ev = pd.DataFrame({"explained": pca.explained_variance_ratio_,
                       "cumulative": np.cumsum(pca.explained_variance_ratio_)},
                      index=[f"PC{i + 1}" for i in range(n_components)])
    print("PCA explained variance:\n", ev.round(5), "\n")
    for n in (1, 2, 3, 4):
        print(f"reconstruction error with {n} PCs: {100 * reconstruction_error(PCA(n).fit(X), X):.3f} %")
    plot_scree(pca); plot_latent(Z, z_obs, U)

    print("\nCan the latent vector recover each parameter? (5-fold CV)")
    idt = identifiability(Z, U)
    print(idt.round(3))
    idt.to_csv(C.TABLE_DIR / "identifiability.csv")

    print("\nk-means silhouette scan (higher = more natural clusters):")
    print(kmeans_scan(Z).round(3).to_string(index=False))

    pd.DataFrame(Z, index=U.index, columns=[f"z{i + 1}" for i in range(n_components)]).to_csv(
        C.TABLE_DIR / "latent_vectors.csv")
    return pca, Z, z_obs


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    run_all()
