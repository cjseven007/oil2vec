"""STEP 9 - Which proxy is best?  (upgrade 5)

Three families of "cheap stand-ins for the simulator", all trained on cases 1-70 and scored on the
held-out cases 71-85 (cases 86-100 stay untouched):

  GP        PCA + Gaussian process        small-data specialist, smooth, gives uncertainty
  MLP-ens   PCA + 5 small neural nets     the 5 nets are averaged -> "ensemble"; their disagreement
                                          is a cheap uncertainty estimate
  LSTM-ens  5 LSTMs (as in paper 1)       reads the curve one time-step at a time; input at every
                                          step = the 4 parameters + the time
  ENSEMBLE  average of everything         usually beats each member alone

Run alone:  python -m src.proxy_compare
(the LSTM part is skipped automatically if PyTorch is not installed)
"""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from . import config as C
from . import embed as E
from . import load_data as L
from . import mcmc as M
from . import proxy_baseline as PB

try:
    import torch
    import torch.nn as nn

    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


# ---------------------------------------------------------------------------
# Model 1: PCA + GP  (re-use the class from proxy_baseline)
# ---------------------------------------------------------------------------
def fit_gp(theta_tr, X_tr, n_pca=8, n_pc=6):
    pca = E.fit_pca(X_tr, X_tr[:1], n_pca)[0]
    return PB.PCAGPProxy(n_pc).fit(theta_tr, X_tr, pca)


# ---------------------------------------------------------------------------
# Model 2: PCA + ensemble of MLPs
# ---------------------------------------------------------------------------
class PCAMLPEnsemble:
    def __init__(self, n_members=5, n_pca=8, hidden=(64, 64)):
        self.n_members, self.n_pca, self.hidden = n_members, n_pca, hidden

    def fit(self, theta, X):
        self.pca = E.fit_pca(X, X[:1], self.n_pca)[0]
        Z = self.pca.transform(X)
        self.sx, self.sz = MinMaxScaler().fit(theta), StandardScaler().fit(Z)
        t, z = self.sx.transform(theta), self.sz.transform(Z)
        self.nets = [
            MLPRegressor(
                hidden_layer_sizes=self.hidden,
                activation="tanh",
                solver="adam",
                learning_rate_init=3e-3,
                alpha=1e-5,
                max_iter=4000,
                tol=1e-9,
                n_iter_no_change=200,
                random_state=s,
            ).fit(t, z)
            for s in range(self.n_members)
        ]
        return self

    def members(self, theta):
        t = self.sx.transform(theta)
        return np.stack(
            [
                self.pca.inverse_transform(self.sz.inverse_transform(n.predict(t)))
                for n in self.nets
            ]
        )

    def predict_curves(self, theta):
        return self.members(theta).mean(axis=0)


# ---------------------------------------------------------------------------
# Model 3: LSTM ensemble (PyTorch)
# ---------------------------------------------------------------------------
if HAVE_TORCH:

    class _Net(nn.Module):
        def __init__(self, n_in, n_out, hidden=64):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, num_layers=2, batch_first=True)
            self.head = nn.Linear(hidden, n_out)

        def forward(self, x):
            return self.head(self.lstm(x)[0])


class LSTMEnsemble:
    """Output at every time step = [cum_oil, cum_wat, rate_oil, rate_wat]. Rates have one step fewer
    than cumulatives; the missing last step is masked out of the loss."""

    def __init__(self, lens, n_members=5, hidden=64, epochs=2500, lr=3e-3):
        self.lens, self.n_members, self.hidden, self.epochs, self.lr = (
            lens,
            n_members,
            hidden,
            epochs,
            lr,
        )
        self.T = max(lens)

    def _to_seq(self, X):
        """flat features (n, sum(lens)) -> (n, T, 4) with NaN padding"""
        out = np.full((len(X), self.T, len(self.lens)), np.nan)
        e = np.cumsum(self.lens)
        for b, (end, ln) in enumerate(zip(e, self.lens)):
            out[:, :ln, b] = X[:, end - ln : end]
        return out

    def _to_flat(self, Y):
        return np.hstack([Y[:, :ln, b] for b, ln in enumerate(self.lens)])

    def _inputs(self, theta):
        t = self.sx.transform(theta)
        tt = np.linspace(0, 1, self.T)
        return torch.tensor(
            np.concatenate(
                [
                    np.repeat(t[:, None, :], self.T, 1),
                    np.tile(tt[None, :, None], (len(t), 1, 1)),
                ],
                axis=2,
            ),
            dtype=torch.float32,
        )

    def fit(self, theta, X):
        self.sx = MinMaxScaler().fit(theta)
        Y = self._to_seq(X)
        self.mu = np.nanmean(Y, axis=0)  # mean curve per (time, block)
        self.sd = np.nanstd(Y, axis=0) + 1e-3  # spread per (time, block), floored
        Yn = (Y - self.mu) / self.sd
        mask = torch.tensor(~np.isnan(Yn), dtype=torch.float32)
        Yt = torch.tensor(np.nan_to_num(Yn), dtype=torch.float32)
        Xi = self._inputs(theta)
        self.nets = []
        for s in range(self.n_members):
            torch.manual_seed(s)
            net = _Net(Xi.shape[2], Yt.shape[2], self.hidden)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = (((net(Xi) - Yt) ** 2) * mask).sum() / mask.sum()
                loss.backward()
                opt.step()
                sched.step()
            self.nets.append(net.eval())
        return self

    def members(self, theta):
        Xi = self._inputs(theta)
        with torch.no_grad():
            outs = [n(Xi).numpy() * self.sd + self.mu for n in self.nets]
        return np.stack([self._to_flat(o) for o in outs])

    def predict_curves(self, theta):
        return self.members(theta).mean(axis=0)


class Average:
    """Average of several fitted proxies."""

    def __init__(self, models):
        self.models = models

    def predict_curves(self, theta):
        return np.mean([m.predict_curves(theta) for m in self.models], axis=0)


# ---------------------------------------------------------------------------
def score(model, theta, X, slices):
    P, rows = model.predict_curves(theta), {}
    for nm, sl in zip(M.BLOCK_NAMES, slices):
        T = X[:, sl]
        final = np.abs(P[:, sl][:, -1] - T[:, -1]) / np.abs(T[:, -1])
        nrmse = np.sqrt(((P[:, sl] - T) ** 2).mean(axis=1)) / np.abs(T[:, -1])
        rows[nm] = {
            "final_err_%": 100 * final.mean(),
            "curve_nRMSE_%": 100 * nrmse.mean(),
        }
    return pd.DataFrame(rows).T


def run_all(include_lstm=True):
    C.ensure_dirs()
    U = L.load_uncertainty()
    X, x_obs, dates, lens = M.build_data()
    theta = U[C.PARAMS].values
    tr, va = np.array(C.TRAIN_CASES) - 1, np.array(C.VAL_CASES) - 1
    like = M.Likelihood(None, x_obs, lens, (theta.min(0), theta.max(0)))
    models, times = {}, {}

    t = time.time()
    models["GP"] = fit_gp(theta[tr], X[tr])
    times["GP"] = time.time() - t
    t = time.time()
    models["MLP-ens"] = PCAMLPEnsemble().fit(theta[tr], X[tr])
    times["MLP-ens"] = time.time() - t
    if include_lstm and HAVE_TORCH:
        t = time.time()
        models["LSTM-ens"] = LSTMEnsemble(lens).fit(theta[tr], X[tr])
        times["LSTM-ens"] = time.time() - t
    elif include_lstm:
        print("PyTorch not installed -> LSTM skipped (pip install torch)")
    models["ENSEMBLE(all)"] = Average(list(models.values()))
    times["ENSEMBLE(all)"] = sum(times.values())

    res = {k: score(m, theta[va], X[va], like.slices) for k, m in models.items()}
    summary = pd.DataFrame({k: v["curve_nRMSE_%"] for k, v in res.items()}).T
    summary["mean"] = summary.mean(axis=1)
    summary["train_seconds"] = pd.Series(times)
    print(
        "Held-out error on validation cases 71-85 (curve nRMSE, % of the observed final value; lower = better):"
    )
    print(summary.round(3).to_string())
    print("\nFinal-value error (%) at the last time point:")
    print(
        pd.DataFrame({k: v["final_err_%"] for k, v in res.items()})
        .T.round(3)
        .to_string()
    )
    summary.to_csv(C.TABLE_DIR / "proxy_comparison.csv")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    summary.drop(columns=["mean", "train_seconds"]).plot.bar(ax=ax, rot=0)
    ax.set_yscale("log")
    ax.set_ylabel("curve nRMSE % (log scale)")
    ax.set_title("Held-out error by proxy model and output block")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "40_proxy_comparison.png", dpi=130)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    import sys

    pd.set_option("display.width", 200)
    run_all(include_lstm="--skip-lstm" not in sys.argv)
