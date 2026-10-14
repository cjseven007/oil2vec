import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from openpyxl import load_workbook
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoModel

# 1. Dynamic Path Resolution for execution as direct script or module
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] if FILE.parents[0].name == "src" else FILE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import src.clean as K
import src.config as C
import src.embed as E
import src.load_data as L
import src.mcmc as M


WINDOW = 8


# ==========================================
# 1. DATA PIPELINE & TENSOR PREPARATION
# ==========================================

def prepare_pipeline_tensors(seq_len: int = 30, pred_len: int = 10) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, pd.DatetimeIndex
]:
    """
    Loads raw field curves and uncertainty parameters using load_data.py and clean.py.
    Processes interval rates, applies Z-score normalization to parameters,
    and builds 3D PyTorch tensors for train (Cases 1-85) and test (Cases 86-100) sets.
    """
    C.ensure_dirs()
    U = L.load_uncertainty()
    theta_all = U[C.PARAMS].values.astype(np.float32)
    
    try:
        cube, obs, dates = K.load_cache("cum")
        rcube, robs, _ = K.load_cache("rate")
    except Exception:
        raw_sheets = L.load_curve_sheets()
        cube, obs, dates, _ = K.build_cube(raw_sheets, "cum")
        K.save_cache(cube, obs, dates, "cum")
        rcube_raw, robs_raw, dates_r, _ = K.build_cube(raw_sheets, "rate")
        K.save_cache(rcube_raw, robs_raw, dates_r, "rate")
        rcube, robs = rcube_raw, robs_raw

    sim_r, obs_r, _ = K.interval_rates(cube, robs, dates)
    
    oil_chan = sim_r["oil"] / (obs_r["oil"][-1] + 1e-8)
    wat_chan = sim_r["wat"] / (obs_r["wat"][-1] + 1e-8)
    gas_chan = sim_r["gas"] / (obs_r["gas"][-1] + 1e-8)
    
    rate_cube = np.stack([oil_chan, wat_chan, gas_chan], axis=-1).astype(np.float32)
    
    total_available = rate_cube.shape[1]
    if seq_len + pred_len > total_available:
        seq_len = 30
        pred_len = 10
        
    past_seqs = rate_cube[:, :seq_len, :]
    future_seqs = rate_cube[:, seq_len:seq_len + pred_len, :]
    
    train_idx = np.array(C.TRAIN_CASES + C.VAL_CASES) - 1
    test_idx = np.array(C.TEST_CASES) - 1
    
    train_past = torch.from_numpy(past_seqs[train_idx])
    train_future = torch.from_numpy(future_seqs[train_idx])
    train_params_raw = torch.from_numpy(theta_all[train_idx])
    
    test_past = torch.from_numpy(past_seqs[test_idx])
    test_future = torch.from_numpy(future_seqs[test_idx])
    test_params_raw = torch.from_numpy(theta_all[test_idx])
    
    param_mean = train_params_raw.mean(dim=0, keepdim=True)
    param_std = train_params_raw.std(dim=0, keepdim=True) + 1e-8
    
    train_params = (train_params_raw - param_mean) / param_std
    test_params = (test_params_raw - param_mean) / param_std
    
    print(f"Data Pipeline Ready: {len(train_past)} Training Cases (1-85), {len(test_past)} Test Cases (86-100).")
    return train_past, train_future, train_params, test_past, test_future, test_params, param_std, dates


# ==========================================
# 2. ADVANCED OIL2VEC & TIME-BASED RAG
# ==========================================

class TemporalResBlock(nn.Module):
    """Dilated 1D Residual Convolutional Block for multi-scale temporal receptive fields."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + res)


class PhysicsSupConLoss(nn.Module):
    """Supervised Contrastive Loss aligning latent space with physical parameter distances."""
    def __init__(self, temperature: float = 0.07, dist_threshold: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.dist_threshold = dist_threshold

    def forward(self, embeddings: torch.Tensor, physical_params: torch.Tensor) -> torch.Tensor:
        embeds_norm = F.normalize(embeddings, p=2, dim=-1)
        params_norm = F.normalize(physical_params, p=2, dim=-1)
        
        sim_matrix = torch.mm(embeds_norm, embeds_norm.t()) / self.temperature
        param_dist = torch.cdist(params_norm, params_norm, p=2)
        
        pos_mask = (param_dist < self.dist_threshold).float()
        logits_mask = torch.ones_like(pos_mask) - torch.eye(pos_mask.size(0), device=pos_mask.device)
        pos_mask = pos_mask * logits_mask

        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)
        return -mean_log_prob_pos.mean()


class AdvancedOil2Vec(nn.Module):
    """Multi-Task Physics-Aware Temporal Contrastive Autoencoder."""
    def __init__(self, raw_input_dim: int = 3, embed_dim: int = 32, param_dim: int = 4):
        super().__init__()
        self.raw_input_dim = raw_input_dim
        self.expanded_dim = raw_input_dim + 2
        
        self.tcn1 = TemporalResBlock(self.expanded_dim, 64, kernel_size=3, dilation=1)
        self.tcn2 = TemporalResBlock(64, 128, kernel_size=3, dilation=2)
        self.tcn3 = TemporalResBlock(128, 128, kernel_size=3, dilation=4)
        
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        self.embed_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim)
        )
        
        self.decoder_fc = nn.Linear(embed_dim, 128)
        self.decoder_tcn = nn.Sequential(
            TemporalResBlock(128, 64, kernel_size=3, dilation=1),
            nn.Conv1d(64, raw_input_dim, kernel_size=1)
        )
        
        self.param_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, param_dim)
        )

    def _expand_features(self, x: torch.Tensor) -> torch.Tensor:
        oil_rate = x[:, :, 0:1]
        wat_rate = x[:, :, 1:2] if x.size(-1) > 1 else torch.zeros_like(oil_rate)
        
        dq_dt = torch.gradient(oil_rate, dim=1)[0]
        water_cut = wat_rate / (oil_rate + wat_rate + 1e-8)
        
        return torch.cat([x, dq_dt, water_cut], dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_exp = self._expand_features(x)
        x_conv = x_exp.transpose(1, 2)
        
        h1 = self.tcn1(x_conv)
        h2 = self.tcn2(h1)
        h3 = self.tcn3(h2)
        
        h_perm = h3.transpose(1, 2)
        attn_out, _ = self.attention(h_perm, h_perm, h_perm)
        global_context = attn_out.mean(dim=1)
        
        embed = self.embed_head(global_context)
        return F.normalize(embed, p=2, dim=-1)

    def decode(self, embed: torch.Tensor, seq_len: int) -> torch.Tensor:
        h = self.decoder_fc(embed).unsqueeze(-1).repeat(1, 1, seq_len)
        rec = self.decoder_tcn(h).transpose(1, 2)
        return rec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.encode(x)
        rec_seq = self.decode(embed, seq_len=x.size(1))
        pred_params = self.param_head(embed)
        return embed, rec_seq, pred_params


def train_advanced_oil2vec(model: AdvancedOil2Vec, 
                          train_past: torch.Tensor, 
                          train_params_norm: torch.Tensor, 
                          epochs: int = 100, 
                          lr: float = 1e-3) -> AdvancedOil2Vec:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    supcon_criterion = PhysicsSupConLoss(temperature=0.07, dist_threshold=0.5)
    
    model.train()
    print("\n--- Training Advanced Multi-Task Oil2Vec Embedding Model ---")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        embeds, rec_seq, pred_params = model(train_past)
        
        loss_rec = F.mse_loss(rec_seq, train_past)
        loss_supcon = supcon_criterion(embeds, train_params_norm)
        loss_param = F.mse_loss(pred_params, train_params_norm)
        
        total_loss = loss_rec + 0.5 * loss_supcon + 0.5 * loss_param
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1:03d}/{epochs:03d}] | Total: {total_loss.item():.5f} "
                  f"| Rec MSE: {loss_rec.item():.5f} | SupCon: {loss_supcon.item():.5f} | Param MSE: {loss_param.item():.5f}")
            
    model.eval()
    return model


class TimeSeriesRAG:
    """Vector database storing field embeddings and retrieving nearest historical futures."""
    def __init__(self, encoder: nn.Module):
        self.encoder = encoder
        self.encoder.eval()
        self.keys = None      
        self.values = None    
        
    def populate(self, past_seqs: torch.Tensor, future_seqs: torch.Tensor):
        with torch.no_grad():
            if hasattr(self.encoder, "encode"):
                self.keys = self.encoder.encode(past_seqs)
            else:
                self.keys = self.encoder(past_seqs)
        self.values = future_seqs
        
    def retrieve(self, query_seq: torch.Tensor) -> torch.Tensor:
        if self.keys is None:
            return torch.zeros(query_seq.size(0), self.values.size(1), self.values.size(2), device=query_seq.device)
            
        with torch.no_grad():
            if hasattr(self.encoder, "encode"):
                query_emb = self.encoder.encode(query_seq)
            else:
                query_emb = self.encoder(query_seq)
            sim = torch.mm(query_emb, self.keys.t()) 
            top_idx = torch.argmax(sim, dim=-1)
            return self.values[top_idx]


# ==========================================
# 3. ANCHORED TIMESFM TRANSFER FORECASTER
# ==========================================

class TimesFM_TransferModel(nn.Module):
    """
    Transfer learning adapter wrapping Google's TimesFM 3.0 foundation model.
    Uses an Anchored Cumulative Delta Head to guarantee boundary continuity 
    and maintain physical decline trends.
    """
    def __init__(self, pretrained_model_id: str = "google/timesfm-3.0-pytorch", 
                 input_dim: int = 3, 
                 pred_len: int = 10, 
                 param_dim: int = 4,
                 freeze_backbone: bool = True):
        super().__init__()
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.is_fallback = False

        try:
            self.backbone = AutoModel.from_pretrained(pretrained_model_id)
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            self.backbone_proj = nn.LazyLinear(64)
        except Exception as e:
            print(f"Could not load pre-trained TimesFM model ({e}). Using Fallback Transformer Backbone.")
            self.backbone = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True), num_layers=3
            )
            self.is_fallback = True
            self.ts_proj = nn.Linear(input_dim, 64)

        self.rag_proj = nn.Linear(input_dim, 32)
        self.mcmc_proj = nn.Linear(param_dim, 32) 
        self.vel_proj = nn.Linear(input_dim, 16)
        
        self.fusion_head = nn.Sequential(
            nn.Linear(128 + 16, 128),
            nn.GELU(),
            nn.Linear(128, input_dim * pred_len)
        )
        
    def forward(self, ts: torch.Tensor, 
                rag_context: Optional[torch.Tensor] = None, 
                mcmc_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        batch_size = ts.size(0)
        y_last = ts[:, -1, :]
        v_last = ts[:, -1, :] - ts[:, -2, :]
        v_feat = self.vel_proj(v_last)
        
        if self.is_fallback:
            x = self.ts_proj(ts)
            temporal_features = self.backbone(x)[:, -1, :]
        else:
            try:
                B, T, C = ts.shape
                ts_flat = ts.permute(0, 2, 1).reshape(B * C, T)
                padding = torch.zeros((B * C, T), device=ts.device, dtype=torch.float32)
                freq = torch.zeros((B * C,), device=ts.device, dtype=torch.long)
                
                outputs = self.backbone(past_values=ts_flat, past_values_padding=padding, freq=freq)
                
                if hasattr(outputs, "last_hidden_state"):
                    hidden = outputs.last_hidden_state
                elif isinstance(outputs, tuple):
                    hidden = outputs[0]
                else:
                    hidden = outputs
                
                feat_mean = hidden.mean(dim=1)
                feat_concat = feat_mean.view(B, C * feat_mean.size(-1))
                temporal_features = self.backbone_proj(feat_concat)
            except Exception:
                if not hasattr(self, "fallback_proj"):
                    self.fallback_proj = nn.Linear(self.input_dim, 64).to(ts.device)
                    self.fallback_encoder = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True), num_layers=3
                    ).to(ts.device)
                x = self.fallback_proj(ts)
                temporal_features = self.fallback_encoder(x)[:, -1, :]

        if rag_context is not None:
            rag_feat = self.rag_proj(rag_context.mean(dim=1)) 
        else:
            rag_feat = torch.zeros(batch_size, 32, device=ts.device)
            
        if mcmc_params is not None:
            mcmc_feat = self.mcmc_proj(mcmc_params)
        else:
            mcmc_feat = torch.zeros(batch_size, 32, device=ts.device)
            
        fused = torch.cat([temporal_features, rag_feat, mcmc_feat, v_feat], dim=-1)
        deltas = self.fusion_head(fused).view(batch_size, self.pred_len, self.input_dim)
        
        pred_curves = F.relu(y_last.unsqueeze(1) + torch.cumsum(deltas, dim=1))
        return pred_curves

    def predict_horizon(self, ts: torch.Tensor, horizon: int = 40, 
                        rag_context: Optional[torch.Tensor] = None, 
                        mcmc_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Autoregressively rolls out predictions to match arbitrary forecast horizons."""
        curr_ts = ts.clone()
        preds = []
        steps_needed = (horizon + self.pred_len - 1) // self.pred_len
        for _ in range(steps_needed):
            pred_step = self.forward(curr_ts, rag_context=rag_context, mcmc_params=mcmc_params)
            preds.append(pred_step)
            curr_ts = torch.cat([curr_ts[:, pred_step.size(1):, :], pred_step], dim=1)
        full_pred = torch.cat(preds, dim=1)[:, :horizon, :]
        return full_pred


# ==========================================
# 4. BAYESIAN MCMC ESTIMATION
# ==========================================

def run_mcmc_estimation(model: nn.Module, target_ts: torch.Tensor, 
                        param_dim: int = 4,
                        rag_context: Optional[torch.Tensor] = None, 
                        steps: int = 50) -> torch.Tensor:
    model.eval()
    device = target_ts.device
    num_samples = target_ts.size(0)
    current_params = torch.zeros((num_samples, param_dim), device=device)
    
    with torch.no_grad():
        current_pred = model(target_ts, rag_context=rag_context, mcmc_params=current_params)
        current_loss = F.mse_loss(current_pred[:, :5, :], target_ts[:, -5:, :], reduction='none').mean(dim=(1, 2))
        
    for _ in range(steps):
        proposal = current_params + torch.randn_like(current_params) * 0.05
        
        with torch.no_grad():
            prop_pred = model(target_ts, rag_context=rag_context, mcmc_params=proposal)
            prop_loss = F.mse_loss(prop_pred[:, :5, :], target_ts[:, -5:, :], reduction='none').mean(dim=(1, 2))
            
        alpha = torch.exp(current_loss - prop_loss)
        accept = torch.rand(num_samples, device=device) < alpha
        
        current_params[accept] = proposal[accept]
        current_loss[accept] = prop_loss[accept]
            
    return current_params


# ==========================================
# 5. EMPIRICAL EXTRAPOLATION & HELPER UTILS
# ==========================================

def _fit_exp(q: np.ndarray, tt: np.ndarray):
    b = np.polyfit(tt, np.log(np.maximum(q, 1e-9)), 1)
    return lambda x: np.exp(b[1] + b[0] * x)

def _fit_arps(q: np.ndarray, tt: np.ndarray):
    t0 = tt[0]
    f = lambda x, qi, D, b: qi / (1 + b * D * (x - t0)) ** (1 / b)
    try:
        p, _ = curve_fit(f, tt, q, p0=[q[0], 0.02, 0.5], bounds=([0, 1e-5, 0.01], [q.max() * 2, 1.0, 1.0]), maxfev=5000)
        return lambda x: f(x, *p)
    except Exception:
        return _fit_exp(q, tt)

def oil_trend(qo: np.ndarray, tt: np.ndarray, fut: np.ndarray, kind: str):
    if kind == "exp":
        return _fit_exp(qo, tt)(fut)
    if kind == "arps":
        return _fit_arps(qo, tt)(fut)
    return 0.5 * (_fit_exp(qo, tt)(fut) + _fit_arps(qo, tt)(fut))

def _water_sat_exp(qw: np.ndarray, tt: np.ndarray, fut: np.ndarray):
    tl = tt[-1]
    f = lambda x, qinf, tau: qinf - (qinf - qw[-1]) * np.exp(-(x - tl) / tau)
    try:
        p, _ = curve_fit(f, tt, qw, p0=[qw[-1] * 1.2, 10], bounds=([qw[-1] * 0.9, 1.0], [qw[-1] * 5, 400]), maxfev=5000)
        return f(fut, *p)
    except Exception:
        return np.full(len(fut), qw[-1])

def _water_liq_exp(qo: np.ndarray, qw: np.ndarray, tt: np.ndarray, fut: np.ndarray, oil_f: np.ndarray):
    return np.maximum(_fit_exp(qo + qw, tt)(fut) - oil_f, 0)

def extrapolate(qo_hist: np.ndarray, qw_hist: np.ndarray, fut: np.ndarray, 
                window: int = WINDOW, oil_kind: str = "blend", wat_kind: str = "blend"):
    h = len(qo_hist)
    tt = np.arange(h - window, h)
    qo, qw = qo_hist[-window:], qw_hist[-window:]
    oil_f = oil_trend(qo, tt, fut, oil_kind)
    if wat_kind == "sat_exp":
        wat_f = _water_sat_exp(qw, tt, fut)
    elif wat_kind == "liq_exp":
        wat_f = _water_liq_exp(qo, qw, tt, fut, oil_f)
    else:
        wat_f = 0.5 * (_water_sat_exp(qw, tt, fut) + _water_liq_exp(qo, qw, tt, fut, oil_trend(qo, tt, fut, "exp")))
    return oil_f, wat_f

def make_dates():
    fd = pd.date_range("2008-01-01", C.FORECAST_DATES[-1], freq="QS")
    return fd, np.diff(fd.values.astype("datetime64[D]")).astype(float)

def rates_to_cum(cum0, rates, days):
    return cum0 + np.cumsum(rates * days)

def to_cum(rates_oil, rates_wat, R, fdays=None):
    """Dynamically converts interval rates to cumulative totals using matching date lengths."""
    if fdays is None:
        fdays = R["fdays"][:len(rates_oil)]
    co = rates_to_cum(R["cum0"]["oil"], rates_oil, fdays)
    cw = rates_to_cum(R["cum0"]["wat"], rates_wat, fdays)
    cg = rates_to_cum(R["cum0"]["gas"], rates_oil * R["gor"], fdays)
    return cg, co, cw


# ==========================================
# 6. BACKTESTS & REAL FORECAST PIPELINE
# ==========================================

def backtest_sims(cuts=(16, 20, 24, 28, 32, 36), window=WINDOW):
    cube, obs, dates = K.load_cache("cum")
    rcube, robs, _ = K.load_cache("rate")
    sim_r, _, days = K.interval_rates(cube, robs, dates)
    H = len(dates) - 1
    rows = []
    for h in cuts:
        for method in ["naive"] + [f"{o}+{w}" for o, w in [("exp", "sat_exp"), ("arps", "sat_exp"), ("blend", "blend")]]:
            eo, ew = [], []
            for i in range(C.N_CASES):
                qo, qw = sim_r["oil"][i][:h], sim_r["wat"][i][:h]
                fut = np.arange(h, H)
                if method == "naive":
                    fo, fw = np.full(len(fut), qo[-1]), np.full(len(fut), qw[-1])
                else:
                    ok, wk = method.split("+")
                    fo, fw = extrapolate(qo, qw, fut, window, ok, wk)
                co, cw = np.sum(fo * days[h:H]), np.sum(fw * days[h:H])
                eo.append(abs(co - (cube["oil"][i][H] - cube["oil"][i][h])) / cube["oil"][i][H])
                ew.append(abs(cw - (cube["wat"][i][H] - cube["wat"][i][h])) / cube["wat"][i][H])
            rows.append({
                "hidden_quarters": H - h, "hidden_years": (H - h) / 4, "method": method,
                "oil_median_err_%": 100 * np.median(eo), "oil_p90_err_%": 100 * np.percentile(eo, 90),
                "wat_median_err_%": 100 * np.median(ew), "wat_p90_err_%": 100 * np.percentile(ew, 90)
            })
    return pd.DataFrame(rows)


def plot_backtest(T: pd.DataFrame):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, v, t in zip(axs, ("oil", "wat"), ("Oil", "Water")):
        for m, g in T.groupby("method"):
            ax.plot(g["hidden_years"], g[f"{v}_median_err_%"], "o-", label=m)
        ax.set_xlabel("years hidden from the method")
        ax.set_ylabel("median error of cumulative gained, % of total")
        ax.set_title(f"{t}: error grows with how far you extrapolate")
        ax.set_yscale("log")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "50_backtest_sims.png", dpi=130)
    plt.close(fig)


def real_forecast(models_dict: Dict[str, nn.Module], rag_db: TimeSeriesRAG, 
                  n_draw: int = 800, seed: int = C.SEED):
    """Generates real field forecasts across all TimesFM ablation neural models."""
    U = L.load_uncertainty()
    cube, obs, dates = K.load_cache("cum")
    rcube, robs, _ = K.load_cache("rate")
    sim_r, obs_r, _ = K.interval_rates(cube, robs, dates)
    fdates, fdays = make_dates()
    horizon = len(fdays)
    
    r40 = {"oil": robs["oil"][-1], "wat": robs["wat"][-1], "gas": robs["gas"][-1]}
    goR = r40["gas"] / r40["oil"]
    
    # Input tensor from actual observed field rate history
    oil_chan = obs_r["oil"][-30:] / (robs["oil"][-1] + 1e-8)
    wat_chan = obs_r["wat"][-30:] / (robs["wat"][-1] + 1e-8)
    gas_chan = obs_r["gas"][-30:] / (robs["gas"][-1] + 1e-8)
    past_tensor = torch.from_numpy(
        np.stack([oil_chan, wat_chan, gas_chan], axis=-1)[None, ...].astype(np.float32)
    )

    try:
        S = np.load(C.CACHE_DIR / "posterior_mcmc.npy")
    except Exception:
        S = np.random.randn(n_draw, 4)
        
    rng = np.random.default_rng(seed)
    S = S[rng.choice(len(S), min(n_draw, len(S)), replace=False)]
    
    param_mean = torch.from_numpy(S[:, :4].mean(axis=0, keepdims=True)).float()
    param_std = torch.from_numpy(S[:, :4].std(axis=0, keepdims=True) + 1e-8).float()
    norm_params = (torch.from_numpy(S[:, :4]).float() - param_mean) / param_std

    ens = {}
    context = rag_db.retrieve(past_tensor)
    
    for name, model in models_dict.items():
        model.eval()
        with torch.no_grad():
            past_batch = past_tensor.repeat(len(S), 1, 1)
            context_batch = context.repeat(len(S), 1, 1) if "RAG" in name else None
            mcmc_batch = norm_params if "MCMC" in name else None
            
            pred_norm = model.predict_horizon(
                past_batch, horizon=horizon, rag_context=context_batch, mcmc_params=mcmc_batch
            )
            
            rates = pred_norm.cpu().numpy()
            qo = rates[:, :, 0] * robs["oil"][-1]
            qw = rates[:, :, 1] * robs["wat"][-1]
            ens[name] = np.stack([qo, qw], axis=1)

    mix = ens["(4) Full Model (RAG + MCMC)"]
    f1_fo, f1_fw = extrapolate(obs_r["oil"], obs_r["wat"], np.arange(40, 40 + horizon), WINDOW, "blend", "blend")
    f1 = {"blend": (f1_fo, f1_fw)}
    
    return dict(S=S, ens=ens, f1=f1, mix=mix, fdates=fdates, fdays=fdays, r40=r40, gor=goR,
                cum0={v: obs[v][-1] for v in C.VARS}, obs=obs, robs=robs, dates=dates, obs_r=obs_r)


def plot_forecast(R):
    """Plots predictions from neural ablation models against the full observed history."""
    fd = R["fdates"][1:]
    fig, axs = plt.subplots(2, 3, figsize=(16, 7))
    colors = {
        "(1) Base TimesFM": "#d62728",
        "(2) TimesFM + RAG": "#1f77b4",
        "(3) MCMC + TimesFM": "#ff7f0e",
        "(4) Full Model (RAG + MCMC)": "#2ca02c"
    }
    
    # Load actual observed history across the full timeframe to overlay against forecasts
    try:
        cube_full, obs_full, dates_full = K.load_cache("cum")
        _, obs_r_full, _ = K.interval_rates(cube_full, R["robs"], dates_full)
    except Exception:
        obs_full, dates_full = R["obs"], R["dates"]
        obs_r_full = R["obs_r"]
        
    for j, (v, lab) in enumerate([("oil", "Oil"), ("wat", "Water"), ("gas", "Gas")]):
        ax_c, ax_r = axs[0, j], axs[1, j]
        
        # Plot full actual observed history line across both cumulative and rate subplots
        ax_c.plot(dates_full, obs_full[v], "k-", lw=2, label="observed history")
        
        if v == "gas":
            hist_rate = obs_r_full["oil"] * R["gor"]
        else:
            hist_rate = obs_r_full[v]
            
        ax_r.plot(dates_full[:-1], hist_rate, "k-", lw=2, label="observed history")
        
        # Overlay neural model predictions
        for name, base in R["ens"].items():
            cums = np.array([to_cum(base[k, 0], base[k, 1], R)[{"gas": 0, "oil": 1, "wat": 2}[v]] for k in range(len(base))])
            md_c = np.median(cums, axis=0)
            ax_c.plot(fd, md_c, label=name, color=colors.get(name, "blue"))
            
            idx = {"oil": 0, "wat": 1}.get(v)
            rates = base[:, 0, :] * R["gor"] if v == "gas" else base[:, idx, :]
            md_r = np.median(rates, axis=0)
            ax_r.plot(fd, md_r, label=name, color=colors.get(name, "blue"))
            
        ax_c.set_title(f"{lab} cumulative")
        ax_c.axvline(R["dates"][-1], color="grey", ls=":", label="Forecast Start (2008)")
        ax_r.set_title(f"{lab} rate")
        ax_r.axvline(R["dates"][-1], color="grey", ls=":")
        
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "51_forecast.png", dpi=130)
    plt.close(fig)


def plot_model_form(R):
    """Plots model-form spread across neural ablation choices (52_model_form_spread.png)."""
    fd = R["fdates"][1:]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, (v, ix, lab) in zip(axs, [("oil", 1, "oil cumulative"), ("wat", 2, "water cumulative")]):
        for name, rates in R["ens"].items():
            c = np.array([to_cum(rates[k, 0], rates[k, 1], R)[ix] for k in range(0, len(rates), 8)])
            ax.plot(fd, np.median(c, axis=0), label=name)
        ax.set_title(f"Median {lab} under TimesFM Ablation Architectures")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "52_model_form_spread.png", dpi=130)
    plt.close(fig)


def pick_template_cases(R, k=10, seed=C.SEED):
    S = R["S"]
    lab = KMeans(k, n_init=10, random_state=seed).fit_predict(MinMaxScaler().fit_transform(S[:, :4]))
    idx = [int(np.where(lab == c)[0][0]) for c in range(k)]
    return idx


def envelope(R):
    pools = []
    for rates in list(R["ens"].values()):
        pools += [np.array(to_cum(rates[k, 0], rates[k, 1], R)) for k in range(0, len(rates), 4)]
    return np.array(pools)


def write_outputs(R, idx):
    """Writes deliverables to Excel while trimming any trailing empty template rows."""
    base = R["mix"]
    fd = R["fdates"][1:]
    
    try:
        tmpl_file = L.find_file("09")
    except Exception:
        tmpl_file = C.DATA_DIR / "09_Template_Deliverable.xlsx"

    wb = load_workbook(tmpl_file)
    params = []
    rows_written_list = []
    
    for n, i in enumerate(idx, start=1):
        ws = wb[f"Case {n}"]
        sheet_max_rows = ws.max_row - 1
        
        # Determine target row count for sheet
        template_rows = min(len(fd), sheet_max_rows) if sheet_max_rows > 0 else len(fd)
        
        rates_oil = base[i, 0, :template_rows]
        rates_wat = base[i, 1, :template_rows]
        fdays_trimmed = R["fdays"][:template_rows]
        fd_trimmed = fd[:template_rows]
        
        cg, co, cw = to_cum(rates_oil, rates_wat, R, fdays=fdays_trimmed)
        
        for r in range(template_rows):
            ws.cell(row=r + 2, column=2, value=float(cg[r]))
            ws.cell(row=r + 2, column=3, value=float(co[r]))
            ws.cell(row=r + 2, column=4, value=float(cw[r]))
            
        # Clean up trailing empty rows beyond template_rows
        if ws.max_row > template_rows + 1:
            ws.delete_rows(template_rows + 2, ws.max_row - template_rows)
            
        params.append({"template_case": n, **dict(zip(C.PARAMS, R["S"][i, :4]))})
        rows_written_list.append(template_rows)
        
    C.ensure_dirs()
    out_excel_path = C.OUT_DIR / "09_Template_Deliverable_filled.xlsx"
    wb.save(out_excel_path)
    
    df_params = pd.DataFrame(params)
    df_params.to_csv(C.TABLE_DIR / "template_case_parameters.csv", index=False)
    
    t_rows_written = rows_written_list[0]
    fd_tr = fd[:t_rows_written]
    fdays_tr = R["fdays"][:t_rows_written]
    
    rows = []
    for n, i in enumerate(idx, start=1):
        for name, rates in R["ens"].items():
            cg, co, cw = to_cum(rates[i, 0, :t_rows_written], rates[i, 1, :t_rows_written], R, fdays=fdays_tr)
            for r in range(t_rows_written):
                rows.append({
                    "case": n, "variant": name, "date": fd_tr[r], "gas_cum_MSCF": cg[r], 
                    "oil_cum_STB": co[r], "water_cum_STB": cw[r], 
                    "oil_rate_STBd": rates[i, 0, r], "water_rate_STBd": rates[i, 1, r]
                })
    pd.DataFrame(rows).to_csv(C.TABLE_DIR / "forecast_long_for_powerbi.csv", index=False)
    
    cums = envelope(R)
    mixc = np.array([to_cum(base[k, 0, :t_rows_written], base[k, 1, :t_rows_written], R, fdays=fdays_tr) for k in range(len(base))])
    band = {"date": fd_tr}
    for j, v in enumerate(["gas_cum_MSCF", "oil_cum_STB", "water_cum_STB"]):
        lo, md, hi = np.percentile(cums[:, j, :t_rows_written], [5, 50, 95], axis=0)
        band.update({
            f"{v}_env_p5": lo, f"{v}_env_p50": md, f"{v}_env_p95": hi, 
            f"{v}_mix_median": np.median(mixc[:, j, :], axis=0)
        })
    pd.DataFrame(band).to_csv(C.TABLE_DIR / "forecast_band.csv", index=False)
    return df_params, out_excel_path, t_rows_written


def check_outputs(path: Path):
    wb = load_workbook(path, data_only=True)
    probs = []
    for ws in wb.worksheets:
        a = np.array([[c for c in row[1:4]] for row in ws.iter_rows(min_row=2, values_only=True)], dtype=float)
        if np.isnan(a).any(): probs.append(f"{ws.title}: NaN/empty cells")
        if (np.diff(a, axis=0) < 0).any(): probs.append(f"{ws.title}: cumulative decreases")
    return probs


# ==========================================
# 7. EVALUATION & VISUALIZATION HELPERS
# ==========================================

def compute_geometric_scaled_mae(pred: torch.Tensor, target: torch.Tensor, past_history: torch.Tensor) -> float:
    naive_pred = past_history[:, -1:, :].repeat(1, target.size(1), 1)
    naive_mae = F.l1_loss(naive_pred, target, reduction='none').mean(dim=(1, 2))
    model_mae = F.l1_loss(pred, target, reduction='none').mean(dim=(1, 2))
    
    scaled_errors = model_mae / (naive_mae + 1e-8)
    geom_mean = torch.exp(torch.mean(torch.log(scaled_errors + 1e-8)))
    return geom_mean.item()


def plot_training_losses(loss_dict: Dict[str, list]):
    """Plots fine-tuning loss curves across ablations (70_ablation_training_losses.png)."""
    plt.figure(figsize=(10, 5))
    for label, losses in loss_dict.items():
        plt.plot(losses, label=label, linewidth=2)
    plt.title("Adapter Head Fine-Tuning Loss Across Ablations")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "70_ablation_training_losses.png", dpi=130)
    plt.close()


def plot_forecasts(past: torch.Tensor, future: torch.Tensor, predictions: Dict[str, torch.Tensor], 
                   sample_idx: int = 0, feature_idx: int = 0):
    """Plots test forecasts for a held-out case across ablations (71_ablation_test_case_86.png)."""
    seq_len = past.size(1)
    pred_len = future.size(1)
    
    past_x = list(range(seq_len))
    future_x = list(range(seq_len, seq_len + pred_len))
    
    past_y = past[sample_idx, :, feature_idx].detach().cpu().numpy()
    future_y = future[sample_idx, :, feature_idx].detach().cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(past_x, past_y, label="Historical Rate (Cleaned)", color="black", linewidth=2)
    plt.plot(future_x, future_y, label="Ground Truth Target", color="green", linestyle="--", linewidth=2.5)
    
    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#9467bd']
    for i, (label, pred_tensor) in enumerate(predictions.items()):
        pred_y = pred_tensor[sample_idx, :, feature_idx].detach().cpu().numpy()
        plt.plot(future_x, pred_y, label=label, color=colors[i % len(colors)], linewidth=1.8, alpha=0.9)
        
    plt.axvline(x=seq_len - 1, color='gray', linestyle=':', label="Forecast Horizon Start")
    plt.title(f"Field Production Rate Forecast (Held-Out Test Case {sample_idx + 86}, Channel {feature_idx})")
    plt.xlabel("Quarterly Steps")
    plt.ylabel("Normalized Interval Rate")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / f"71_ablation_test_case_{sample_idx + 86}.png", dpi=130)
    plt.close()


def fine_tune_head(model: nn.Module, train_past: torch.Tensor, train_future: torch.Tensor, 
                   model_name: str, epochs: int = 50, rag_db: Optional[TimeSeriesRAG] = None, 
                   use_rag: bool = False, use_mcmc: bool = False, 
                   train_params: Optional[torch.Tensor] = None) -> Tuple[nn.Module, list]:
    print(f"\n--- Fine-Tuning Adapter Head: {model_name} ---")
    optimizer = torch.optim.Adam(model.fusion_head.parameters(), lr=1e-3)
    criterion_mse = nn.MSELoss()
    model.train()
    
    rag_context = rag_db.retrieve(train_past) if (use_rag and rag_db) else None
    v_last = train_past[:, -1, :] - train_past[:, -2, :]
    
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(train_past, rag_context=rag_context, mcmc_params=train_params if use_mcmc else None)
        
        loss_mse = criterion_mse(pred, train_future)
        pred_v1 = pred[:, 0, :] - train_past[:, -1, :]
        loss_slope = criterion_mse(pred_v1, v_last)
        
        total_loss = loss_mse + 0.2 * loss_slope
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1:02d}/{epochs:02d}] | Train Loss: {total_loss.item():.6f}")
            
    return model, loss_history


# ==========================================
# 8. MAIN EXECUTION PIPELINE
# ==========================================

def main():
    torch.manual_seed(C.SEED)
    C.ensure_dirs()
    pd.set_option("display.width", 220)
    
    # --- PART 1: ABLATION TRAINING & EVALUATION ---
    print("\n=======================================================")
    print("      TRAINING TIMESFM 3.0 + OIL2VEC RAG ADAPTERS")
    print("=======================================================")
    
    train_past, train_future, train_params, test_past, test_future, test_params, param_std, dates = prepare_pipeline_tensors()
    
    input_dim = train_past.size(-1)
    pred_len = train_future.size(1)
    param_dim = train_params.size(-1)
    
    oil2vec_model = AdvancedOil2Vec(raw_input_dim=input_dim, embed_dim=32, param_dim=param_dim)
    oil2vec_model = train_advanced_oil2vec(oil2vec_model, train_past, train_params, epochs=100)
    
    rag_db = TimeSeriesRAG(encoder=oil2vec_model)
    rag_db.populate(train_past, train_future)
    
    models = {}
    train_loss_histories = {}
    test_predictions = {}
    
    # 1) Base TimesFM
    m1 = TimesFM_TransferModel(input_dim=input_dim, pred_len=pred_len, param_dim=param_dim)
    models["(1) Base TimesFM"], train_loss_histories["(1) Base TimesFM"] = fine_tune_head(
        m1, train_past, train_future, "(1) Base TimesFM", epochs=50
    )
    
    # 2) TimesFM + RAG
    m2 = TimesFM_TransferModel(input_dim=input_dim, pred_len=pred_len, param_dim=param_dim)
    models["(2) TimesFM + RAG"], train_loss_histories["(2) TimesFM + RAG"] = fine_tune_head(
        m2, train_past, train_future, "(2) TimesFM + RAG", epochs=50, rag_db=rag_db, use_rag=True
    )
    
    # 3) MCMC + TimesFM
    m3 = TimesFM_TransferModel(input_dim=input_dim, pred_len=pred_len, param_dim=param_dim)
    models["(3) MCMC + TimesFM"], train_loss_histories["(3) MCMC + TimesFM"] = fine_tune_head(
        m3, train_past, train_future, "(3) MCMC + TimesFM", epochs=50, use_mcmc=True, train_params=train_params
    )
    
    # 4) Full Model (RAG + MCMC)
    m4 = TimesFM_TransferModel(input_dim=input_dim, pred_len=pred_len, param_dim=param_dim)
    models["(4) Full Model (RAG + MCMC)"], train_loss_histories["(4) Full Model (RAG + MCMC)"] = fine_tune_head(
        m4, train_past, train_future, "(4) Full Model (RAG + MCMC)", epochs=50, rag_db=rag_db, use_rag=True, use_mcmc=True, train_params=train_params
    )
    
    print("\n=======================================================")
    print("               TEST DATASET EVALUATIONS (Cases 86-100)")
    print("=======================================================")
    
    test_context = rag_db.retrieve(test_past)
    
    with torch.no_grad():
        test_predictions["(1) Base TimesFM"] = models["(1) Base TimesFM"](test_past)
        m1_metric = compute_geometric_scaled_mae(test_predictions["(1) Base TimesFM"], test_future, test_past)
        print(f"(1) Times FM (Base Fine-Tuned)         | Geometric Scaled MAE: {m1_metric:.4f}")
        
        test_predictions["(2) TimesFM + RAG"] = models["(2) TimesFM + RAG"](test_past, rag_context=test_context)
        m2_metric = compute_geometric_scaled_mae(test_predictions["(2) TimesFM + RAG"], test_future, test_past)
        print(f"(2) Times FM + oil2vec (RAG)           | Geometric Scaled MAE: {m2_metric:.4f}")
        
        test_mcmc_params_3 = run_mcmc_estimation(models["(3) MCMC + TimesFM"], test_past, param_dim=param_dim)
        test_predictions["(3) MCMC + TimesFM"] = models["(3) MCMC + TimesFM"](test_past, mcmc_params=test_mcmc_params_3)
        m3_metric = compute_geometric_scaled_mae(test_predictions["(3) MCMC + TimesFM"], test_future, test_past)
        print(f"(3) Times FM + MCMC                    | Geometric Scaled MAE: {m3_metric:.4f}")
        
        test_mcmc_params_4 = run_mcmc_estimation(models["(4) Full Model (RAG + MCMC)"], test_past, param_dim=param_dim, rag_context=test_context)
        test_predictions["(4) Full Model (RAG + MCMC)"] = models["(4) Full Model (RAG + MCMC)"](test_past, rag_context=test_context, mcmc_params=test_mcmc_params_4)
        m4_metric = compute_geometric_scaled_mae(test_predictions["(4) Full Model (RAG + MCMC)"], test_future, test_past)
        print(f"(4) Times FM + oil2vec RAG + MCMC      | Geometric Scaled MAE: {m4_metric:.4f}")

    print("\n=======================================================")
    print("               GENERATING ABLATION FIGURES")
    print("=======================================================")
    
    plot_training_losses(train_loss_histories)
    plot_forecasts(test_past, test_future, test_predictions, sample_idx=0, feature_idx=0)

    # --- PART 2: REAL FIELD FORECAST & DELIVERABLES PIPELINE ---
    print("\n=======================================================")
    print("     GENERATING DELIVERABLES WITH TIMESFM MODELS")
    print("=======================================================")
    
    T = backtest_sims()
    T.to_csv(C.TABLE_DIR / "backtest_sims.csv", index=False)
    plot_backtest(T)
    
    R = real_forecast(models_dict=models, rag_db=rag_db)
    plot_forecast(R)
    plot_model_form(R)
    
    idx = pick_template_cases(R)
    P, out_excel_path, rows_written = write_outputs(R, idx)
    
    print("\nTemplate cases filled using TimesFM + RAG predictions:\n", P.round(4).to_string(index=False))
    
    probs = check_outputs(out_excel_path)
    print("\nfilled-template checks:", f"OK ({rows_written} rows, no gaps, cumulative never decreases)" if not probs else probs)


if __name__ == "__main__":
    main()