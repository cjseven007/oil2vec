# oil2vec pipeline (DataWorks Challenge 2026)

## Setup
    pip install -r requirements.txt
    # put the organizers' .xlsx files (01 ... 09) in  oil2vec/data/
    python run_all.py            # all 11 steps, ~10 min   (python run_all.py --skip-lstm  ~6 min)
    # (optional) other location:  export DWC_DATA_DIR=/path   |  Windows: set DWC_DATA_DIR=C:\path

Run from the project folder. Works with both `02 Field Production.xlsx` and `02_Field_Production.xlsx` naming.
Python 3.10+ (uses `dict | None` type hints).

## Modules (each runnable alone with `python -m src.<name>` -> easy to debug)
| file | job | output |
|---|---|---|
| `src/config.py` | paths, splits, weights, forecast dates | - |
| `src/load_data.py` | read files 01 + 02 (no cleaning) | prints shapes |
| `src/clean.py` | tidy sheets, sanity checks, interpolate all cases onto the 41 quarterly observed dates | `outputs/cache/*.npz` |
| `src/eda.py` | parameter design, curves, observed data, correlations, mismatch ranking | `outputs/figures`, `outputs/tables` |
| `src/embed.py` | PCA "oil2vec" embedding, scree, latent plots, which parameters are recoverable, k-means scan | figures + `identifiability.csv`, `latent_vectors.csv` |
| `src/proxy_baseline.py` | PCA+GP forward proxy, validation on cases 71-85, ABC history match, k-means top-10 | `proxy_validation.csv`, `posterior_summary.csv`, `top10_param_sets.csv` |
| `src/eda_advanced.py` | range vs effect, porosity-removed residuals, engineered rate features, Sobol sensitivity | figures 07-09, 10b, 11b |
| `src/match_v2.py` | ABC sanity check: cumulative-only vs cumulative+rates, tolerance sweep | `match_variants.csv` |
| `src/mcmc.py` | **MCMC history match** on cumulatives AND rates, with a learned model-discrepancy term (sigma per group) | `posterior_mcmc.npy`, `mcmc_posterior_summary.csv`, figs 30-32 |
| `src/proxy_compare.py` | GP vs MLP-ensemble vs LSTM-ensemble vs average, held-out error (`--skip-lstm` saves ~4 min) | `proxy_comparison.csv`, fig 40 |
| `src/active_design.py` | sequential design: does "simulate where unsure" beat random? + list of recommended new simulations | figs 60-61, `recommended_new_simulations.csv` |
| `src/forecast.py` | rate-decline forecast 2008-2028 with posterior band, back-tests, fills the organisers' template | figs 50-52, `outputs/09_Template_Deliverable_filled.xlsx`, `forecast_long_for_powerbi.csv`, `forecast_band.csv` |
| `src/load_grids.py` | OPTIONAL: convert the 250-330 MB grid workbooks (03-08) to small .npz | `outputs/cache/grids/` |

## Debug tips
* Run steps separately: `python run_all.py --steps 2` then `--steps 3`, etc. Steps 3-5 need step 2's cache.
* Every figure/table is produced by its own small function in `eda.py` / `embed.py`.
* Test cases 86-100 are never used for fitting; keep it that way until final reporting.

## Rate convention (important, easy to get wrong)
The observed rate printed at date t is the AVERAGE over the NEXT quarter (observed cumulative =
cumsum(rate x days)). The simulator's rate sheets are instantaneous at t. Compare like with like:
simulated rate = change in cumulative / days  (`clean.interval_rates`). Mixing them shifts everything by one
quarter and makes the 1998 ramp-up look like a huge error.

## Known limits (read before trusting the numbers)
* All curves stop at 2008-01-01. There are NO forecast labels. The forecast is an extrapolation checked by
  back-tests that hide only 2-4 years, not 20.
* Forecast uncertainty is dominated by the choice of decline-curve form (about 10% on oil), not by the
  parameter posterior (about 1%). `forecast_band.csv` has both the posterior band and the all-models envelope.
* The 10 template cases are near-identical curves, because the late-time trend is anchored to the observed
  history. Ask the organisers how the "10 best forecast cases" are judged.
* Dynamic grid properties are the same in every case (initial state).
* Fault transmissibility has no detectable effect on any output; the aquifer posterior piles toward its upper bound (200).
* Test cases 86-100 are never used for fitting.
