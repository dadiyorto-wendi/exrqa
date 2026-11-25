# exrqa: extended recurrence plots and quantification for noisy continuous systems

A small, numpy-only Python implementation of

> D. Wendi & N. Marwan (2018): *Extended recurrence plot and quantification for noisy continuous dynamical systems.* Chaos **28**, 085722. [doi:10.1063/1.5025485](https://doi.org/10.1063/1.5025485)

It comes with executed example notebooks for the Lorenz, Rössler, Duffing and Mackey–Glass systems.

## What the paper is about

A recurrence plot (RP) marks the times when a trajectory returns close to an earlier state. Recurrence quantification analysis (RQA) turns it into measures such as **determinism (DET)**, the fraction of recurrences that form diagonal lines. For continuous systems measured with noise, both steps fail:

1. **The RP fills with artefacts.** Noise adds many shallow local minima to every distance profile. Threshold-based and local-minima RPs then show block clusters and vertical lines that are not part of the dynamics.
2. **DET collapses.** Noise breaks diagonal lines apart and shifts them sideways, so DET falls towards the value of noise even when the dynamics are deterministic.

The paper proposes one remedy for each problem:

| | remedy | idea | `exrqa` |
|---|---|---|---|
| RP | **LM2P** | Keep only local minima of each distance profile that are below ε **and** at least τm samples apart. τm lies between the spacing of noise-induced minima and the decorrelation time of the signal. | `local_minima_rp(D, eps, tau_m)` |
| measure | **DDET** | Slide a window of `ws` diagonals along the RP and count a line as connected across gaps of up to `d_max` samples. DDET = ln(⟨DD_w⟩ / ⟨RR_w⟩). | `ddet(R, ws, tau_w, d_max)` |

On the Lorenz system with 20 dB noise, the LM2P plot stays close to the noise-free one. In a parameter sweep, DDET still finds both periodic windows, where DET does not.

## What this repository adds

* `exrqa`, a reusable package: RPs (threshold, LM, LMT, LM2P), RQA, DDET, the τm rule and the test systems.
* **Matched noise baselines.** The white-noise DDET of ≈ −1 holds only for the paper's settings. `ddet_iid` predicts it for other settings, and `ddet_baseline` measures DDET and DET for white noise, shuffled surrogates or AR(1) red noise with exactly your settings.
* Five executed notebooks: a step-by-step reproduction of the paper, three further chaotic systems, and a study of the noise baseline.

## Install

```bash
git clone <this repository> && cd <repository folder>
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # numpy only for the package; [plot] adds matplotlib, [dev] adds tests and notebooks
python -m pytest -q tests
```

## Quick start

```python
import exrqa as xr

x  = xr.zscore(my_series)                            # evenly sampled 1-D series
tm = xr.suggest_tau_m(x, m=3, tau=10)                # {'lower': 3, 'upper': 23, 'recommended': 13}
R  = xr.recurrence_plot(x, m=3, tau=10, method="lm2p", perc=5, tau_m=tm["recommended"])

q = xr.rqa(R)                                        # RR, DET, L, Lmax, LAM, TT
d = xr.ddet(R, ws=2, tau_w=2, d_max=2)               # DDETResult(ddet, rr_w, dd_w, offsets)

# distance from noise, with a baseline computed with the same settings
b = xr.ddet_baseline(x=x, m=3, tau=10, tau_m=tm["recommended"])                    # shuffled surrogates
r = xr.ddet_baseline(x=x, m=3, tau=10, tau_m=tm["recommended"], surrogate="ar1")   # red noise: the harder test
d.ddet - b["mean"], q["DET"] - b["det_mean"]         # ΔDDET, ΔDET
```

Lower-level building blocks, for example to reuse one distance matrix across methods:

```python
D   = xr.distance_matrix(xr.embed(x, m=3, tau=10))
eps = xr.eps_from_percentile(D, 5)
R_thr  = xr.threshold_rp(D, eps)
R_lm   = xr.local_minima_rp(D)                       # LM
R_lmt  = xr.local_minima_rp(D, eps)                  # LMT  (Schultz et al. 2011)
R_lm2p = xr.local_minima_rp(D, eps, tau_m=10)        # LM2P (Wendi & Marwan 2018)
```

| module | contents |
|---|---|
| `recurrence` | `embed`, `distance_matrix`, `eps_from_percentile`, `threshold_rp`, `local_minima`, `local_minima_rp`, `recurrence_plot` |
| `quantification` | `rqa`, `line_lengths`, `diagonal_windows`, `ddet`, `DDETResult`, `ddet_iid` (DDET of a random RP) |
| `tuning` | `acf`, `acf_first_crossing`, `minima_spacing`, `suggest_tau_m` (τm rule), `ddet_baseline` (DDET and DET of white noise, shuffled or AR(1) surrogates) |
| `data` | `lorenz`, `rossler`, `duffing`, `mackey_glass` (RK4; an array of parameter values simulates a whole sweep at once), `peak_values`, `zscore`, `add_noise` (SNR in dB), `ar1_noise` |
| `plotting` | `plot_rp`, `exceedance`, `plot_exceedance` |

## Choosing settings

Paper defaults: m = 3, τ = 10, ε = 5th percentile of the distances, τm = 10, `ws = 2`, `tau_w = 2`, `d_max = 2`.

* **Compare with a matched baseline, not with −1.** For a random RP, DDET ≈ ln(ws² · d_max(d_max+1)/2 · RR), which is ln(12 · RR) for the paper's window. The paper's ε and τm give white noise RR ≈ 0.03, hence DDET ≈ −1. Other settings move this anywhere from about −2.5 to +2.2. Coloured noise and an embedding delay τ ≤ `d_max` raise it further.
* **Test against red noise as well.** Against white noise, DET and DDET both recognise the four chaotic test signals down to about 0 dB. Noise with memory is harder, and there DDET holds much longer than DET.
* **Don't oversample.** `ws` and `d_max` are counted in samples. In every example, downsampling the same noisy record by 2 widened the gap between periodic and chaotic DDET.
* **Keep `ws` small (2–3)**, and try m = 5 if m = 3 gives a weak contrast.

## Example notebooks

All notebooks are in `notebooks/` and are stored with their outputs. Each example takes about 2.5 minutes to re-run; `ddet_white_noise.ipynb` takes about 8.

| notebook | content |
|---|---|
| `lorenz_example.ipynb` | Step-by-step reproduction of the paper's Lorenz study (Figs. 1–10): LM/LMT/LM2P, choice of τm, DDET windows, the ρ = 80…105 sweep, sensitivity to SNR, `ws` and `m`. Also the distance from noise and a recipe for your own data. |
| `rossler_example.ipynb`, `duffing_example.ipynb`, `mackey_glass_example.ipynb` | The method on three other chaotic systems: RPs under noise, a parameter sweep through periodic windows (ground truth from the bifurcation diagram), noise level, sampling and embedding, distance from noise. |
| `ddet_white_noise.ipynb` | Whether white-noise DDET is always −1 (it is not), and all four systems compared against matched white- and red-noise baselines. |

Results with 20 dB noise and LM2P plots:

| system | sweep | ΔDDET / ΔDET, chaotic | AUC periodic vs chaotic, DDET / DET | AUC vs red noise at 0 dB, DDET / DET |
|---|---|---|---|---|
| Lorenz | ρ = 80…105 | +1.53 / +0.22 | 0.96 / 0.65 | 0.50 / 0.01 |
| Rössler (a = b = 0.2) | c = 2.5…6.5 | +2.88 / +0.71 | 0.97 / 0.84 | 1.00 / 0.99 |
| Duffing (δ = 0.3, ω = 1.2) | γ = 0.20…0.65 | +1.79 / +0.39 | 0.99 / 0.93 | 0.94 / 0.41 |
| Mackey–Glass (n = 10) | τ_d = 8…30 | +1.78 / +0.26 | 0.99 / 0.98 | 0.92 / 0.54 |

* **ΔDDET, ΔDET**: median distance of the chaotic parameter values from 20 white-noise series analysed with the same settings.
* **AUC periodic vs chaotic**: the probability that a periodic parameter value scores higher than a chaotic one (1 = perfect separation, 0.5 = chance). For Lorenz the periodic windows come from the Lyapunov exponent; for the other systems they come from the bifurcation diagram.
* **AUC vs red noise**: the probability that a noisy segment of the chaotic signal scores higher than AR(1) noise with the same lag-1 autocorrelation (20 against 20 series). Lorenz, analysed with the paper's m = 3 and fine sampling, is the hardest case: at 10 dB the values are 0.83 for DDET and 0.00 for DET.

## Notes and assumptions

The paper gives no code. The interpretation choices made here are documented in the `quantification` module docstring and in §9 of `lorenz_example.ipynb`:

* minima are searched along rows, and LAM counts runs in columns;
* the ε threshold is applied before the τm suppression, keeping the deepest minimum first;
* `P(D)` holds gap counts, the denominator is `l_w` minus the connected length, and `A_w` is the number of cells in the window;
* the log is natural, and windows cover the upper triangle from lag 1 to N − ws.

**Known deviations from the paper:**

* Periodic DDET reaches larger values, because a window completely filled by a line gives `DD_w ≈ l_w`. The option `max_lag` drops the short windows near the RP corner.
* DDET increases with `ws` instead of decreasing as in Fig. 9. Please keep `ws` at 2–3 in either case.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{Wendi2018,
  author  = {Wendi, D. and Marwan, N.},
  title   = {Extended recurrence plot and quantification for noisy continuous dynamical systems},
  journal = {Chaos: An Interdisciplinary Journal of Nonlinear Science},
  volume  = {28},
  number  = {8},
  pages   = {085722},
  year    = {2018},
  doi     = {10.1063/1.5025485}
}
```
