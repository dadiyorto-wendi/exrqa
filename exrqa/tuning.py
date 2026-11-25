"""Parameter guidance: choosing ``tau_m`` for LM2P, and the stochastic DDET
baseline for the settings in use.

Paper rule (Sec. II, Fig. 3): ``tau_m`` should be
* larger than the spacing of local minima produced by pure noise
  (Gaussian white noise: ≈ 3 samples), and
* smaller than the first lag where the ACF of the signal drops to <= 0.1.
"""
from __future__ import annotations

import numpy as np

from .data import ar1_noise
from .quantification import ddet, ddet_iid, rqa
from .recurrence import distance_matrix, embed, local_minima, recurrence_plot

__all__ = ["acf", "acf_first_crossing", "minima_spacing", "suggest_tau_m", "ddet_baseline"]


def acf(x, max_lag: int | None = None) -> np.ndarray:
    """Sample autocorrelation function for lags ``0..max_lag``."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    max_lag = n - 1 if max_lag is None else min(max_lag, n - 1)
    f = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    return r / r[0]


def acf_first_crossing(x, threshold: float = 0.1, max_lag: int | None = None) -> int:
    """First lag at which the ACF is <= ``threshold`` (paper: 0.1)."""
    r = acf(x, max_lag)
    below = np.flatnonzero(r <= threshold)
    if below.size == 0:
        raise ValueError("ACF never drops below the threshold; increase max_lag")
    return int(below[0])


def minima_spacing(D, eps: float | None = None, stat=np.median) -> np.ndarray:
    """Per reference state ``i``: ``stat`` of the spacing between consecutive local
    minima of its distance profile ``D[i, :]`` (Fig. 3a).  NaN if < 2 minima."""
    mask = local_minima(D)
    if eps is not None:
        mask &= np.asarray(D) <= eps
    out = np.full(mask.shape[0], np.nan)
    for i, row in enumerate(mask):
        idx = np.flatnonzero(row)
        if idx.size > 1:
            out[i] = stat(np.diff(idx))
    return out


def suggest_tau_m(x, m: int = 3, tau: int = 1, acf_threshold: float = 0.1,
                  n_ref: int = 1000, seed: int | None = 0) -> dict:
    """Safe range for ``tau_m`` following the paper's ACF / white-noise rule.

    * ``lower``: 95th percentile of the minima spacing of a Gaussian white noise
      reference embedded with the same ``m``, ``tau`` (≈ 3)
    * ``upper``: first lag where the ACF of ``x`` is <= ``acf_threshold``
    * ``recommended``: midpoint of the two (a pragmatic default)
    """
    upper = acf_first_crossing(x, acf_threshold)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(min(len(x), n_ref) + (m - 1) * tau)
    D = distance_matrix(embed(noise, m, tau))
    lower = float(np.nanpercentile(minima_spacing(D), 95))
    rec = int(round((lower + upper) / 2))
    if upper <= lower:
        print(f"warning: ACF bound ({upper}) <= noise bound ({lower:.0f}); "
              "signal looks noise-like, LM2P may not help")
    return {"lower": lower, "upper": upper, "recommended": rec}


def ddet_baseline(x=None, n: int | None = None, m: int = 1, tau: int = 1,
                  method: str = "lm2p", perc: float = 5.0, tau_m: int = 10,
                  ws: int = 2, tau_w: int = 2, d_max: int = 2, max_lag: int | None = None,
                  metric: str = "euclidean", n_real: int = 20, seed: int | None = 0,
                  surrogate: str = "shuffle") -> dict:
    """DDET and DET of a stochastic reference, computed with exactly the given settings.

    The white-noise DDET is not a constant: it depends on eps (``perc``),
    ``tau_m``, ``ws`` and ``d_max`` (see :func:`exrqa.quantification.ddet_iid`).
    Compare a signal's DDET (and DET) with this baseline, not with a fixed -1.

    * ``x`` given, ``surrogate="shuffle"``: ``n_real`` shuffled surrogates of
      ``x`` (same values, no temporal order), i.e. white noise with the
      amplitude distribution of ``x``.
    * ``x`` given, ``surrogate="ar1"``: ``n_real`` AR(1) red-noise series with
      the lag-1 autocorrelation of ``x`` (noise with short memory). This is a
      harder reference than white noise for smooth signals.
    * otherwise: ``n_real`` Gaussian white noise series of length ``n``.

    Returns a dict with the DDET ``mean``, ``sd`` and ``values``, the DET
    ``det_mean``, ``det_sd`` and ``det_values`` (from the same RPs), the mean
    window recurrence rate ``rr``, the random-RP prediction ``theory`` (valid
    for white noise only) and, for AR(1) surrogates, ``phi``.
    """
    if x is None and n is None:
        raise ValueError("give a series x (surrogates) or a length n (white noise)")
    if surrogate not in ("shuffle", "ar1"):
        raise ValueError("surrogate must be 'shuffle' or 'ar1'")
    rng = np.random.default_rng(seed)
    phi = None
    if x is not None:
        x = np.asarray(x, dtype=float)
        if surrogate == "ar1":
            phi = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    vals, dets, rrs = [], [], []
    for _ in range(n_real):
        if x is None:
            s = rng.standard_normal(n)
        elif surrogate == "ar1":
            s = ar1_noise(x.size, phi, rng)
        else:
            s = rng.permutation(x)
        R = recurrence_plot(s, m=m, tau=tau, method=method, perc=perc, tau_m=tau_m, metric=metric)
        d = ddet(R, ws=ws, tau_w=tau_w, d_max=d_max, max_lag=max_lag)
        vals.append(d.ddet)
        dets.append(rqa(R)["DET"])
        rrs.append(d.rr_w.mean())
    vals, dets = np.array(vals), np.array(dets)
    rr = float(np.mean(rrs))
    sd = lambda v: float(v.std(ddof=1)) if n_real > 1 else 0.0
    out = {"mean": float(vals.mean()), "sd": sd(vals), "values": vals,
           "det_mean": float(dets.mean()), "det_sd": sd(dets), "det_values": dets,
           "rr": rr, "theory": ddet_iid(rr, ws, d_max, exclusive=method.lower() != "threshold")}
    if phi is not None:
        out["phi"] = phi
    return out
