"""exrqa — extended recurrence plots and quantification for noisy continuous systems.

Implements Wendi & Marwan (2018), Chaos 28, 085722, doi:10.1063/1.5025485:

* LM2P recurrence plot  — local minima with threshold ``eps`` *and* a minimal
  time distance ``tau_m`` between minima (removes noise-induced blocks and
  vertical lines).
* DDET — determinism indicator from a diagonal sliding window that tolerates
  wiggling (``ws``) and disrupted (``d_max``) diagonal lines.

Quick start::

    import exrqa as xr
    X = xr.embed(signal, m=3, tau=10)
    D = xr.distance_matrix(X)
    eps = xr.eps_from_percentile(D, 5)
    R = xr.local_minima_rp(D, eps, tau_m=10)      # LM2P
    xr.rqa(R)["DET"], xr.ddet(R, ws=2, tau_w=2).ddet
"""
from .data import lorenz, rossler, duffing, mackey_glass, peak_values, zscore, add_noise, ar1_noise
from .plotting import exceedance, plot_exceedance, plot_rp
from .quantification import DDETResult, ddet, ddet_iid, diagonal_windows, line_lengths, rqa
from .recurrence import (distance_matrix, embed, eps_from_percentile, local_minima,
                         local_minima_rp, recurrence_plot, threshold_rp)
from .tuning import acf, acf_first_crossing, ddet_baseline, minima_spacing, suggest_tau_m

__version__ = "0.1.0"
