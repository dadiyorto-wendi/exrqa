"""Minimal plotting helpers (matplotlib is imported lazily)."""
from __future__ import annotations

import numpy as np

__all__ = ["plot_rp", "exceedance", "plot_exceedance"]


def plot_rp(R, ax=None, start: int | None = None, stop: int | None = None,
            title: str | None = None, **kwargs):
    """Show an RP with ``origin='lower'`` (x = j, y = i). ``start/stop`` zoom in."""
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    s = slice(start, stop)
    R = np.asarray(R)[s, s]
    a = 0 if start is None else start
    ax.imshow(R, origin="lower", cmap="binary", interpolation="nearest",
              extent=(a - 0.5, a + R.shape[1] - 0.5, a - 0.5, a + R.shape[0] - 0.5),
              **kwargs)
    ax.grid(False)
    ax.set_xlabel("time")
    ax.set_ylabel("time")
    if title:
        ax.set_title(title)
    return ax


def exceedance(values):
    """Sorted values (descending) and their probability of exceedance."""
    v = np.sort(np.asarray(values, dtype=float))[::-1]
    p = np.arange(1, v.size + 1) / v.size
    return v, p


def plot_exceedance(values, ax=None, **kwargs):
    """Probability-of-exceedance plot (as Figs. 4a and 6a,b), log y-axis.

    Probabilities use all values; zeros are not drawn on the log axis, so the
    curve ends where the remaining values are zero (as in the paper).
    """
    import matplotlib.pyplot as plt

    ax = ax or plt.gca()
    v, p = exceedance(values)
    keep = v > 0
    ax.semilogy(p[keep], v[keep], **kwargs)
    ax.set_xlabel("probability of exceedance")
    return ax
