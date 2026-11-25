"""Recurrence quantification: conventional RQA and the diagonal sliding window DDET.

DDET (Wendi & Marwan 2018, Sec. III, Eqs. 5-6)
---------------------------------------------
A window of ``ws`` neighbouring diagonals slides from the line of identity
(LOI) towards the corner of the RP, starting every ``tau_w`` diagonals.  In each
window ``w``:

* ``RR_w`` = recurrence points in the window / window area            (Eq. 5)
* the ``ws`` cells of each row inside the window are OR-ed into one binary
  sequence ``d_k`` along the diagonal (tolerates small sideways wiggles);
* the gaps ``D = Δk`` between consecutive ``d_k = 1`` are collected; gaps up to
  ``d_max`` are treated as connected (tolerates small disruptions);
* ``DD_w = Σ_{D<=d_max} D·P(D) / (l_w - Σ_{D<=d_max} D·P(D))``          (Eq. 6)
  i.e. the diagonal length covered by (nearly) connected points relative to
  the length that is not, with ``l_w`` the window length along the diagonal.

Finally ``DDET = ln( mean_w DD_w / mean_w RR_w )``.

Interpretation notes: Eq. (6) is typeset ambiguously in the paper.  Here
``P(D)`` is the count histogram of gaps (so ``Σ D·P(D)`` is a length), the
denominator is read as ``l_w`` minus the connected length, ``A_w`` is the
number of cells in the window (``ws`` diagonals) and ``log`` is the natural
log.  With these choices white noise gives DDET ≈ -1 for the paper's settings
(as in the paper; the value depends on the settings, see ``ddet_iid``) and a
clean chaotic Lorenz RP ≈ 0.7-1.1 (paper: ≈ 1.2).  Windows that one line fills
completely give ``DD_w ≈ l_w`` (denominator -> 1), so periodic signals reach
larger values than in the paper; see ``max_lag`` for the corner windows.
Only the upper triangle is scanned; use ``R.T`` for the other one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import as_strided

__all__ = ["line_lengths", "rqa", "DDETResult", "diagonal_windows", "ddet", "ddet_iid"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _runs(B) -> np.ndarray:
    """Lengths of runs of ones along the last axis of a 2-D binary array."""
    B = np.asarray(B, dtype=np.int8)
    pad = np.zeros(B.shape[:-1] + (1,), dtype=np.int8)
    d = np.diff(np.concatenate([pad, B, pad], axis=-1), axis=-1).ravel()
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def _shear(R) -> np.ndarray:
    """Upper-triangle diagonals as rows: ``S[k, i] = R[i, i+k]`` (0 where i+k >= N)."""
    R = np.ascontiguousarray(R, dtype=np.uint8)
    N = R.shape[0]
    P = np.zeros((N, 2 * N), dtype=np.uint8)
    P[:, :N] = R
    s0, s1 = P.strides
    V = as_strided(P, shape=(N, N), strides=(s0 + s1, s1), writeable=False)  # V[i,k]=R[i,i+k]
    return np.ascontiguousarray(V.T)


def _theiler_mask(N: int, theiler: int) -> np.ndarray:
    i = np.arange(N)
    return np.abs(i[:, None] - i[None, :]) >= theiler


# --------------------------------------------------------------------------- #
# conventional RQA
# --------------------------------------------------------------------------- #
def line_lengths(R, kind: str = "diagonal", theiler: int = 1) -> np.ndarray:
    """All diagonal or vertical line lengths of ``R``.

    Diagonals with ``|i-j| < theiler`` (``theiler=1`` -> only the LOI) are
    excluded.  Both triangles are used, since LM-type RPs are not symmetric.
    Vertical lines are runs within a column ``R[:, j]``.
    """
    R = np.asarray(R, dtype=np.uint8)
    if kind == "diagonal":
        up = _shear(R)[theiler:]
        lo = _shear(R.T)[max(theiler, 1):]
        return np.concatenate([_runs(up), _runs(lo)])
    if kind == "vertical":
        Rm = R * _theiler_mask(R.shape[0], theiler)
        return _runs(Rm.T)
    raise ValueError("kind must be 'diagonal' or 'vertical'")


def rqa(R, lmin: int = 2, vmin: int = 2, theiler: int = 1) -> dict:
    """Basic RQA measures: RR, DET (Eq. 3), L, Lmax, LAM (Eq. 4), TT.

    Points within the Theiler window (``|i-j| < theiler``) are ignored.
    """
    R = np.asarray(R, dtype=np.uint8)
    N = R.shape[0]
    mask = _theiler_mask(N, theiler)
    n_rec = int((R * mask).sum())
    l = line_lengths(R, "diagonal", theiler)
    v = line_lengths(R, "vertical", theiler)
    l_ok, v_ok = l[l >= lmin], v[v >= vmin]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "RR": n_rec / mask.sum(),
            "DET": l_ok.sum() / n_rec if n_rec else np.nan,
            "L": l_ok.mean() if l_ok.size else np.nan,
            "Lmax": int(l.max()) if l.size else 0,
            "LAM": v_ok.sum() / n_rec if n_rec else np.nan,
            "TT": v_ok.mean() if v_ok.size else np.nan,
        }


# --------------------------------------------------------------------------- #
# diagonal sliding window (DDET)
# --------------------------------------------------------------------------- #
@dataclass
class DDETResult:
    ddet: float            # ln(mean DD_w / mean RR_w)
    rr_w: np.ndarray       # recurrence rate per window, Eq. (5)
    dd_w: np.ndarray       # DD per window, Eq. (6)
    offsets: np.ndarray    # first diagonal (lag from the LOI) of each window

    def __repr__(self) -> str:
        return (f"DDETResult(ddet={self.ddet:.3f}, n_windows={self.offsets.size}, "
                f"mean_RR_w={self.rr_w.mean():.4f}, mean_DD_w={self.dd_w.mean():.4f})")


def diagonal_windows(R, ws: int = 2, tau_w: int = 2, d_max: int = 2, theiler: int = 1,
                     max_lag: int | None = None):
    """``RR_w`` and ``DD_w`` for every diagonal window (upper triangle, as in Fig. 5a).

    Parameters
    ----------
    ws      : window width in diagonals (paper: 2; keep small, 2-3).
    tau_w   : step between window starts, in diagonals (paper: 2).
    d_max   : largest gap Δk still counted as connected (1 = strictly
              consecutive, 2 = one missing point allowed).
    theiler : first diagonal used (1 skips the LOI).
    max_lag : last window start (default: ``N - ws``, the RP corner, as in the
              paper).  Short windows near the corner can be completely filled
              by one line, which makes ``DD_w`` explode (denominator -> 1);
              ``max_lag = N // 2`` is a pragmatic way to stabilise DDET.

    Returns ``(rr_w, dd_w, offsets)``.
    """
    R = np.asarray(R, dtype=np.uint8)
    N = R.shape[0]
    S = _shear(R)
    last = N - ws if max_lag is None else min(max_lag, N - ws)
    offsets = np.arange(theiler, last + 1, tau_w)
    if offsets.size == 0:
        raise ValueError("RP too small for the chosen ws / theiler")

    d = np.zeros((offsets.size, N), dtype=bool)   # OR-aggregated cells d_k
    count = np.zeros(offsets.size)
    area = np.zeros(offsets.size)
    for t in range(ws):
        rows = S[offsets + t]
        d |= rows.astype(bool)
        count += rows.sum(axis=1)
        area += N - (offsets + t)
    rr_w = count / area

    w, k = np.nonzero(d)                          # sorted by window, then k
    same = w[1:] == w[:-1]
    gaps = np.diff(k)[same]
    gw = w[1:][same]
    short = gaps <= d_max
    connected = np.bincount(gw[short], weights=gaps[short], minlength=offsets.size)
    l_w = N - offsets
    dd_w = connected / (l_w - connected)          # l_w - connected >= 1 always
    return rr_w, dd_w, offsets


def ddet(R, ws: int = 2, tau_w: int = 2, d_max: int = 2, theiler: int = 1,
         max_lag: int | None = None) -> DDETResult:
    """Determinism indicator from the diagonal sliding window, Eq. (6).

    Parameters as in :func:`diagonal_windows`.  Typical values with the paper's
    settings: ≈ -1 for white noise, ≈ 1 for chaotic Lorenz, clearly higher for
    periodic dynamics. The white-noise value depends on the settings (see
    :func:`ddet_iid` and :func:`exrqa.tuning.ddet_baseline`), so compare with
    a baseline computed with the same settings.
    """
    rr_w, dd_w, offsets = diagonal_windows(R, ws, tau_w, d_max, theiler, max_lag)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = float(np.log(dd_w.mean() / rr_w.mean()))
    return DDETResult(value, rr_w, dd_w, offsets)


def ddet_iid(rr, ws: int = 2, d_max: int = 2, exclusive: bool = True) -> float:
    """Expected DDET of an RP whose cells are independent with density ``rr``.

    This is the white-noise reference. A row of a window is "on" with
    probability ``q``. Gaps between "on" rows are geometric, so the connected
    fraction is ``c = q * sum_{D<=d_max} D q (1-q)^(D-1)`` and
    ``DDET = ln( c / (1 - c) / rr )``. For small ``rr`` this is about
    ``ln(ws**2 * d_max (d_max + 1) / 2 * rr)``, i.e. ``ln(12 rr)`` for the
    paper's ``ws = d_max = 2``: white noise gives DDET = -1 only when its RR is
    about 0.03, as with the paper's eps (5th percentile) and tau_m = 10.

    ``rr`` is the mean window recurrence rate (``ddet(R).rr_w.mean()``).
    ``exclusive=True`` for local-minima RPs, where two cells of the same row
    in one window cannot both be minima (``q = ws * rr``, valid for
    ``ws <= tau_m``); ``False`` for threshold RPs (``q = 1 - (1 - rr)**ws``).
    The model fails when the RP cells are not independent, e.g. for coloured
    noise or an embedding delay ``tau <= d_max``.
    """
    q = ws * rr if exclusive else 1.0 - (1.0 - rr) ** ws
    g = sum(d * q * (1.0 - q) ** (d - 1) for d in range(1, d_max + 1))
    c = q * g
    return float(np.log(c / (1.0 - c) / rr))
