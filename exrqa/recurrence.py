"""Recurrence plot construction: threshold RP, LM, LMT and LM2P.

Conventions
-----------
* ``D[i, j]`` is the phase-space distance between state ``x_i`` and ``x_j``.
* For the local-minima family the *distance profile* of a reference state
  ``i`` is the row ``D[i, :]`` (Fig. 1d in Wendi & Marwan 2018): ``R[i, j] = 1``
  when ``j`` is a local minimum of that profile.  Shown with
  ``imshow(R, origin="lower")`` (x = j, y = i) the noise-induced artefacts
  appear as vertical lines, as in Fig. 2d of the paper, and LAM counts them.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "embed",
    "distance_matrix",
    "eps_from_percentile",
    "threshold_rp",
    "local_minima",
    "local_minima_rp",
    "recurrence_plot",
]


def embed(x, m: int = 3, tau: int = 1) -> np.ndarray:
    """Time-delay embedding, Eq. (1): ``x_i = (u_i, u_{i+tau}, ..., u_{i+(m-1)tau})``.

    ``x`` may be 1-D (scalar series) or 2-D ``(n, d)`` (each column is delayed).
    Returns an array of shape ``(n - (m-1)*tau, m*d)``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x) - (m - 1) * tau
    if n <= 0:
        raise ValueError("series too short for the requested m and tau")
    return np.concatenate([x[k * tau : k * tau + n] for k in range(m)], axis=1)


def distance_matrix(X, metric: str = "euclidean") -> np.ndarray:
    """Pairwise distances between state vectors (rows of ``X``).

    ``metric``: ``"euclidean"`` (default), ``"maximum"`` or ``"manhattan"``.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if metric == "euclidean":
        sq = np.einsum("ij,ij->i", X, X)
        D2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
        np.maximum(D2, 0.0, out=D2)
        D = np.sqrt(D2)
        np.fill_diagonal(D, 0.0)
        return D
    diff = np.abs(X[:, None, :] - X[None, :, :])
    if metric == "maximum":
        return diff.max(axis=-1)
    if metric == "manhattan":
        return diff.sum(axis=-1)
    raise ValueError(f"unknown metric {metric!r}")


def eps_from_percentile(D, perc: float = 5.0) -> float:
    """Recurrence threshold as a percentile of the off-diagonal distances (paper: 5th)."""
    D = np.asarray(D)
    iu = np.triu_indices(D.shape[0], k=1)
    return float(np.percentile(D[iu], perc))


def threshold_rp(D, eps: float) -> np.ndarray:
    """Classic RP, Eq. (2): ``R_ij = Theta(eps - ||x_i - x_j||)``."""
    return (np.asarray(D) <= eps).astype(np.uint8)


def local_minima(D) -> np.ndarray:
    """Boolean mask of local minima of every row profile ``D[i, :]``.

    Ends of the profile count as minima if smaller than their only neighbour;
    on flat plateaus the left-most sample is taken.
    """
    D = np.asarray(D, dtype=float)
    P = np.pad(D, ((0, 0), (1, 1)), constant_values=np.inf)
    return (D < P[:, :-2]) & (D <= P[:, 2:])


def _min_separation(D, cand, tau_m: int) -> np.ndarray:
    """Keep, per row, the deepest candidate minima so that kept ones are >= tau_m apart.

    Greedy (deepest first) suppression, i.e. the same rule as
    ``findpeaks(..., 'MinPeakDistance', tau_m)`` applied to ``-D[i, :]``.
    """
    n_rows, n_cols = cand.shape
    keep = np.zeros_like(cand)
    for i in range(n_rows):
        idx = np.flatnonzero(cand[i])
        if idx.size == 0:
            continue
        if idx.size == 1:
            keep[i, idx] = True
            continue
        order = idx[np.argsort(D[i, idx], kind="stable")]
        blocked = np.zeros(n_cols, dtype=bool)
        for j in order:
            if not blocked[j]:
                keep[i, j] = True
                blocked[max(0, j - tau_m + 1) : j + tau_m] = True
    return keep


def local_minima_rp(D, eps: float | None = None, tau_m: int | None = None) -> np.ndarray:
    """Local-minima based RP.

    * ``eps=None, tau_m=None``  -> **LM**   (all local minima)
    * ``eps``                   -> **LMT**  (minima with distance <= eps; Schultz et al. 2011)
    * ``eps`` and ``tau_m``     -> **LM2P** (LMT + minimum time distance ``tau_m``
      between the minima kept in each profile; Wendi & Marwan 2018, Sec. II)
    """
    D = np.asarray(D, dtype=float)
    mask = local_minima(D)
    if eps is not None:
        mask &= D <= eps
    if tau_m is not None and tau_m > 1:
        mask = _min_separation(D, mask, int(tau_m))
    return mask.astype(np.uint8)


def recurrence_plot(
    x,
    m: int = 1,
    tau: int = 1,
    method: str = "lm2p",
    eps: float | None = None,
    perc: float = 5.0,
    tau_m: int = 10,
    metric: str = "euclidean",
) -> np.ndarray:
    """One-call RP: embed ``x`` (if ``m > 1``), compute distances, build the RP.

    ``method``: ``"threshold"``, ``"lm"``, ``"lmt"`` or ``"lm2p"``.
    ``eps`` defaults to the ``perc``-th percentile of the distances.
    """
    X = embed(x, m, tau) if m > 1 else np.asarray(x, dtype=float)
    D = distance_matrix(X, metric)
    method = method.lower()
    if method == "lm":
        return local_minima_rp(D)
    if eps is None:
        eps = eps_from_percentile(D, perc)
    if method == "threshold":
        return threshold_rp(D, eps)
    if method == "lmt":
        return local_minima_rp(D, eps)
    if method == "lm2p":
        return local_minima_rp(D, eps, tau_m)
    raise ValueError(f"unknown method {method!r}")
