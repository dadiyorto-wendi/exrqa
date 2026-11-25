"""Test signals: classic chaotic flows, additive noise at a given SNR, and a
bifurcation-diagram helper.

All simulators use fixed-step RK4 (``dt / substeps``) and return samples every
``dt`` after discarding ``transient`` samples. Each control parameter may be a
scalar (output ``(n, d)``) or an array of K values (output ``(K, n, d)``), so a
whole parameter sweep is integrated at once.
"""
from __future__ import annotations

import numpy as np

__all__ = ["lorenz", "rossler", "duffing", "mackey_glass", "peak_values",
           "zscore", "add_noise", "ar1_noise"]


def _rk4_flow(f, s, n, dt, substeps, transient):
    h = dt / substeps
    out = np.empty((n,) + s.shape)
    for t in range(transient + n):
        for _ in range(substeps):
            k1 = f(s)
            k2 = f(s + 0.5 * h * k1)
            k3 = f(s + 0.5 * h * k2)
            k4 = f(s + h * k3)
            s = s + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if t >= transient:
            out[t - transient] = s
    return np.moveaxis(out, 0, -2)


def _setup(params, x0):
    """Broadcast the parameters to one shape and the initial state to (..., d)."""
    params = np.broadcast_arrays(*[np.asarray(p, dtype=float) for p in params])
    x0 = np.asarray(x0, dtype=float)
    s = np.broadcast_to(x0, params[0].shape + x0.shape[-1:]).copy()
    return params, s


def lorenz(n: int, rho=28.0, sigma=10.0, beta=8.0 / 3.0, dt: float = 0.02,
           substeps: int = 10, transient: int = 2000, x0=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Lorenz-63, columns (x, y, z). Chaotic for the classic rho = 28; rho = 80..105
    has periodic windows near 92-93 and 99-100 (Wendi & Marwan 2018, Fig. 7)."""
    (rho, sigma, beta), s = _setup((rho, sigma, beta), x0)

    def f(s):
        x, y, z = s[..., 0], s[..., 1], s[..., 2]
        return np.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], axis=-1)

    return _rk4_flow(f, s, n, dt, substeps, transient)


def rossler(n: int, c=5.7, a=0.2, b=0.2, dt: float = 0.15, substeps: int = 10,
            transient: int = 2000, x0=(1.0, 1.0, 0.0)) -> np.ndarray:
    """Rossler (1976), columns (x, y, z).

    dx = -y - z,  dy = x + a y,  dz = b + z (x - c).
    One rotation of (x, y) takes about 6 time units, so the default ``dt = 0.15``
    gives about 40 samples per rotation. With a = b = 0.2, c = 2.5 is a period-1
    orbit, period doubling leads to chaos near c = 4.2, and c = 5.7 is the
    classic chaotic attractor.
    """
    (c, a, b), s = _setup((c, a, b), x0)

    def f(s):
        x, y, z = s[..., 0], s[..., 1], s[..., 2]
        return np.stack([-y - z, x + a * y, b + z * (x - c)], axis=-1)

    return _rk4_flow(f, s, n, dt, substeps, transient)


def duffing(n: int, gamma=0.5, delta=0.3, alpha=-1.0, beta=1.0, omega=1.2,
            dt: float | None = None, substeps: int = 10, transient: int = 2000,
            x0=(1.0, 0.0)) -> np.ndarray:
    """Forced double-well Duffing oscillator, columns (x, v).

    x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t).
    The default ``dt`` samples each forcing period ``2 pi / omega`` 40 times.
    With delta = 0.3, omega = 1.2: gamma = 0.20 period 1, 0.28 period 2,
    0.29 period 4, 0.37 period 5, 0.50 chaotic, 0.65 period 2.
    Integrated as an autonomous system with the forcing phase as a third state,
    which starts at 0. Coexisting attractors exist, so results depend on ``x0``.
    """
    (gamma, delta, alpha, beta, omega), _ = _setup((gamma, delta, alpha, beta, omega), (0.0,))
    if dt is None:
        dt = 2 * np.pi / float(np.mean(omega)) / 40
    x0 = np.asarray(x0, dtype=float)
    s = np.concatenate([np.broadcast_to(x0, gamma.shape + (2,)), np.zeros(gamma.shape + (1,))], axis=-1)

    def f(s):
        x, v, th = s[..., 0], s[..., 1], s[..., 2]
        return np.stack([v, gamma * np.cos(th) - delta * v - alpha * x - beta * x ** 3,
                         np.broadcast_to(omega, th.shape)], axis=-1)

    return _rk4_flow(f, s, n, dt, substeps, transient)[..., :2]


def mackey_glass(n: int, tau=17.0, beta=0.2, gamma=0.1, power: float = 10.0,
                 dt: float = 1.0, substeps: int = 10, transient: int = 1000,
                 x0: float = 1.2) -> np.ndarray:
    """Mackey-Glass delay equation, a single column (x).

    dx/dt = beta x(t - tau) / (1 + x(t - tau)^power) - gamma x(t).
    RK4 with step h = dt / substeps and a constant history x0. The delay is
    rounded to a whole number of steps, and delayed values at half steps are
    linearly interpolated. With the defaults the orbit is periodic for
    tau < ~13, period doubles up to ~16.8 and is chaotic above, as in the
    classic tau = 17 benchmark.
    """
    (tau, beta, gamma, power), _ = _setup((tau, beta, gamma, power), (0.0,))
    shape = tau.shape
    tau, beta, gamma, power = (p.ravel() for p in (tau, beta, gamma, power))
    h = dt / substeps
    d = np.maximum(np.rint(tau / h).astype(int), 1)          # delay in steps
    L = int(d.max()) + 1
    rows = np.arange(tau.size)
    buf = np.full((tau.size, L), float(x0))                    # ring buffer of past values
    x = np.full(tau.size, float(x0))
    ptr = 0

    def f(x, xd):
        return beta * xd / (1.0 + xd ** power) - gamma * x

    out = np.empty((n, tau.size))
    for t in range(transient + n):
        for _ in range(substeps):
            xd0 = buf[rows, (ptr - d) % L]                     # x(t - tau)
            xd1 = buf[rows, (ptr - d + 1) % L]                 # x(t - tau + h)
            xdm = 0.5 * (xd0 + xd1)
            k1 = f(x, xd0)
            k2 = f(x + 0.5 * h * k1, xdm)
            k3 = f(x + 0.5 * h * k2, xdm)
            k4 = f(x + h * k3, xd1)
            x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            ptr = (ptr + 1) % L
            buf[:, ptr] = x
        if t >= transient:
            out[t - transient] = x
    return np.moveaxis(out.reshape((n,) + shape + (1,)), 0, -2)


def peak_values(x) -> np.ndarray:
    """Values of the local maxima of a sampled series, refined by a parabola
    through each peak and its two neighbours. Plotted against a control
    parameter they give the bifurcation diagram; a periodic orbit has only a
    few distinct peak values, a chaotic one has many."""
    x = np.asarray(x, dtype=float)
    i = np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])) + 1
    a, b, c = x[i - 1], x[i], x[i + 1]
    den = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(den != 0, 0.5 * (a - c) / den, 0.0)
    return b - 0.25 * (a - c) * shift


def zscore(x, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean(axis, keepdims=True)) / x.std(axis, keepdims=True)


def add_noise(x, snr_db: float, seed: int | None = None) -> np.ndarray:
    """Add Gaussian white noise so that ``10 log10(var(x)/var(noise)) = snr_db``."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    sd = np.sqrt(x.var() / 10 ** (snr_db / 10))
    return x + rng.normal(0.0, sd, x.shape)


def ar1_noise(n: int, phi: float, seed=None, burn: int = 500) -> np.ndarray:
    """Red noise: AR(1) process ``x_t = phi * x_(t-1) + e_t`` with Gaussian ``e_t``.

    ``phi`` is the lag-1 autocorrelation (|phi| < 1). ``seed`` may be an int or
    a ``numpy.random.Generator`` (successive calls then continue its stream).
    The first ``burn`` samples are discarded so the series starts stationary.
    """
    if not -1.0 < phi < 1.0:
        raise ValueError("phi must lie in (-1, 1)")
    e = np.random.default_rng(seed).standard_normal(n + burn)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, e.size):
        x[i] = phi * x[i - 1] + e[i]
    return x[burn:]
