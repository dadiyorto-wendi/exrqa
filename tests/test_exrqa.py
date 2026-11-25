import numpy as np
import pytest

import exrqa as xr


def test_embed_shape_and_values():
    X = xr.embed(np.arange(10.0), m=3, tau=2)
    assert X.shape == (6, 3)
    np.testing.assert_array_equal(X[0], [0, 2, 4])
    np.testing.assert_array_equal(X[-1], [5, 7, 9])


def test_distance_metrics_agree_on_1d():
    x = np.random.default_rng(0).normal(size=20)
    D = xr.distance_matrix(x)
    np.testing.assert_allclose(D, np.abs(x[:, None] - x[None, :]), atol=1e-12)
    np.testing.assert_allclose(xr.distance_matrix(x, "maximum"), D)


def test_local_minima_row_profile():
    D = np.array([[0.0, 3.0, 1.0, 2.0, 0.5, 4.0]])
    np.testing.assert_array_equal(xr.local_minima(D)[0], [1, 0, 1, 0, 1, 0])


def test_lm_family_nesting_and_separation():
    x = xr.add_noise(xr.zscore(xr.lorenz(400, rho=28, dt=0.02)[:, 0]), 20, seed=1)
    D = xr.distance_matrix(xr.embed(x, 3, 3))
    eps = xr.eps_from_percentile(D, 5)
    lm, lmt, lm2p = xr.local_minima_rp(D), xr.local_minima_rp(D, eps), xr.local_minima_rp(D, eps, 10)
    assert np.all(lmt <= lm) and np.all(lm2p <= lmt)
    assert np.all(D[lmt.astype(bool)] <= eps)
    for row in lm2p:                         # kept minima are >= tau_m apart
        idx = np.flatnonzero(row)
        assert np.all(np.diff(idx) >= 10)
    assert np.all(np.diag(lm2p) == 1)        # LOI (distance 0) always kept


def test_recurrence_plot_dispatch():
    x = np.sin(np.linspace(0, 20 * np.pi, 300))
    for method in ["threshold", "lm", "lmt", "lm2p"]:
        R = xr.recurrence_plot(x, m=2, tau=5, method=method)
        assert R.shape == (295, 295)
    with pytest.raises(ValueError):
        xr.recurrence_plot(x, method="nope")


def test_rqa_on_synthetic_lines():
    R = np.eye(20, dtype=np.uint8)
    R[np.arange(5), np.arange(5) + 8] = 1    # one diagonal line, length 5
    R[15, 2] = 1                             # one isolated point
    q = xr.rqa(R, theiler=1)
    assert q["Lmax"] == 5
    assert q["DET"] == pytest.approx(5 / 6)
    R2 = np.zeros((10, 10), np.uint8)
    R2[2:6, 8] = 1                           # vertical line, length 4
    assert xr.rqa(R2)["LAM"] == pytest.approx(1.0)


def test_ddet_separates_noise_from_lorenz():
    n = 600
    x = xr.zscore(xr.lorenz(n + 20, rho=80, dt=0.01)[:, 0])
    g = np.random.default_rng(3).standard_normal(n + 20)
    d_lor = xr.ddet(xr.recurrence_plot(x, m=3, tau=10, method="lm2p", tau_m=10))
    d_noise = xr.ddet(xr.recurrence_plot(g, m=3, tau=10, method="lm2p", tau_m=10))
    assert d_noise.ddet < -0.5 < 0.0 < d_lor.ddet
    assert d_lor.rr_w.shape == d_lor.dd_w.shape == d_lor.offsets.shape


def test_ddet_perfect_line_window():
    N = 50
    R = np.zeros((N, N), np.uint8)
    R[np.arange(N - 10), np.arange(N - 10) + 10] = 1    # one full line at lag 10
    rr, dd, off = xr.diagonal_windows(R, ws=2, tau_w=2, d_max=2, theiler=1)
    w = np.flatnonzero(off == 9)[0]                     # window covering diagonals 9, 10
    assert dd[w] == pytest.approx((N - 11) / (N - 9 - (N - 11)))
    assert rr[w] == pytest.approx((N - 10) / ((N - 9) + (N - 10)))
    assert np.all(dd[off != 9] == 0)
    assert xr.diagonal_windows(R, max_lag=20)[2].max() <= 20


def test_suggest_tau_m_bounds():
    x = xr.zscore(xr.lorenz(800, rho=80, dt=0.01)[:, 0])
    s = xr.suggest_tau_m(x, m=3, tau=10, n_ref=500)
    assert 2 <= s["lower"] <= 5
    assert s["lower"] < s["recommended"] < s["upper"]


def _distinct_peak_ratio(s):
    p = xr.peak_values(s)
    return np.unique(np.round(p / (1e-3 * s.std()))).size / p.size


@pytest.mark.parametrize("gen, periodic, chaotic", [
    (lambda p, n: xr.rossler(n, c=p, dt=0.06, substeps=5)[..., 0], 2.5, 5.7),
    (lambda p, n: xr.duffing(n, gamma=p, dt=2 * np.pi / 1.2 / 100, substeps=2)[..., 0], 0.2, 0.5),
    (lambda p, n: xr.mackey_glass(n, tau=p, dt=0.2, substeps=2, transient=10000)[..., 0], 10.0, 17.0),
])
def test_simulators_periodic_vs_chaotic(gen, periodic, chaotic):
    both = gen(np.array([periodic, chaotic]), 6000)          # vectorised over the parameter
    assert both.shape == (2, 6000) and np.isfinite(both).all()
    np.testing.assert_allclose(both[0], gen(periodic, 6000), rtol=0, atol=1e-12)
    assert _distinct_peak_ratio(both[0]) < 0.25 < 0.5 < _distinct_peak_ratio(both[1])


def test_peak_values_parabolic_refinement():
    t = np.arange(0, 20, 0.37)
    p = xr.peak_values(np.cos(t))
    np.testing.assert_allclose(p, 1.0, atol=5e-3)            # raw samples would be off by up to 1.7 %


def test_ddet_iid_matches_random_rp():
    R = (np.random.default_rng(0).random((800, 800)) < 0.03).astype(np.uint8)
    d = xr.ddet(R)
    assert d.ddet == pytest.approx(xr.ddet_iid(d.rr_w.mean(), exclusive=False), abs=0.15)
    assert xr.ddet_iid(0.03) == pytest.approx(np.log(12 * 0.03), abs=0.1)   # small-RR limit
    assert xr.ddet_iid(0.03, ws=3) > xr.ddet_iid(0.03) > xr.ddet_iid(0.03, d_max=1)


def test_ddet_baseline_white_noise_and_surrogates():
    b = xr.ddet_baseline(n=620, m=3, tau=10, tau_m=10, n_real=5)
    assert -1.3 < b["mean"] < -0.8 and b["values"].shape == (5,)
    assert abs(b["mean"] - b["theory"]) < 0.15
    x = xr.zscore(xr.lorenz(620, rho=80, dt=0.01)[:, 0])
    s = xr.ddet_baseline(x=x, m=3, tau=10, tau_m=10, n_real=3)
    assert s["mean"] < 0 < xr.ddet(xr.recurrence_plot(x, m=3, tau=10, tau_m=10)).ddet
    with pytest.raises(ValueError):
        xr.ddet_baseline()


def test_ar1_noise_autocorrelation():
    x = xr.ar1_noise(5000, 0.9, seed=0)
    assert x.shape == (5000,)
    assert np.corrcoef(x[:-1], x[1:])[0, 1] == pytest.approx(0.9, abs=0.02)
    rng = np.random.default_rng(1)
    assert not np.allclose(xr.ar1_noise(100, 0.5, rng), xr.ar1_noise(100, 0.5, rng))   # a Generator continues
    with pytest.raises(ValueError):
        xr.ar1_noise(10, 1.0)


def test_ddet_baseline_det_and_red_noise():
    w = xr.ddet_baseline(n=620, m=3, tau=10, tau_m=10, n_real=4)
    assert 0 < w["det_mean"] < 0.2 and w["det_values"].shape == (4,)
    x = xr.zscore(xr.lorenz(620, rho=80, dt=0.01)[:, 0])
    r = xr.ddet_baseline(x=x, m=3, tau=10, tau_m=10, n_real=4, surrogate="ar1")
    assert r["phi"] > 0.95                       # smooth signal -> very red noise
    assert r["mean"] > w["mean"] + 0.3           # red noise looks more deterministic than white
    with pytest.raises(ValueError):
        xr.ddet_baseline(x=x, surrogate="phase")
