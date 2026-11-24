"""The plotting functions run on a tiny synthetic rollout (no _data needed)."""
import numpy as np

from resflow import plotting as P
from resflow.plotting import style as fs


def _rollout(tmp_path, ni=8, nj=6, ndays=6):
    rng = np.random.default_rng(0)
    ii, jj = np.meshgrid(np.arange(1, ni + 1), np.arange(1, nj + 1))
    ijk = np.c_[ii.ravel(), jj.ravel(), np.ones(ni * nj, int)]
    nc = len(ijk)
    true = rng.random((ndays, nc, 8)).astype("float32") + 1.0
    pred = true + 0.01 * rng.standard_normal(true.shape).astype("float32")
    stat = np.zeros((nc, 7), "float32")
    stat[:, 3] = 10 ** rng.uniform(0, 3, nc)
    f = tmp_path / "rollout_x.sr3.npz"
    days = np.arange(ndays, dtype=float)
    np.savez(f, days=days, pred_days=days, true=true, pred=pred, ijk=ijk, stat=stat,
             var_names=np.array(["Rs", "P", "So", "Sg", "Sw", "Vo_std", "Vg_std", "Vw_std"]),
             well_marker_ij=np.array([[2.0, 2.0], [6.0, 5.0]]), well_marker_type=np.array(["producer", "injector"]),
             well_marker_name=np.array(["P1", "I1"]))
    return f


def test_maps_and_perm(tmp_path):
    fs.setup(60)
    f = _rollout(tmp_path)
    days = (1, 3, 5)
    ro = P.read_rollout_npz(f, days=days)
    assert ro.true.shape == (3, 48, 8)
    mk = (ro.marker_ij, ro.marker_type, ro.marker_name)
    fig = P.plot_permeability(ro.kx, ro.ijk, mk)
    fs.save(fig, tmp_path / "perm.png", 60)
    fig = P.plot_state_maps(P.state_fields(ro), days, ro.ijk, mk)
    fs.save(fig, tmp_path / "maps.png", 60)
    assert (tmp_path / "perm.png").stat().st_size > 0 and (tmp_path / "maps.png").stat().st_size > 0


def test_cumulative(tmp_path):
    fs.setup(60)
    d = np.arange(1, 50, dtype=float)
    ser = {}
    for comp, wells in (("Oil production", ["Producer 1"]), ("Water production", ["Producer 1"]),
                        ("Water injection", ["Water Injector 1"])):
        for w in wells:
            t = np.cumsum(np.ones_like(d))
            ser[(comp, w)] = (d, t, 1.05 * t)
    fig = P.plot_cumulative(ser, tmax=50)
    fs.save(fig, tmp_path / "cum.png", 60)
    assert (tmp_path / "cum.png").exists()
