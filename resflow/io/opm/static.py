"""Static reservoir graph (``res_static.pt``) from OPM ``EGRID``/``INIT`` data.

Same content as the IMEX converter: node features ``stat = [depth, pore volume, porosity, permI, permJ, permK,
KR table]``, undirected edges listed in both directions (the forward half sorted by cell then +I, +J, +K), edge
transmissibility in md.m (OPM ``TRAN`` / 0.00852702), centre-to-face edge distance and the pore-volume /
transmissibility modified distance of the IMEX converter (reference grid: 100, 100, 1 md and porosity 0.1).
"""
import numpy as np
import torch
from torch_geometric.data import Data

from .reader import DARCY_METRIC

REF_PERM = (100.0, 100.0, 1.0)
REF_PORO = 0.1


def _cell_corners(grid):
    """Corner points of every cell of the full grid, shape (nz, ny, nx, 2[k], 2[j], 2[i], 3), straight pillars."""
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    coord = grid["coord"].reshape(ny + 1, nx + 1, 6)
    z = grid["zcorn"].reshape(nz, 2, ny, 2, nx, 2)  # k, face, j, jj, i, ii
    # pillar of corner (j+jj, i+ii)
    pil = np.empty((ny, 2, nx, 2, 6))
    for jj in range(2):
        for ii in range(2):
            pil[:, jj, :, ii] = coord[jj:jj + ny, ii:ii + nx]
    pil = np.broadcast_to(pil[None, None], (nz, 2, ny, 2, nx, 2, 6))
    x0, y0, z0, x1, y1, z1 = (pil[..., i] for i in range(6))
    dz = np.where(z1 == z0, 1.0, z1 - z0)
    x = x0 + (z - z0) * (x1 - x0) / dz
    y = y0 + (z - z0) * (y1 - y0) / dz
    c = np.stack([x, y, z], axis=-1)  # (k, face, j, jj, i, ii, 3)
    return c.transpose(0, 2, 4, 1, 3, 5, 6)  # (k, j, i, face, jj, ii, 3)


def _face(corners, axis, plus):
    """Four corners of the +/- face of each cell along axis 0=I, 1=J, 2=K -> (..., 4, 3), in cyclic order."""
    idx = 1 if plus else 0
    if axis == 0:
        f = corners[..., :, :, idx, :]  # (face, jj, 3) at ii=idx
        return np.stack([f[..., 0, 0, :], f[..., 0, 1, :], f[..., 1, 1, :], f[..., 1, 0, :]], axis=-2)
    if axis == 1:
        f = corners[..., :, idx, :, :]  # (face, ii, 3)
        return np.stack([f[..., 0, 0, :], f[..., 0, 1, :], f[..., 1, 1, :], f[..., 1, 0, :]], axis=-2)
    f = corners[..., idx, :, :, :]  # (jj, ii, 3)
    return np.stack([f[..., 0, 0, :], f[..., 0, 1, :], f[..., 1, 1, :], f[..., 1, 0, :]], axis=-2)


def _half_geometry(corners, axis, plus):
    """Distance cell centre -> face centre, and |A.d|/(d.d) of the cell's own face (ECLIPSE half-transmissibility
    geometry), for every cell of the full grid."""
    centre = corners.reshape(*corners.shape[:3], 8, 3).mean(axis=3)
    f = _face(corners, axis, plus)
    fc = f.mean(axis=-2)
    area = 0.5 * np.cross(f[..., 2, :] - f[..., 0, :], f[..., 3, :] - f[..., 1, :])
    d = fc - centre
    dd = (d * d).sum(-1)
    ad = np.abs((area * d).sum(-1))
    return np.sqrt(dd), ad / dd


def build_static(grid, kr_index=1):
    """Return ``(Data, n_cells)``; ``grid`` is the output of :func:`resflow.io.opm.reader.read_grid`."""
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    n_full = nx * ny * nz
    act = grid["actnum"]
    ini = grid["init"]
    n_act = int(act.sum())
    cell_of = np.full(n_full, -1, dtype=np.int64)
    cell_of[act] = np.arange(n_act)

    def full(name, fill=0.0):
        a = np.full(n_full, fill)
        a[act] = ini[name]
        return a.reshape(nz, ny, nx)

    ntg = full("NTG", 1.0)
    tran_opm = [full("TRANX"), full("TRANY"), full("TRANZ")]
    poro, porv = full("PORO"), full("PORV")
    por_vol = porv * ntg  # same product as the IMEX converter (BLOCKPVOL x NET/GROSS)
    with np.errstate(divide="ignore", invalid="ignore"):
        bulk = porv / (poro * ntg)
    ratio_pv = por_vol / (bulk * REF_PORO)

    corners = _cell_corners(grid)
    kk, jj_, ii_ = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    ids = (ii_ + nx * jj_ + nx * ny * kk)
    fwd, e_tran, e_dist, e_mod = [], [], [], []
    for axis, shape_ok, step in ((0, nx > 1, 1), (1, ny > 1, nx), (2, nz > 1, nx * ny)):
        if not shape_ok:
            continue
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[2 - axis] = slice(0, (nx, ny, nz)[axis] - 1)
        sl_b[2 - axis] = slice(1, None)
        sl_a, sl_b = tuple(sl_a), tuple(sl_b)
        ia, ib = ids[sl_a].ravel(), ids[sl_b].ravel()
        ok = act[ia] & act[ib]
        ia, ib = ia[ok], ib[ok]
        da, ga = _half_geometry(corners, axis, True)
        db, gb = _half_geometry(corners, axis, False)
        da, ga = da[sl_a].ravel()[ok], ga[sl_a].ravel()[ok]
        db, gb = db[sl_b].ravel()[ok], gb[sl_b].ravel()[ok]
        t_ref = 1.0 / (1.0 / (REF_PERM[axis] * ga) + 1.0 / (REF_PERM[axis] * gb))
        tran = np.ravel(tran_opm[axis])[ia] / DARCY_METRIC
        d_orig = da + db
        d_mod = (da * np.sqrt(np.ravel(ratio_pv)[ia]) + db * np.sqrt(np.ravel(ratio_pv)[ib])) / np.sqrt(tran / t_ref)
        fwd.append(np.stack([ia, ib]))
        e_tran.append(tran)
        e_dist.append(d_orig)
        e_mod.append(d_mod)
    edge = np.concatenate(fwd, axis=1)
    tran, dist, dmod = np.concatenate(e_tran), np.concatenate(e_dist), np.concatenate(e_mod)
    order = np.lexsort((edge[1], edge[0]))  # by first cell, then neighbour (+I < +J < +K)
    edge, tran, dist, dmod = edge[:, order], tran[order], dist[order], dmod[order]
    edge_a = np.stack([cell_of[edge[0]], cell_of[edge[1]]])
    edge_both = np.concatenate([edge_a, edge_a[::-1]], axis=1)

    depth = ini["DEPTH"]
    stat = np.stack([depth, ini["PORV"], ini["PORO"], ini["PERMX"], ini["PERMY"], ini["PERMZ"],
                     np.full(n_act, float(kr_index))], axis=1)
    ijk_full = np.stack([ii_.ravel() + 1, jj_.ravel() + 1, kk.ravel() + 1], axis=1)[act]
    num_edges = edge_both.shape[1]
    graph = Data(
        stat=torch.tensor(stat, dtype=torch.float32),
        edge_index=torch.tensor(edge_both, dtype=torch.int64),
        edge_tran=torch.tensor(np.concatenate([tran, tran]).reshape(-1, 1), dtype=torch.float32),
        edge_dist=torch.tensor(np.concatenate([dist, dist]), dtype=torch.float32),
        edge_dist_mod=torch.tensor(np.concatenate([dmod, dmod]), dtype=torch.float32),
        ijk=torch.tensor(ijk_full, dtype=torch.int64),
        num_nodes=n_act,
        num_edges=num_edges,
    )
    mask = np.zeros(n_act, dtype=bool)
    mask[np.unique(edge_both.reshape(-1))] = True
    graph.mask = torch.tensor(mask).unsqueeze(-1)
    graph.validate(raise_on_error=True)
    if graph.has_self_loops() or graph.is_directed():
        raise ValueError("Static graph has self loops or is directed")
    return graph, n_act
