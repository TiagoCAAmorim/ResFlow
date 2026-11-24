"""Well layouts: optimise positions per permeability patch and well count, then pick the injectors.

Every well is first optimised as a producer (simulated annealing on the discounted drained volume Y_at,
minimum distance between wells enforced). With `k_min` > 0, cells below it are barriers and the wells fall into
regions of mutually connected cells (I/J neighbours). Roles are then assigned from the optimised positions:
  - every region with wells gets one injector first: its lowest-Y_at position;
  - each further injector goes to the region with the lowest #injectors/#producers ratio (a region always keeps
    a producer), at its lowest-Y_at position that is at least `inj_factor * d_min` from every injector already
    chosen and not "behind" an injector (an injector between it and a producer of its region, within
    `SHADOW_DEG`); the distance rule, then the shadow rule, are dropped if no position qualifies;
  - the other positions are producers (P1 = highest Y_at).
A layout is invalid if a well sits on a cell below `k_min` (such a well scores Y_at = 0 while annealing), is
isolated from every other well, has Y_at below `score_min_frac` x the mean Y_at of the other wells, or cannot
get a producer and an injector in every region; the annealing is then retried with another seed, up to
`max_attempts` times, and the well count is skipped if none is valid.
If the total number of wells does not fit with d_min, wells are removed one at a time, a producer first, then an
injector, and so on, down to (1, 1); injectors only go below 2 when there are 2 producers (4P2I -> 3P2I -> 2P2I ->
2P1I -> 1P1I, never 4P1I). A revised count equal to one of the requested counts, or already generated for the
patch, is skipped.
"""
import zlib
from pathlib import Path

import numpy as np

from defaults import DEFAULT_SEED
from grid_geometry import write_corners_file
from well_placement.config import GridConfig, IndexConfig, SAConfig
from well_placement.connections import build_connections
from well_placement.distances import DistanceEngine
from well_placement.model_io import load_model, read_inc_values
from well_placement.optimize import InvalidLayoutError, WellObjective, max_wells_fit, optimize_wells, respects_d_min
from well_placement.quality import compute_index

SHADOW_DEG = 30.0   # an injector within this angle of the line to a producer, and nearer, shadows what is behind it
DEFAULT_COUNTS = [(2, 1), (3, 2), (4, 2), (5, 3), (6, 4), (7, 4), (8, 4)]


def parse_counts(text):
    """'2/1,3/2' -> [(2, 1), (3, 2)]. Producer-only counts such as '1/0' are allowed."""
    out = [tuple(int(v) for v in t.split("/")) for t in text.split(",") if t.strip()]
    if any(len(c) != 2 or c[0] < 1 or c[1] < 0 for c in out):
        raise ValueError(f"well counts must be P/I pairs with P >= 1 and I >= 0: {text!r}")
    return out


def reduced_counts(p, i):
    """Remove one producer, then one injector, and so on, down to (1, 1): (p-1, i), (p-1, i-1), ...

    Injectors only go below 2 when there are 2 producers, so 4P2I gives 3P2I, 2P2I, 2P1I, 1P1I (never 4P1I).
    """
    turn_producer = True
    while p > 1 or i > 1:
        can_drop_injector = i > 1 and (i > 2 or p <= 2)
        if turn_producer and p > 1:
            p -= 1
            turn_producer = False
        elif can_drop_injector:
            i -= 1
            turn_producer = True
        else:           # the injector's turn, but it may not go below 2 yet: keep dropping producers
            p -= 1
        yield p, i


def resolve_counts(requested, max_fit):
    """Final counts for one patch. Returns (cases, skipped).

    cases:   list of (requested (P, I), final (P, I)); final differs from requested when reduced.
    skipped: list of (requested (P, I), reason).
    """
    asked = set(requested)
    cases, skipped, produced = [], [], set()
    for pi in requested:
        if sum(pi) <= max_fit and pi not in produced:
            cases.append((pi, pi))
            produced.add(pi)
    for pi in requested:
        if sum(pi) <= max_fit:
            continue
        final = next((c for c in reduced_counts(*pi) if sum(c) <= max_fit), None)
        if final is None:
            skipped.append((pi, f"no room for 1P+1I (room for {max_fit} wells)"))
        elif final in asked:
            skipped.append((pi, f"revised count {final[0]}/{final[1]} is a requested count"))
        elif final in produced:
            skipped.append((pi, f"revised count {final[0]}/{final[1]} already generated"))
        else:
            cases.append((pi, final))
            produced.add(final)
    return cases, skipped


def _behind_injector(w, injectors, producers, xy, shadow_deg):
    """True if an injector lies between position w and a producer, within shadow_deg of the line to it."""
    cos_limit = np.cos(np.radians(shadow_deg))
    for i in injectors:
        v_inj = xy[i] - xy[w]
        for q in producers:
            v_prod = xy[q] - xy[w]
            n_inj, n_prod = np.linalg.norm(v_inj), np.linalg.norm(v_prod)
            if 0 < n_inj < n_prod and v_inj @ v_prod >= cos_limit * n_inj * n_prod:
                return True
    return False


def assign_roles(y_at, xy, n_inj, d_min, inj_factor=1.5, groups=None, shadow_deg=SHADOW_DEG):
    """Indices (into the optimised wells) of the injectors (worst first) and producers (best first).

    groups: region label of each well (None = one region). Every region gets an injector first, at its
    lowest-Y_at well; the rest go to the region with the lowest #injectors/#producers ratio.
    Returns (injectors, producers, n_spacing_relaxed, n_behind_injector), the last two counting injectors
    placed although the distance rule / the shadow rule could not be met.
    """
    n = len(y_at)
    if n_inj == 0:      # producer-only case (e.g. the 1P0I test layout): nothing to place
        return [], list(reversed(np.argsort(y_at, kind="stable"))), 0, 0
    groups = np.zeros(n, dtype=int) if groups is None else np.asarray(groups)
    regions = sorted(set(groups.tolist()))
    members = {r: [w for w in np.argsort(y_at, kind="stable") if groups[w] == r] for r in regions}
    if n_inj < len(regions):
        raise ValueError(f"{n_inj} injector(s) cannot serve {len(regions)} separate regions")
    inj, relaxed, behind = [], 0, 0

    def pick(region):
        nonlocal relaxed, behind
        free = [w for w in members[region] if w not in inj]
        here = [j for j in inj if groups[j] == region]
        spaced = [w for w in free if all(np.linalg.norm(xy[w] - xy[j]) >= inj_factor * d_min for j in inj)]
        shadowed = {w: _behind_injector(w, here, [q for q in free if q != w], xy, shadow_deg) for w in free}
        choice = next((w for w in spaced if not shadowed[w]), None)
        if choice is None:
            choice = next((w for w in spaced), None)
        if choice is None:
            choice = next((w for w in free if not shadowed[w]), free[0])
        relaxed += choice not in spaced
        behind += shadowed[choice]
        inj.append(choice)

    for r in regions:                       # first injector of each region: its worst position
        inj.append(members[r][0])
    for _ in range(n_inj - len(regions)):
        eligible = [r for r in regions if len(members[r]) - sum(groups[j] == r for j in inj) >= 2]
        if not eligible:
            raise ValueError("no region has room for another injector")
        ratio = lambda r: (sum(groups[j] == r for j in inj) / (len(members[r]) - sum(groups[j] == r for j in inj)),
                           -len(members[r]), r)
        pick(min(eligible, key=ratio))
    order = list(np.argsort(y_at, kind="stable"))
    prod = [w for w in reversed(order) if w not in inj]
    return inj, prod, relaxed, behind


def write_base_dat(path, corners_path, ni, nj):
    """Minimal CMG deck for the placement code: grid + constant porosity; PERM is overridden per run."""
    Path(path).write_text(f"CORNER {ni} {nj} 1\nINCLUDE '{Path(corners_path).resolve()}'\n\n"
                          "NULL CON 1\nPOR CON 0.10\nPERMI CON 100.0\nPERMJ EQUALSI\nPERMK CON 0.1\n")


def ensure_grid(grid_dir, ni, nj, cell, nk=1):
    """CORNERS include and base deck of an ni x nj grid in the common `grid_dir`; returns the base deck path.

    The placement works on one layer. A patch of nk layers is collapsed into one layer nk times as thick (see
    collapse_layers), so the grid has dz = nk * cell dz; its files get the suffix _nk<nk> when nk > 1.
    """
    grid_dir = Path(grid_dir)
    grid_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if nk == 1 else f"_nk{nk}"
    corners = grid_dir / f"CORNERS_{ni}x{nj}x1{tag}.inc"
    base = grid_dir / f"base_{ni}x{nj}{tag}.dat"
    if not base.exists():
        dx, dy, dz = cell
        write_corners_file((ni, nj, 1), (dx, dy, dz * nk), (0.0, 0.0), 5450.0, corners, include_comments=False)
        write_base_dat(base, corners, ni, nj)
    return base


def collapse_layers(perm_file, ni, nj, nk, out_file):
    """Write the layer-averaged permeability of an ni x nj x nk patch file as a one-layer include file.

    The map the placement sees for a multi-layer patch: lateral transmissibilities add over the layers
    (sum k_l dz = mean(k) * nk dz) and so do the oil volumes (nk dz of the same porosity), i.e. the same as one
    layer of permeability mean(k) and thickness nk * dz (the ensure_grid grid). Vertical flow is ignored.
    """
    v = read_inc_values(perm_file)
    if v.size != ni * nj * nk:
        raise ValueError(f"{perm_file}: {v.size} values, expected {ni}x{nj}x{nk}")
    mean_k = v.reshape(nk, nj, ni).mean(axis=0)
    Path(out_file).write_text("** layer-averaged permeability for the well placement\n"
                              + "\n".join(" ".join(f"{x:.6g}" for x in row) for row in mean_k) + "\n")


def place_patch(family_id, perm_file, ni, nj, requested, grid_dir, *, nk=1, cell=(50.0, 50.0, 5.0), d_min=500.0,
                rate=0.10, x_base=15.0, w_bal=0.25, w_dist=1.0, n_iter=2000, seed=DEFAULT_SEED, inj_factor=1.5,
                k_min=0.0, score_min_frac=0.0, max_attempts=3, dist_limit=None):
    """Layouts of one patch for every requested count. Returns (rows, skipped, room).

    rows: list of dicts (one per simulated well count); skipped: list of (requested, reason).
    The grid files are shared by all patches of the same size (see ensure_grid).
    """
    base = ensure_grid(grid_dir, ni, nj, cell, nk)
    perm_file = Path(perm_file).resolve()
    collapsed = None
    if nk > 1:
        collapsed = Path(grid_dir) / f"_collapsed_{family_id}.inc"
        collapse_layers(perm_file, ni, nj, nk, collapsed)
        perm_file = collapsed.resolve()
    try:
        model = load_model(base, GridConfig(cell_size="corners", overrides={"PERMI": perm_file, "PERMJ": perm_file}),
                           label=family_id, verbose=False)
    finally:
        if collapsed is not None:
            collapsed.unlink(missing_ok=True)
    engine = DistanceEngine(model, build_connections(model.geometry), limit=dist_limit, verbose=False)   # cells farther than dist_limit (equivalent m) count as unreachable
    y_at, _, _ = compute_index(model, engine, IndexConfig(rate=rate, x_base=x_base), verbose=False)

    tight = np.sqrt(model.permi.ravel() * model.permj.ravel()) < k_min
    room = max_wells_fit(np.where(tight, np.nan, y_at), model, d_min)   # wells only go on cells >= k_min
    cases, skipped = resolve_counts(requested, room)
    patch_seed = (seed + zlib.crc32(family_id.encode())) % 2**31
    optimised, rows = {}, []
    for req, (n_p, n_i) in cases:
        n = n_p + n_i
        if (n_p, n_i) not in optimised:
            sa = SAConfig(n_wells=n, n_injectors=n_i, d_min=d_min, w_bal=w_bal, w_dist=w_dist, n_iter=n_iter, seed=patch_seed,
                          k_min=k_min, score_min_frac=score_min_frac, max_attempts=max_attempts)
            try:
                optimised[(n_p, n_i)] = optimize_wells(model, engine, y_at, rate, x_base, sa, verbose=False)
            except InvalidLayoutError as error:
                optimised[(n_p, n_i)] = error
        opt = optimised[(n_p, n_i)]
        if isinstance(opt, InvalidLayoutError):
            skipped.append((req, f"{n_p}P{n_i}I: {opt}"))
            continue
        assert respects_d_min(opt.wells, model.geometry.centre, d_min), "minimum distance violated"
        groups = WellObjective._flow_regions(model, tight)[opt.wells] if k_min > 0 else None
        inj, prod, relaxed, behind = assign_roles(opt.y_at_per_producer, opt.wells_xy, n_i, d_min, inj_factor, groups)
        row = dict(family_id=family_id, case_id=f"{family_id}_{n_p}P{n_i}I", n_prod=n_p, n_inj=n_i,
                   requested_prod=req[0], requested_inj=req[1], reduced=req != (n_p, n_i),
                   inj_spacing_relaxed=relaxed, inj_behind_injector=behind, score=float(opt.score))
        for rank, w in enumerate(prod, 1):
            row[f"P{rank}I"], row[f"P{rank}J"] = int(opt.wells_ijk[w][0]), int(opt.wells_ijk[w][1])
        for rank, w in enumerate(inj, 1):
            row[f"I{rank}I"], row[f"I{rank}J"] = int(opt.wells_ijk[w][0]), int(opt.wells_ijk[w][1])
        rows.append(row)
    return rows, skipped, room
