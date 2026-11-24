"""Helpers that compare converted raw folders and processed datasets with golden ones.

Integers, booleans and strings (JSON content, names, indices, masks) must be equal; floats
must satisfy allclose(rtol=1e-6). The maximum absolute and relative differences are recorded.
"""
import json
from pathlib import Path

import numpy as np
import torch

RTOL = 1e-6
ATOL = 0.0
# Keys of the golden files that the port deliberately does not write.
REMOVED_MODEL_KEYS = {'file_path'}                      # absolute machine path
REMOVED_STATIC_KEYS = {'shortest_path_dist', 'shortest_path_dist_mod'}  # dense N x N matrices
REMOVED_DATA_JSON_KEYS = {'path'}                       # absolute machine path
REMOVED_PROCESSED_KEYS = {'val_mask', 'test_mask'}      # all-False, never read


class Stats:
    """Accumulates the comparison result."""

    def __init__(self):
        self.n_tensors = 0
        self.n_float = 0
        self.n_exact = 0
        self.max_abs = 0.0
        self.max_rel = 0.0
        self.problems = []

    def add_problem(self, msg):
        if len(self.problems) < 25:
            self.problems.append(msg)
        else:
            self.problems.append(None)

    def merge(self, other):
        self.n_tensors += other.n_tensors
        self.n_float += other.n_float
        self.n_exact += other.n_exact
        self.max_abs = max(self.max_abs, other.max_abs)
        self.max_rel = max(self.max_rel, other.max_rel)
        self.problems += other.problems

    @property
    def n_problems(self):
        return len(self.problems)

    def summary(self):
        return (f"{self.n_tensors} tensors ({self.n_float} float, {self.n_exact} of them bit-identical), "
                f"max abs diff {self.max_abs:.3g}, max rel diff {self.max_rel:.3g}, "
                f"{self.n_problems} problems")


def compare_tensors(a, b, label, stats):
    """Compare two tensors: dtype, shape, exact for non-floats, allclose for floats."""
    stats.n_tensors += 1
    if a.dtype != b.dtype:
        stats.add_problem(f"{label}: dtype {a.dtype} vs {b.dtype}")
        return
    if a.shape != b.shape:
        stats.add_problem(f"{label}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
        return
    if not a.is_floating_point():
        if not torch.equal(a, b):
            stats.add_problem(f"{label}: integer/bool tensors differ")
        return
    stats.n_float += 1
    a_ = a.double()
    b_ = b.double()
    same_nan = torch.isnan(a_) == torch.isnan(b_)
    if not bool(same_nan.all()):
        stats.add_problem(f"{label}: NaN pattern differs")
        return
    if torch.equal(a, b) or bool(((a_ == b_) | torch.isnan(a_)).all()):
        stats.n_exact += 1
        return
    diff = (a_ - b_).abs()
    diff = torch.nan_to_num(diff)
    abs_d = float(diff.max())
    rel_d = float((diff / b_.abs().clamp(min=1e-30)).max())
    stats.max_abs = max(stats.max_abs, abs_d)
    stats.max_rel = max(stats.max_rel, rel_d)
    if not torch.allclose(a_, b_, rtol=RTOL, atol=ATOL, equal_nan=True):
        stats.add_problem(f"{label}: not allclose (max abs {abs_d:.3g}, max rel {rel_d:.3g})")


def compare_values(a, b, label, stats):
    """Compare nested structures of tensors / numbers / strings."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            stats.add_problem(f"{label}: type {type(a).__name__} vs {type(b).__name__}")
            return
        compare_tensors(a, b, label, stats)
    elif isinstance(a, dict):
        if set(a) != set(b):
            stats.add_problem(f"{label}: keys {sorted(set(a) ^ set(b))} differ")
            return
        for k in a:
            compare_values(a[k], b[k], f"{label}.{k}", stats)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            stats.add_problem(f"{label}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare_values(x, y, f"{label}[{i}]", stats)
    elif isinstance(a, float) and isinstance(b, float):
        stats.n_tensors += 1
        if a != b and not np.isclose(a, b, rtol=RTOL, atol=ATOL):
            stats.add_problem(f"{label}: {a} vs {b}")
    else:
        stats.n_tensors += 1
        if a != b:
            stats.add_problem(f"{label}: {a!r} vs {b!r}")


def data_items(data):
    """{key: value} of a torch_geometric Data (any version)."""
    return {k: data[k] for k in data.keys()}  # pylint: disable=consider-using-dict-items


def _load(path):
    return torch.load(path, weights_only=False)


def compare_raw_folder(golden, port, stats):
    """Compare one converted simulation folder."""
    golden, port = Path(golden), Path(port)
    g_files = sorted(p.relative_to(golden).as_posix() for p in golden.rglob('*') if p.is_file())
    p_files = sorted(p.relative_to(port).as_posix() for p in port.rglob('*') if p.is_file())
    if g_files != p_files:
        stats.add_problem(f"{golden.name}: files differ: only golden {sorted(set(g_files) - set(p_files))}, "
                          f"only port {sorted(set(p_files) - set(g_files))}")
    for rel in sorted(set(g_files) & set(p_files)):
        label = f"{golden.name}/{rel}"
        if rel.endswith('.json'):
            a = json.loads((golden / rel).read_text())
            b = json.loads((port / rel).read_text())
            if rel == 'model.json':
                a = {k: v for k, v in a.items() if k not in REMOVED_MODEL_KEYS}
                if 'file_path' in b:
                    stats.add_problem(f"{label}: file_path must not be stored")
            compare_values(a, b, label, stats)
        elif rel == 'res_static.pt':
            a = data_items(_load(golden / rel))
            b = data_items(_load(port / rel))
            a = {k: v for k, v in a.items() if k not in REMOVED_STATIC_KEYS}
            compare_values(a, b, label, stats)
        elif rel.endswith('.pth') or rel.endswith('.pt'):
            compare_values(_load(golden / rel), _load(port / rel), label, stats)
        else:
            stats.add_problem(f"{label}: unknown file type")


def split_sim_folders(root):
    """{(split, simulation): path} of the raw folders under a converted root."""
    out = {}
    for split in ('train', 'val', 'test'):
        raw = Path(root) / split / 'raw'
        if raw.is_dir():
            for d in sorted(raw.iterdir()):
                if d.is_dir():
                    out[(split, d.name)] = d
    return out


def compare_raw_trees(golden_root, port_root, only=None):
    """Compare every simulation folder; `only(split, name) -> bool` selects a subset."""
    stats = Stats()
    g = split_sim_folders(golden_root)
    p = split_sim_folders(port_root)
    if set(g) != set(p):
        stats.add_problem(f"simulation sets differ: only golden {sorted(set(g) - set(p))}, "
                          f"only port {sorted(set(p) - set(g))}")
    n = 0
    for key in sorted(set(g) & set(p)):
        if only is not None and not only(*key):
            continue
        compare_raw_folder(g[key], p[key], stats)
        n += 1
    return stats, n


def compare_dataset_split(golden, port, stats, subset=False):
    """Compare one built split: data.json and every processed file.

    With `subset`, only the entries (file numbers) present in the port's data.json are
    compared (the port was built from a subset of the golden simulations).
    """
    golden, port = Path(golden), Path(port)
    label = f"{golden.parent.name}/{golden.name}"
    a = json.loads((golden / 'data.json').read_text())
    b = json.loads((port / 'data.json').read_text())
    if subset:
        a = {k: v for k, v in a.items() if k in b}
    a = {k: {kk: vv for kk, vv in v.items() if kk not in REMOVED_DATA_JSON_KEYS} for k, v in a.items()}
    if any(REMOVED_DATA_JSON_KEYS & set(v) for v in b.values()):
        stats.add_problem(f"{label}/data.json: path must not be stored")
    compare_values(a, b, f"{label}/data.json", stats)
    g_files = sorted(p_.name for p_ in (golden / 'processed').glob('data_*.pt'))
    p_files = sorted(p_.name for p_ in (port / 'processed').glob('data_*.pt'))
    if subset:
        g_files = [f for f in g_files if f in p_files]
    if g_files != p_files:
        stats.add_problem(f"{label}: processed files differ")
    for name in sorted(set(g_files) & set(p_files)):
        da = data_items(_load(golden / 'processed' / name))
        db = data_items(_load(port / 'processed' / name))
        da = {k: v for k, v in da.items() if k not in REMOVED_PROCESSED_KEYS}
        if REMOVED_PROCESSED_KEYS & set(db):
            stats.add_problem(f"{label}/{name}: val_mask/test_mask must not be stored")
        compare_values(da, db, f"{label}/{name}", stats)
