"""Graph dataset of simulation steps (PyTorch Geometric).

A dataset root holds one folder per split, ``train``, ``val`` and ``test``, each with a
``raw`` folder (one sub-folder per simulation, written by `resflow.data.convert`) and a
``processed`` folder written by `build_dataset`. ``data.json`` in the split folder
summarises the processed files.

Each processed file (``data_<k>.pt``) holds a series of consecutive outputs spaced exactly
`step_size` days apart, as a torch_geometric ``Data`` with:

- global_attr [11]: file index, start day, start index (i1), end index (i2), number of
  steps (i2 - i1), number of producer / gas-injector / water-injector connections
  (n_conp, n_cong, n_conw) and wells (n_wellp, n_wellg, n_wellw).
- x [n_steps + 1, n_nodes, 4]: cell state (Rs, P, So, Sg).
- weights [n_steps, n_nodes]: sampling weights; train_mask [n_steps, n_nodes]: cells chosen
  for training (see `build_dataset`).
- stat [n_nodes, 7]: depth, reference pore volume, porosity, perm I, J, K, KREL table;
  edge_index, edge_tran, edge_dist, edge_dist_mod, ijk, mask (static graph, see
  `resflow.preprocess.reservoir`).
- wellp [n_steps + 1, n_well_p, 4] (cumulative Np, Gp, Wp, BHP), wellg / wellw
  [n_steps + 1, n_well, 2] (cumulative injection, BHP); conp / cong / conw: the same per
  connection with step volumes; conp_index / cong_index / conw_index (cell of each
  connection), conp_stat / cong_stat / conw_stat [2, n_con] (depth, well index),
  wellp_index / wellg_index / wellw_index (well of each connection), wellp_stat /
  wellg_stat / wellw_stat (BHP datum depth).

``data.json`` keys per file index: name (SR3 file name), day_ini, timestep_ini,
timestep_end, n_steps, step_size, n_conp/cong/conw, n_wellp/wellg/wellw, conp/cong/conw
(connection names), prod/ginj/winj (well names).

Normalisation statistics are NOT part of the dataset: building and reading it needs
none (see `resflow.data.normalization`).

Copyright (c) Tiago Amorim. MIT License.
"""
import copy
import json
import math
import random
import string
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset  # type: ignore
from tqdm import tqdm

from ..physics.transform import Transform
from .resample import estimate_bins, reduce_product_to_limit, resample
from .scalers import asinh_fit_transform

MAX_BINS = 100000  # limit of the number of cells of the resampling histogram
UNIFORM_RATIO = 0.99  # flatness of the resampling (see resample)
SAMPLES_RATIO = 0.70  # paper value: fraction of the cell-step samples drawn for training
# Split folders of a dataset root.
SPLITS = ('train', 'val', 'test')


def _random_string(length=8):
    """Random string of fixed length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


# MARK: Resampling features
def volume_increment_features(batch, transform: Transform) -> torch.Tensor:
    """Features whose distribution the resampling flattens: [n_steps, n_nodes, 9].

    Columns: the state at the start of each step (Rs, P, So, Sg), Sw, and the increments of
    the standard volumes (dVo, dVg, dVw) over the step; each column is transformed with a
    fitted asinh scaler. Only used to choose the training samples, never stored.
    """
    t = batch.x
    t_ = t.reshape(-1, t.shape[-1])
    por_vol = batch.stat[:, 1]
    por_vol_ = por_vol.unsqueeze(0).repeat(t.shape[0], 1).reshape(-1)
    v = transform.t2vols(por_vol_, t_)
    v = v.reshape(t.shape[0], t.shape[1], v.shape[1])
    deltas = v[1:] - v[:-1]
    all_data = torch.cat((t[:-1], (1 - t[:-1, :, 2] - t[:-1, :, 3]).unsqueeze(-1), deltas), dim=-1)
    transformed = [asinh_fit_transform(all_data[..., i:i + 1]) for i in range(all_data.shape[-1])]
    return torch.cat(transformed, dim=-1)


# MARK: SimDataset
class SimDataset(Dataset):
    """Dataset of simulation steps as graphs, from the ``raw`` folder of `root`.

    Building (processing the raw folders) happens when ``processed`` is missing or when
    `build_dataset` forces it; afterwards the dataset only reads the processed files.

    Args:
        root: split folder (``<dataset>/train``) holding ``raw/`` and ``processed/``.
        min_day, max_day: window of days to use. Each file is cut to the window; reading is
            lazy, so the window can differ from the one used to build. `min_day` must be a
            multiple of `step_size`.
        samples_ratio: fraction of the cell-step samples drawn as training samples when
            processing (see build_dataset).
        step_size: days between outputs. Needed to process; when reading an existing
            dataset it is taken from ``data.json``.
        transform: callable ``transform(batch) -> [n_steps, n_nodes, F]`` giving the features
            whose distribution the resampling flattens (processing only; see
            `volume_increment_features`). If None the plain state increments are used.
        force_reload: clear ``processed`` and process again.
        resume: continue an interrupted processing: simulations already recorded in
            ``data.json`` are skipped and new ones are appended.
        log_path: progress log of the processing (default ``<root>/build.log``).
        sim_order: names (raw folder names) to process first, in this order. Processing
            order matters only to reproduce a previous random draw exactly; by default the
            simulations are processed in sorted order.
    """

    def __init__(self, root, **kwargs):
        self.min_day = kwargs.get('min_day', 0)
        self.max_day = kwargs.get('max_day', 99999)
        assert self.max_day > self.min_day, "max_day must be greater than min_day."

        self.samples_ratio = kwargs.get('samples_ratio', SAMPLES_RATIO)
        assert 0.0 < self.samples_ratio <= 1.0, "samples_ratio must be in (0.0, 1.0]."

        self.read_step_size = 'step_size' in kwargs
        self.step_size = kwargs.get('step_size', 1.0)
        assert self.step_size > 0, "step_size must be a positive number."
        # An off-grid min_day would make get_days list days that are not simulation outputs
        # (max_day may be off-grid: the last window is just truncated).
        assert math.isclose(round(self.min_day / self.step_size) * self.step_size,
                            self.min_day, abs_tol=1e-6), (
            f"min_day ({self.min_day}) must be a multiple of step_size ({self.step_size}).")

        self.sim_order = list(kwargs.get('sim_order') or [])
        self.resume = kwargs.get('resume', False)
        log_path = kwargs.get('log_path', None)
        self.log_path = Path(log_path) if log_path is not None else Path(root) / 'build.log'

        force_reload = kwargs.get('force_reload', False)
        self.transform_ = kwargs.get('transform', None)
        if force_reload and not self.resume:
            self._clear_processed_dir(Path(root) / 'processed', clear_metadata=True)

        self.file_list = []
        super().__init__(root, force_reload=force_reload or self.resume)
        self.data = {}

    @property
    def raw_file_names(self):
        """Raw simulation folders (processing order: `sim_order`, then sorted)."""
        raw_dir = Path(self.raw_dir)
        folders = []
        if raw_dir.is_dir():
            folders = sorted(f.name for f in raw_dir.iterdir() if f.is_dir())
        if len(folders) == 0:
            return [_random_string(20)]
        first = [n for n in self.sim_order if n in folders]
        return first + [n for n in folders if n not in first]

    @property
    def processed_file_names(self):
        if len(self.file_list) > 0:
            if not self.read_step_size:
                self.reload_details()
                self.step_size = self.data['0']['step_size']
                self.read_step_size = True
            return self.file_list
        # Read data.json to get the processed file names
        json_path = Path(self.processed_dir).parent / 'data.json'
        if json_path.is_file():
            with open(json_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)

            if len(model_data) == 0:
                raise ValueError("data.json is empty. No processed files found.")

            self.step_size = None

            self.file_list = []
            for k, v in model_data.items():
                if 'step_size' in v:
                    step_size = v['step_size']
                    if self.step_size is None:
                        self.step_size = step_size
                    elif self.step_size != step_size:
                        raise ValueError(
                            f"Inconsistent step_size read: {self.step_size} vs {step_size}.")
                else:
                    step_size = 1.0

                day_ini = v['day_ini']
                day_end = day_ini + (v['timestep_end'] - v['timestep_ini']) * step_size
                if day_ini < self.max_day and day_end > self.min_day:
                    self.file_list.append(f'data_{int(k)}.pt')

            if self.step_size is None:
                self.step_size = 1.0
            self.read_step_size = True
            if len(self.file_list) == 0:
                raise ValueError("No valid processed files meet the criteria.")

            return self.file_list
        return [_random_string(20)]

    def download(self):
        print("Converted simulation data should be in the raw directory:")
        print(f"  {self.raw_dir}")
        raise FileNotFoundError("Nothing to download: convert the simulations first.")

    def _clear_processed_dir(self, dir_=None, clear_metadata=False):
        """Clear the processed directory and optionally its data.json."""
        dir_path = Path(dir_ or self.processed_dir)
        if dir_path.is_dir():
            for file in dir_path.iterdir():
                file.unlink()
        if clear_metadata:
            json_path = dir_path.parent / 'data.json'
            if json_path.is_file():
                json_path.unlink()

    # MARK: Process
    def process(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(self.log_path, 'a', encoding='utf-8')  # pylint: disable=consider-using-with

        def log(msg):
            print(msg, file=log_fh, flush=True)

        log(f"\n{'='*80}")
        log(f"Run started {datetime.now().isoformat(timespec='seconds')} | "
            f"split={Path(self.root).name} | mode={'resume' if self.resume else 'fresh'}")
        log('=' * 80)

        data_list = {}
        completed_names = set()
        idx = 0
        if self.resume:
            json_path = Path(self.processed_dir).parent / 'data.json'
            if json_path.is_file():
                with open(json_path, 'r', encoding='utf-8') as f:
                    data_list = {int(k): v for k, v in json.load(f).items()}
                if data_list:
                    idx = max(int(k) for k in data_list.keys()) + 1
                    completed_names = {v['name'] for v in data_list.values()}
            log(f"Resuming: {len(data_list):,} step entries already recorded across "
                f"{len(completed_names):,} simulations. Continuing at idx={idx}.")
        else:
            self._clear_processed_dir()

        try:
            total_paths = len(self.raw_paths)
            pbar = tqdm(list(enumerate(self.raw_paths)), total=total_paths, unit="sim")
            for raw_idx, raw_path in pbar:
                raw_path = Path(raw_path)
                pbar.set_description(raw_path.name)

                with open(raw_path / 'model.json', "r", encoding="utf-8") as f:
                    model_data = json.load(f)
                name = model_data['name']

                log(f'Processing {raw_path.name} ({raw_idx+1:,}/{total_paths:,}).')

                if name in completed_names:
                    log(f"  Skipping {name}: already processed (resume).")
                    continue

                try:
                    idx = self._process_raw_path(raw_path, model_data, idx, data_list, log)
                except Exception:
                    log(f"  FAILED while processing {name}.")
                    raise

                completed_names.add(name)
                with open(Path(self.processed_dir).parent / 'data.json', 'w', encoding='utf-8') as f:
                    json.dump(data_list, f, indent=2)
        finally:
            log_fh.close()

        self._remove_orphan_processed_files(data_list)

    def _remove_orphan_processed_files(self, data_list):
        """Remove processed files left over from an interrupted run that data_list does not
        reference (e.g. a file that was being written when the run stopped)."""
        valid_indices = {int(k) for k in data_list.keys()}
        processed_dir = Path(self.processed_dir)
        if not processed_dir.is_dir():
            return
        for file in processed_dir.glob('data_*.pt'):
            try:
                file_idx = int(file.stem.split('_')[1])
            except (IndexError, ValueError):
                continue
            if file_idx not in valid_indices:
                file.unlink()

    def _process_raw_path(self, raw_path, model_data, idx, data_list, log):
        """Process one raw simulation folder, appending entries to data_list.

        Returns the next free idx.
        """
        raw_path = Path(raw_path)
        name = model_data['name']
        days = model_data['days']
        steps = model_data['steps']

        if len(steps) == 0:
            log(f"  No series of outputs {self.step_size:g} days apart found. Skipping...")
            return idx
        log(f"  Total groups found: {len(steps):,}.")

        model = torch.load(raw_path / 'res_static.pt', weights_only=False)

        # Well connection static data
        with open(raw_path / 'con_static.json', "r", encoding="utf-8") as f:
            con_data = json.load(f)

        def _get_con_data(type_, key=None, con_data=con_data):
            out = []
            for w in con_data[type_].values():
                for w, c in w.items():
                    out.append(w if key is None else c[key])
            return out

        model.conp_index = torch.tensor(_get_con_data('prod', 'cell'), dtype=torch.long) - 1
        model.cong_index = torch.tensor(_get_con_data('ginj', 'cell'), dtype=torch.long) - 1
        model.conw_index = torch.tensor(_get_con_data('winj', 'cell'), dtype=torch.long) - 1

        model.conp_stat = torch.tensor(
            [_get_con_data('prod', 'depth'), _get_con_data('prod', 'wi')], dtype=torch.float32)
        model.cong_stat = torch.tensor(
            [_get_con_data('ginj', 'depth'), _get_con_data('ginj', 'wi')], dtype=torch.float32)
        model.conw_stat = torch.tensor(
            [_get_con_data('winj', 'depth'), _get_con_data('winj', 'wi')], dtype=torch.float32)

        # Well static data
        with open(raw_path / 'well_static.json', "r", encoding="utf-8") as f:
            well_data = json.load(f)

        def _get_well_index(type_, well_data=well_data, con_data=con_data):
            wells = list(well_data[type_].keys())
            con_wells = []
            for w, v in con_data[type_].items():
                for _ in v.values():
                    con_wells.append(wells.index(w))
            return con_wells

        model.wellp_index = torch.tensor(_get_well_index('prod'), dtype=torch.long)
        model.wellg_index = torch.tensor(_get_well_index('ginj'), dtype=torch.long)
        model.wellw_index = torch.tensor(_get_well_index('winj'), dtype=torch.long)

        model.wellp_stat = torch.tensor(list(well_data['prod'].values()), dtype=torch.float32)
        model.wellg_stat = torch.tensor(list(well_data['ginj'].values()), dtype=torch.float32)
        model.wellw_stat = torch.tensor(list(well_data['winj'].values()), dtype=torch.float32)

        # Dynamic data
        rs = torch.load(raw_path / 'res_dynamic' / 'rs.pth', weights_only=True)
        pres = torch.load(raw_path / 'res_dynamic' / 'pres.pth', weights_only=True)
        so = torch.load(raw_path / 'res_dynamic' / 'so.pth', weights_only=True)
        sg = torch.load(raw_path / 'res_dynamic' / 'sg.pth', weights_only=True)

        props = ['np', 'gp', 'wp', 'gi', 'wi', 'p_prod', 'p_ginj', 'p_winj']
        con = {p: torch.load(raw_path / 'con_dynamic' / f'{p}.pth', weights_only=True)
               for p in props if (raw_path / 'con_dynamic' / f'{p}.pth').is_file()}
        well = {p: torch.load(raw_path / 'well_dynamic' / f'{p}.pth', weights_only=True)
                for p in props if (raw_path / 'well_dynamic' / f'{p}.pth').is_file()}

        n_conp = con['p_prod'].shape[1] if 'p_prod' in con else 0
        n_cong = con['p_ginj'].shape[1] if 'p_ginj' in con else 0
        n_conw = con['p_winj'].shape[1] if 'p_winj' in con else 0

        n_wellp = well['p_prod'].shape[1] if 'p_prod' in well else 0
        n_wellg = well['p_ginj'].shape[1] if 'p_ginj' in well else 0
        n_wellw = well['p_winj'].shape[1] if 'p_winj' in well else 0

        total_steps = 0
        out_of_range_steps = 0

        # Pre-pass: valid steps (clipped to [min_day, max_day]) and their features
        t0 = time.perf_counter()
        valid_steps_info = []
        for step in steps:
            i1, i2 = step
            day_ini = days[i1]
            day_end = days[i2]
            if day_end <= self.min_day or day_ini >= self.max_day:
                out_of_range_steps += 1
                continue
            if day_ini < self.min_day:
                i1 += int(math.ceil((self.min_day - day_ini) / self.step_size - 1e-9))
                day_ini = days[i1]
                assert day_ini >= self.min_day, "Error adjusting day_ini."
            if day_end > self.max_day:
                i2 -= int(math.ceil((day_end - self.max_day) / self.step_size - 1e-9))
                day_end = days[i2]
                assert day_end <= self.max_day, "Error adjusting day_end."
            assert i2 > i1, "Error: i2 must be greater than i1."
            n_steps = i2 - i1
            total_steps += n_steps

            x = torch.stack([rs[i1:i2 + 1], pres[i1:i2 + 1], so[i1:i2 + 1], sg[i1:i2 + 1]],
                            dim=-1)  # [n_steps+1, n_nodes, 4]
            model.x = x
            if self.transform_ is not None:
                data = self.transform_(model)
            else:
                data = x[1:] - x[:-1]
            valid_steps_info.append(
                (i1, i2, day_ini, day_end, n_steps, x, data.reshape(-1, data.shape[-1])))
        log(f"  Read {len(steps):,} step groups in {time.perf_counter()-t0:.1f}s "
            f"({len(valid_steps_info):,} valid).")

        # Resample once on all steps of this simulation together
        if valid_steps_info:
            log(f"  Resampling {len(valid_steps_info):,} valid steps for {name}...")
            all_sim_data = torch.cat([s[6] for s in valid_steps_info], dim=0)
            global_bins = [
                len(estimate_bins(all_sim_data[:, j], min_bins=1)) - 1
                for j in range(all_sim_data.shape[1])
            ]
            global_bins = reduce_product_to_limit(global_bins, MAX_BINS)
            all_mask, all_weights = resample(
                data=all_sim_data,
                num_samples=int(all_sim_data.shape[0] * self.samples_ratio),
                bins=global_bins,
                uniform_ratio=UNIFORM_RATIO,
            )
            del all_sim_data, global_bins
            # Split masks and weights back to each step group
            offset = 0
            enriched = []
            for s in valid_steps_info:
                n_flat = s[4] * model.num_nodes  # n_steps * n_nodes
                enriched.append(
                    s + (all_mask[offset:offset + n_flat], all_weights[offset:offset + n_flat]))
                offset += n_flat
            del all_mask, all_weights
            valid_steps_info = enriched

        t0 = time.perf_counter()
        for i1, i2, day_ini, day_end, n_steps, x, _, step_mask, step_weights in valid_steps_info:
            data_list[idx] = {
                'name': name,
                'day_ini': day_ini,
                'timestep_ini': i1,
                'timestep_end': i2,
                'n_steps': n_steps,
                'step_size': self.step_size,

                'n_conp': n_conp,
                'n_cong': n_cong,
                'n_conw': n_conw,

                'n_wellp': n_wellp,
                'n_wellg': n_wellg,
                'n_wellw': n_wellw,

                'conp': _get_con_data('prod'),
                'cong': _get_con_data('ginj'),
                'conw': _get_con_data('winj'),

                'prod': list(well_data['prod'].keys()),
                'ginj': list(well_data['ginj'].keys()),
                'winj': list(well_data['winj'].keys()),
            }

            model.global_attr = torch.tensor(
                [idx, day_ini, i1, i2, n_steps, n_conp, n_cong, n_conw, n_wellp, n_wellg, n_wellw],
                dtype=torch.float32)

            model.x = x

            model_mask = step_mask.reshape(-1, model.num_nodes)  # [n_steps, n_nodes]
            # Turn on cells with well connections
            model_mask[:, model.conp_index] = True
            model_mask[:, model.cong_index] = True
            model_mask[:, model.conw_index] = True
            model.train_mask = model_mask * model.mask.T

            model_weights = step_weights.reshape(-1, model.num_nodes)  # [n_steps, n_nodes]
            model.weights = model_weights * model.mask.T

            # Well data
            n1 = n_steps + 1
            if all(p in well for p in ('np', 'wp', 'gp', 'p_prod')):
                model.wellp = torch.stack([well['np'][i1:i2 + 1], well['gp'][i1:i2 + 1],
                                           well['wp'][i1:i2 + 1], well['p_prod'][i1:i2 + 1]],
                                          dim=-1)  # [n_steps+1, n_well_p, 4]
            else:
                model.wellp = torch.empty((n1, 0, 4), dtype=torch.float32)

            if 'gi' in well and 'p_ginj' in well:
                model.wellg = torch.stack([well['gi'][i1:i2 + 1], well['p_ginj'][i1:i2 + 1]], dim=-1)
            else:
                model.wellg = torch.empty((n1, 0, 2), dtype=torch.float32)

            if 'wi' in well and 'p_winj' in well:
                model.wellw = torch.stack([well['wi'][i1:i2 + 1], well['p_winj'][i1:i2 + 1]], dim=-1)
            else:
                model.wellw = torch.empty((n1, 0, 2), dtype=torch.float32)

            # Well connections
            if all(p in con for p in ('np', 'wp', 'gp', 'p_prod')):
                model.conp = torch.stack([con['np'][i1:i2 + 1], con['gp'][i1:i2 + 1],
                                          con['wp'][i1:i2 + 1], con['p_prod'][i1:i2 + 1]],
                                         dim=-1)  # [n_steps+1, n_con, 4]
            else:
                model.conp = torch.empty((n1, 0, 4), dtype=torch.float32)

            if 'gi' in con and 'p_ginj' in con:
                model.cong = torch.stack([con['gi'][i1:i2 + 1], con['p_ginj'][i1:i2 + 1]], dim=-1)
            else:
                model.cong = torch.empty((n1, 0, 2), dtype=torch.float32)

            if 'wi' in con and 'p_winj' in con:
                model.conw = torch.stack([con['wi'][i1:i2 + 1], con['p_winj'][i1:i2 + 1]], dim=-1)
            else:
                model.conw = torch.empty((n1, 0, 2), dtype=torch.float32)

            torch.save(copy.copy(model), Path(self.processed_dir) / f'data_{idx}.pt')
            idx += 1
        log(f"  Processed {len(valid_steps_info):,} step groups in {time.perf_counter()-t0:.1f}s. "
            f"Total steps processed: {total_steps:,}.")
        if out_of_range_steps > 0:
            log(f"  Total out of range steps skipped: {out_of_range_steps:,}.")

        return idx

    # MARK: Access
    def len(self):
        return len(self.processed_file_names)

    def get_file_list(self):
        """List of the processed file names inside the day window."""
        return self.processed_file_names

    def get(self, idx):
        return self.get_file(Path(self.processed_dir) / self.processed_file_names[idx])

    def get_by_raw_file_number(self, idx):
        """Data of a processed file by its file number (whole file, no day window)."""
        return self.get_file(Path(self.processed_dir) / f"data_{idx}.pt", day_filter=False)

    def get_file(self, file_path, day_filter=True):
        """Data of a processed file, cut to the day window [min_day, max_day]."""
        if not file_path.is_file():
            raise FileNotFoundError(f"File {file_path} not found.")
        model = torch.load(file_path, weights_only=False)
        if not day_filter:
            return model
        day_ini = model.global_attr[1].item()
        i1 = int(model.global_attr[2].item())
        i2 = int(model.global_attr[3].item())
        day_end = day_ini + (i2 - i1) * self.step_size
        if self.min_day > day_ini or self.max_day < day_end:
            ii1 = int(max(0, self.min_day - day_ini) / self.step_size)
            n_steps = min(self.max_day, day_end) - max(self.min_day, day_ini)
            n_steps = int(n_steps / self.step_size)
            if n_steps <= 0:
                return None
            ii2 = ii1 + n_steps
            global_attr = torch.tensor([
                model.global_attr[0].item(),
                day_ini + ii1 * self.step_size,
                i1 + ii1,
                i1 + ii1 + n_steps,
                n_steps
            ], dtype=torch.float32)
            model.global_attr = torch.concat([global_attr, model.global_attr[5:]])

            model.x = model.x[ii1: ii2 + 1]
            model.weights = model.weights[ii1: ii2]
            model.train_mask = model.train_mask[ii1: ii2]

            model.conp = model.conp[ii1: ii2 + 1]
            model.cong = model.cong[ii1: ii2 + 1]
            model.conw = model.conw[ii1: ii2 + 1]

            model.wellp = model.wellp[ii1: ii2 + 1]
            model.wellg = model.wellg[ii1: ii2 + 1]
            model.wellw = model.wellw[ii1: ii2 + 1]
        return model

    def get_sim_names(self, unique=False):
        """Simulation (SR3 file) name of each sample, or the unique names."""
        data = self.get_all_details(as_dataframe=True)
        if unique:
            return data['name'].unique().tolist()
        return data['name'].tolist()

    def get_days(self, sim_name=None):
        """Sorted list of the days with a sample (the start day of each step) in the window,
        for one simulation, a list of them, or all."""
        data = self.get_all_details(as_dataframe=True)
        if sim_name is not None:
            if isinstance(sim_name, list):
                data = data[data['name'].isin(sim_name)].copy()
            else:
                data = data[data['name'] == sim_name].copy()
        data['min_day'] = self.min_day
        data['max_day'] = self.max_day - self.step_size
        day_ini = data[['day_ini', 'min_day']].max(axis=1)
        n_steps = data['n_steps']
        data['day_end'] = data['day_ini'] + (n_steps - 1) * self.step_size
        day_end = data[['day_end', 'max_day']].min(axis=1)

        def _build_days(start, end):
            step = float(self.step_size)
            values = np.arange(float(start), float(end) + 0.5 * step, step, dtype=float)
            if values.size == 0:
                return []
            if float(self.step_size).is_integer() and float(start).is_integer():
                return values.astype(int).tolist()
            return np.round(values, decimals=10).tolist()

        data = pd.DataFrame({
            'day': [_build_days(day_ini[i], day_end[i]) for i in day_ini.index]
        })
        data = data.explode('day').reset_index(drop=True)
        days = data['day'].unique()
        days.sort()
        return days.tolist()

    def get_n_samples(self):
        """Number of one-step samples in the window."""
        data = self.get_all_details()
        n = 0
        for d in data.values():
            day_end = d['day_ini'] + (d['n_steps'] - 1) * self.step_size
            start = max(self.min_day, d['day_ini'])
            end = min(self.max_day, day_end)
            n += int((end - start) / self.step_size)
        return n

    def get_sim_days(self, sim_name, days=None, verbose=False):
        """Data of one simulation for a list of days, as one multi-step sample.

        Args:
            sim_name: SR3 file name of the simulation.
            days: start days of the steps (all days of the simulation in the window if None).

        Returns a Data with x, y (the next state), weights, train_mask, conp/cong/conw and
        wellp/wellg/wellw at the start of each step and their ``*_y`` at the end of it.
        """
        if days is None:
            days = self.get_days(sim_name)
        if len(days) == 0:
            raise ValueError("Days list cannot be empty.")

        details = self.get_all_details()
        day_found = {d: False for d in days}
        load_dict = {}
        for item in details.values():
            if item['name'] != sim_name:
                continue
            day_ini = item['day_ini']
            day_end = day_ini + item['n_steps'] * self.step_size
            for day in days:
                if day_ini <= day < day_end:
                    if day_found[day]:
                        raise ValueError(f"Duplicate day {day} found in simulation {sim_name}.")
                    day_found[day] = True
                    idx = item['file_index']
                    load_dict.setdefault(idx, []).append(int((day - day_ini) / self.step_size))
        not_found = [day for day, found in day_found.items() if not found]
        if not_found:
            raise ValueError(f"Days {not_found} not found in simulation {sim_name}.")

        fields = ['weights', 'train_mask']
        series = ['conp', 'cong', 'conw', 'wellp', 'wellg', 'wellw']
        acc = {k: [] for k in ['x', 'y'] + fields + series + [s + '_y' for s in series]}
        model = None
        for idx, days_ in tqdm(load_dict.items(), desc=f"  Loading {sim_name}",
                               total=len(load_dict), disable=not verbose):
            model_i = self.get_by_raw_file_number(idx)
            if model is None:
                model = model_i.clone()
            if len(days_) > 0:
                days_shift = [d + 1 for d in days_]
                acc['x'].append(model_i.x[days_])
                acc['y'].append(model_i.x[days_shift])
                for k in fields:
                    acc[k].append(getattr(model_i, k)[days_])
                for k in series:
                    acc[k].append(getattr(model_i, k)[days_])
                    acc[k + '_y'].append(getattr(model_i, k)[days_shift])

        for k, v in acc.items():
            setattr(model, k, torch.cat(v, dim=0))

        first_idx = min(load_dict.keys())
        first_day_offset = load_dict[first_idx][0]
        last_idx = max(load_dict.keys())
        last_day_offset = load_dict[last_idx][-1]

        global_attr = torch.tensor([
            first_idx,
            days[0],
            model.global_attr[2] + first_day_offset,
            model_i.global_attr[2] + last_day_offset,
            len(days),
        ], dtype=torch.float32)
        model.global_attr = torch.concat([global_attr, model.global_attr[5:]])
        return model

    def get_sim_day(self, sim_name, day):
        """Single-step sample of one simulation starting at `day` (see to_single_step)."""
        sim_names = self.get_sim_names()
        if sim_name not in sim_names:
            raise ValueError(f"Simulation name {sim_name} not found in dataset.")

        details = self.get_all_details()
        for idx, item in details.items():
            if item['name'] != sim_name:
                continue
            day_ini = item['day_ini']
            day_end = day_ini + item['n_steps'] * self.step_size
            if day_ini <= day < day_end:
                model = self.get_by_raw_file_number(idx)
                return self.to_single_step(model, idx=int((day - day_ini) / self.step_size))
        raise ValueError(f"Could not find {day} days in {sim_name} data.")

    @staticmethod
    def _copy_steps(model, sel, nxt):
        """Shallow copy of `model` with the per-step fields at `sel` (cloned).

        `y` and the ``*_y`` fields (state and well data at the end of each step) are taken
        from the sample when present, otherwise from the entries at `nxt`.
        """
        out = copy.copy(model)
        out.x = model.x[sel].clone()
        out.y = (model.y[sel] if getattr(model, 'y', None) is not None else model.x[nxt]).clone()
        out.weights = model.weights[sel].clone()
        out.train_mask = model.train_mask[sel].clone()
        for k in ('conp', 'cong', 'conw', 'wellp', 'wellg', 'wellw'):
            setattr(out, k, getattr(model, k)[sel].clone())
            y_field = getattr(model, k + '_y', None)
            setattr(out, k + '_y',
                    (y_field[sel] if y_field is not None else getattr(model, k)[nxt]).clone())
        out.global_attr = model.global_attr.clone()
        return out

    def to_single_step(self, model, idx):
        """Single-step sample (step `idx`, no time dimension) of a multi-step sample.

        Uses a shallow copy: static graph data is shared by reference (nothing downstream
        modifies it in place) and every per-step field is cloned.
        """
        out = self._copy_steps(model, idx, idx + 1)
        out.global_attr[1] = out.global_attr[1] + idx * self.step_size
        out.global_attr[2] = out.global_attr[2] + idx
        out.global_attr[3] = out.global_attr[3] + 1
        out.global_attr[4] = 1
        return out

    def slice_data(self, model, idx1, idx2):
        """Steps idx1 .. idx2 - 1 of a multi-step sample (shallow copy, per-step fields cloned)."""
        out = self._copy_steps(model, slice(idx1, idx2), slice(idx1 + 1, idx2 + 1))
        out.global_attr[1] = model.global_attr[1] + idx1 * self.step_size
        out.global_attr[2] = model.global_attr[2] + idx1
        out.global_attr[3] = out.global_attr[2] + idx2 - idx1
        out.global_attr[4] = idx2 - idx1
        return out

    # MARK: Details
    def reload_details(self):
        """Reload data.json (the details of every processed file)."""
        path = Path(self.processed_dir).parent / 'data.json'
        if not path.is_file():
            raise FileNotFoundError(f"File data.json not found in {self.processed_dir}.")
        with open(path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        return self.data

    def get_details(self, idx=None):
        """Details (data.json entry plus ``file_index``) of dataset item `idx`."""
        if idx is None:
            return self.get_all_details(as_dataframe=False)
        if idx < 0 or idx >= len(self.processed_file_names):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of length {len(self.processed_file_names)}.")
        f_number = int(self.processed_file_names[idx].split('_')[1].split('.')[0])

        if str(f_number) not in self.data:
            self.reload_details()
        if str(f_number) in self.data:
            out = self.data[str(f_number)].copy()
            out['file_index'] = f_number
            return out
        raise KeyError(f"Index {f_number} not found in dataset details.")

    def get_all_details(self, as_dataframe=False):
        """Details of all items. Iterates `self.indices()`, so a permutation applied with
        `.shuffle()` / `.index_select()` is reflected."""
        if len(self.data) == 0:
            self.reload_details()
        all_data = {}
        for i, real_idx in enumerate(self.indices()):
            all_data[i] = self.get_details(real_idx)
        if as_dataframe:
            return pd.DataFrame.from_dict(all_data, orient='index').reset_index()
        return all_data


# MARK: Build
def build_dataset(
        root: Union[str, Path],
        samples_ratio: float = SAMPLES_RATIO,
        step_size: float = 1.0,
        transform: Optional[Callable] = None,
        resume: bool = False,
        seed: Optional[int] = None,
        min_day: float = 0,
        max_day: float = 99999,
        sim_order: Optional[Dict[str, Sequence[str]]] = None,
        log_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, SimDataset]:
    """Process the raw folders of every split present under `root` (train, val, test).

    The physics `Transform` used to choose the training samples is read from the first
    raw model folder (PVT and relative permeability tables) unless `transform` is given
    as a callable ``transform(batch)`` (see SimDataset).

    Args:
        root: dataset root with ``<split>/raw/<simulation>/`` folders.
        samples_ratio: fraction of the cell-step samples of each simulation drawn for
            training by the histogram resampling (paper: 0.70). Cells with well connections
            are always on.
        step_size: days between outputs (1 for daily decks, 10 for 10-day decks); must match
            the step the raw folders were converted with.
        resume: continue an interrupted build instead of starting over.
        seed: seed of the random draw (torch, random, numpy). The splits are processed in
            the order train, val, test with one random stream.
        min_day, max_day: process only the steps inside this window (default: all steps).
            Training windows are chosen when the dataset is read (SimDataset, load_datasets).
        sim_order: {split: [simulation names]} processing order (see SimDataset); by default
            sorted. Only needed to reproduce a previous draw exactly.
        log_path: progress log (default ``<root>/build.log``).

    Returns:
        {split: SimDataset} of the splits built.
    """
    root = Path(root)
    print(f"Building dataset from {root}...")
    log_path = Path(log_path) if log_path is not None else root / 'build.log'
    print(f"Logging to {log_path}...")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if transform is None:
        physics = Transform.from_folder(root)
        transform = lambda batch, physics=physics: volume_increment_features(batch, physics)  # noqa: E731

    built = {}
    for split in SPLITS:
        if not (root / split).is_dir():
            continue
        raw_dir = root / split / 'raw'
        if not raw_dir.is_dir() or not any(p.is_dir() for p in raw_dir.iterdir()):
            print(f"Split {split}: no raw simulations, skipped.")
            continue
        built[split] = SimDataset(
            root=root / split,
            samples_ratio=samples_ratio,
            step_size=step_size,
            min_day=min_day,
            max_day=max_day,
            force_reload=not resume,
            resume=resume,
            log_path=log_path,
            transform=transform,
            sim_order=(sim_order or {}).get(split))
        print(f"{split.capitalize()} dataset length: {len(built[split])}")

    if not built:
        raise FileNotFoundError(f"No raw simulations found under {root}.")
    return built


def load_datasets(path: Union[str, Path], load_train: bool = True, load_val: bool = True,
                  load_test: bool = False, min_day: float = 0, max_day: float = 99999,
                  verbose: bool = False) -> Dict[str, Optional[SimDataset]]:
    """Open the built splits of a dataset root. Needs no normalisation statistics.

    Returns {'train', 'val', 'test'}: a SimDataset for each requested split that exists,
    None otherwise.
    """
    path = Path(path)
    wanted = {'train': load_train, 'val': load_val, 'test': load_test}
    out = {}
    for split, want in wanted.items():
        if not want or not (path / split / 'processed').is_dir():
            out[split] = None
            continue
        ds = SimDataset(root=path / split, min_day=min_day, max_day=max_day)
        if verbose:
            print(f"{split.capitalize()} dataset length: {len(ds):,}, "
                  f"total steps: {ds.get_n_samples():,}")
        out[split] = ds
    return out
