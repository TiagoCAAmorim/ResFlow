"""General model data from an IMEX SR3 file: calendar, wells, PVT and relative permeability.

Copyright (c) Tiago Amorim. MIT License.
"""
import json
from pathlib import Path
from typing import Optional, Union, Any

import numpy as np
import pandas as pd
import torch

from ..io.cmg.sr3reader.sr3reader import Sr3Reader
from ..io.cmg.datreader import dat_parser, dat_pvt

# The wells of the decks open at day 4 (SHUTIN until the first control). A run whose last
# output is at or before this day never opened its wells and is discarded.
WELLS_OPEN_DAY = 4.0


class RunDiscarded(ValueError):
    """A run that cannot be used (never opened its wells, no usable outputs, ...)."""


def build_model_data(
        sr3: Sr3Reader,
        output_dir: Union[str, Path],
        step_size: float = 1.0,
        verbose: Optional[bool] = False,
        wells_open_day: float = WELLS_OPEN_DAY) -> None:
    """Read and save model information (model.json, pvtNN.pth, krelNN.{wo,go}.pth).

    Raises RunDiscarded when the run is unusable: it ended at or before `wells_open_day`
    (wells never opened), no well produced or injected, or no output series exists for
    `step_size`. Otherwise the run is kept up to the last output it reached.
    """
    output_dir = Path(output_dir)

    data = _get_general_data(sr3, step_size, verbose, wells_open_day)

    prod, winj, ginj = _check_wells(sr3, verbose)
    data['prod'] = prod
    data['winj'] = winj
    data['ginj'] = ginj
    data['n_krel'] = sr3.krel.get_number_of_tables()

    pvt = _read_pvt_from_file(sr3, verbose)
    data['n_pvt'] = len(pvt)

    _save_data(output_dir, verbose, data)

    for i, pvt_data in enumerate(pvt):
        _save_pvt(output_dir / f"pvt{i+1:02}.pth", pvt_data, verbose)

    for i in range(data['n_krel']):
        kr_data = sr3.krel.get(i+1)
        _save_krel(output_dir / f"krel{i+1:02}", kr_data, verbose)


def _save_krel(file_stem: Union[str, Path], data: Any, verbose: bool = False) -> None:
    """Save the relative permeability tables of one KREL set.

    Two files are written, ``<file_stem>.wo.pth`` (Sw, Krw, Krow) and ``<file_stem>.go.pth``
    (Sg, Krg, Krog), as float32 tensors of shape (n, 3) sorted by saturation. The (0, 0, max)
    and (1, 1, 0) points are appended if not present.
    """
    file_stem = Path(file_stem)
    file_stem.parent.mkdir(parents=True, exist_ok=True)

    sw = data['sw'].values
    krw = data['krw'].values
    krow = data['krow'].values
    sg = 1 - data['sl'].values
    krg = data['krg'].values
    krog = data['krog'].values

    wo_data = torch.tensor(np.stack([sw, krw, krow]).T, dtype=torch.float32)
    go_data = torch.tensor(np.stack([sg, krg, krog]).T, dtype=torch.float32)

    wo_data = wo_data[torch.argsort(wo_data[:, 0])]
    if wo_data[:, 0].min() > 0:
        wo_data = torch.cat([torch.tensor([[0, 0, wo_data[:, -1].max()]]), wo_data])
    if wo_data[:, 0].max() < 1:
        wo_data = torch.cat([wo_data, torch.tensor([[1, 1, 0]])])

    go_data = go_data[torch.argsort(go_data[:, 0])]
    if go_data[:, 0].min() > 0:
        go_data = torch.cat([torch.tensor([[0, 0, go_data[:, -1].max()]]), go_data])
    if go_data[:, 0].max() < 1:
        go_data = torch.cat([go_data, torch.tensor([[1, 1, 0]])])

    wo_path = Path(str(file_stem) + '.wo.pth')
    go_path = Path(str(file_stem) + '.go.pth')
    torch.save(wo_data, wo_path)
    torch.save(go_data, go_path)

    if verbose:
        print(f"W-O Krel saved to {wo_path.name}")
        print(f"G-O Krel saved to {go_path.name}")


def _save_pvt(file_path: Union[str, Path], data: dict, verbose: bool = False) -> None:
    """Save a PVT dictionary as a dictionary of tensors."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict_to_tensors(data), file_path)

    if verbose:
        print(f"PVT saved to {file_path.name}")


def dict_to_tensors(data):
    """Convert a dictionary to tensors, ignoring items that are not arrays, frames or numbers.

    float64 arrays become float32.
    """
    tensor_dict = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            arr = v.values
        elif isinstance(v, np.ndarray):
            arr = v
        elif isinstance(v, (float, int)):
            arr = np.array(v)
        else:
            continue  # skip non-array-like items
        tensor_dict[k] = torch.from_numpy(arr)
        if tensor_dict[k].dtype == torch.float64:
            tensor_dict[k] = tensor_dict[k].float()
    return tensor_dict


def _read_pvt_from_file(sr3, verbose):
    file = sr3.file.get_filepath().with_suffix('.dat')
    if not file.exists():
        raise FileNotFoundError(f"DAT file {file.name} does not exist.")

    if verbose:
        print(f'Reading DAT file: {file.name}')

    parser = dat_parser.DatParser(
        ignore=['ROCKFLUID', 'INITIAL', 'NUMERICAL', 'RUN',
                'GRID_keys', 'KREL_keys', 'VFP_keys'])
    parser.process(file)

    if 'TITLE1' not in parser.get():
        raise ValueError(f"TITLE1 section not found in {file.name}.")
    if 'GRID' not in parser.get():
        raise ValueError(f"GRID section not found in {file.name}.")

    return dat_pvt.get_from_dat_data(parser.get())


def _get_general_data(sr3, step_size, verbose, wells_open_day):
    file_path = Path(sr3.file.get_filepath())
    is_2phi2k = "FRACTURE" in sr3.elements.get("grid").keys()
    grid_type = '2phi2k' if is_2phi2k else '1phi1k'

    n_pseudos = len(sr3.properties.get_components_list())
    model_type = 'compositional' if n_pseudos > 0 else 'blackoil'

    n_cells = sr3.grid.get_size("n_active")

    days = sr3.dates.get_days('grid')
    if len(days) == 0:
        raise RunDiscarded(f"No time steps found in {file_path.name}.")
    if len(days) == 1:
        raise RunDiscarded(f"Only one time step found in {file_path.name}.")
    days = [float(d) for d in days]
    kept, steps = select_outputs(days, step=step_size)

    if len(kept) == 0:
        gaps = np.diff(days)
        raise RunDiscarded(
            f"No outputs spaced exactly step_size={step_size:g} days apart in {file_path.name} "
            f"(grid outputs: {len(days)}, smallest gap {gaps.min():g} d, largest {gaps.max():g} d). "
            "Use a step size equal to the interval between the grid outputs of the decks "
            "(1 for the daily decks, 10 for the 10-day decks).")

    last_day = days[kept[-1]]
    if last_day <= wells_open_day:
        raise RunDiscarded(
            f"Run ended at day {last_day:g} (<= {wells_open_day:g}): the wells never opened. "
            "Discarded.")

    if verbose:
        print(f'\nModel name: {file_path.name}')
        print(f'  Grid type: {grid_type}')
        print(f'  Model type: {model_type}')
        print(f'  Number of active cells: {n_cells}')
        print(f'  Number of grid time-steps: {len(days)} ({len(kept)} kept, last day {last_day:g})')

    return {
        'name': file_path.name,
        'grid_type': grid_type,
        'model_type': model_type,
        'n_cells': n_cells,
        'days': [days[i] for i in kept],
        'steps': steps,
    }


def select_outputs(values, step=1.0, tol=1e-6):
    """Select the grid outputs that form series spaced exactly `step` apart.

    A series starts at an output and moves to the next one at least `step` later. Outputs
    closer than `step` to the current one (intermediate outputs) are skipped until the
    cumulative time reaches `step`: if it lands on `step` (within `tol`) that output is the
    next in the series, if it overshoots the series ends at the current output. The next
    series starts at the output right after the one where the previous series ended.
    Series with a single output are discarded.

    Parameters
    ----------
    values : list of float
        Output days in ascending order.
    step : float
        Expected difference between consecutive values in the same series.
    tol : float
        Tolerance for comparing differences.

    Returns
    -------
    kept : list of int
        Indices in `values` of the outputs that belong to a series (skipped intermediate
        outputs are not included).
    groups : list of list of int
        Series as [start_index, end_index], indexing the `kept` list (not `values`).

    Examples
    --------
    >>> select_outputs([1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 10.0])
    ([0, 1, 2, 3, 4, 5], [[0, 2], [3, 5]])

    >>> select_outputs([0.0, 0.5, 1.0, 1.5, 2.0])
    ([0, 2, 4], [[0, 2]])

    >>> select_outputs([0.0, 1.0, 1.4, 2.4, 3.4])
    ([0, 1, 2, 3, 4], [[0, 1], [2, 4]])
    """
    n = len(values)
    kept = []
    groups = []
    start = 0
    while start < n - 1:
        members = [start]
        cur = start
        j = start + 1
        while j < n:
            diff = values[j] - values[cur]
            if diff < step - tol:
                j += 1
            elif diff <= step + tol:
                members.append(j)
                cur = j
                j += 1
            else:
                break
        if len(members) > 1:
            groups.append([len(kept), len(kept) + len(members) - 1])
            kept.extend(members)
        start = cur + 1
    return kept, groups


def get_output_selection(sr3, step_size=1.0):
    """Return (kept grid-output indices, series over them) for an SR3 file; see select_outputs."""
    days = [float(d) for d in sr3.dates.get_days('grid')]
    return select_outputs(days, step=step_size)


def _save_data(output_dir, verbose, data):
    output_file = output_dir / "model.json"
    with output_file.open("w") as f:
        json.dump(data, f, indent=4)

    if verbose:
        print(f"Data saved to {output_file.name}")


def get_sg(grid_data, missing_sw_safe=False):
    """Add the gas saturation (clip(1 - so - sw, 0, 1)) to grid data holding SO and SW."""
    so = grid_data['SO'].values
    if 'SW' not in grid_data.keys():
        if not missing_sw_safe:
            raise ValueError("SW not found in grid data.")
        sw = np.zeros_like(so)
    else:
        sw = grid_data['SW'].values
    sg = np.clip(1 - so - sw, 0, 1)
    grid_data['SG'] = (('index', 'day'), sg)


def _check_wells(sr3, verbose):
    """Classify the wells of the SR3 file from their cumulative volumes at the last output.

    A well is shut-in (ignored) when its cumulative oil, gas and water production are all
    < 1 m3; a water (gas) injector has WP (GP) > 1 and NP < 1; a producer has NP > 1.
    Raises RunDiscarded when no well is left (no well ever opened).
    """
    wells = list(sr3.elements.get("well").keys())
    if len(wells) == 0:
        raise RunDiscarded("No wells found.")

    day = sr3.dates.get_days('well')[-1]

    well_data = sr3.data.get(
        element_type="well",
        properties=["NP", "GP", "WP"],
        elements=wells,
        days=day
    )

    prod = {}
    winj = {}
    ginj = {}
    for well in wells:
        prod[well] = well_data["NP"].sel(element=well).values[0]
        winj[well] = well_data["WP"].sel(element=well).values[0]
        ginj[well] = well_data["GP"].sel(element=well).values[0]

    shut_in = [well for well in wells if prod[well] < 1 and winj[well] < 1 and ginj[well] < 1]
    if len(shut_in) > 0 and verbose:
        print(f"  => Shut-in wells: {len(shut_in)}")

    winj = [well for well in wells if prod[well] < 1 < winj[well]]
    ginj = [well for well in wells if prod[well] < 1 < ginj[well]]
    prod = [well for well in wells if prod[well] > 1]

    for well in prod:
        if well in winj:
            raise ValueError(f"Well {well} is both a producer and water injector.")
        if well in ginj:
            raise ValueError(f"Well {well} is both a producer and gas injector.")
    for well in winj:
        if well in ginj:
            raise ValueError(f"Well {well} is both a water and gas injector.")

    if len(prod) + len(winj) + len(ginj) == 0:
        raise RunDiscarded(
            f"No production or injection wells found at day {float(day):g}: "
            "the wells never opened. Discarded.")

    if verbose:
        print('  Wells:')
        print(f'    {len(prod)} producers')
        print(f'    {len(winj)} water injectors')
        print(f'    {len(ginj)} gas injectors')

    return prod, winj, ginj
