"""Well data (datum depth and BHP/volumes) from an IMEX SR3 file and its DAT file.

Copyright (c) Tiago Amorim. MIT License.
"""
import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch

from ..io.cmg.sr3reader import Sr3Reader
from ..io.cmg.datreader import dat_parser, dat_run

from .model import _check_wells


# MARK: Static
def build_static_well_data(
        sr3: Sr3Reader,
        output_dir: Union[str, Path],
        verbose: Optional[bool] = False) -> None:
    """Read and save the static well data (BHP datum depth, m) to ``well_static.json``.

    Layout: {'prod'|'winj'|'ginj': {well: bhp_depth}}.
    """
    output_dir = Path(output_dir)
    if verbose:
        print("\nReading static well data.")

    wells = sr3.elements.get("well").keys()

    parser = dat_parser.DatParser(
        ignore=['TITLE1', 'GRID', 'ROCKFLUID', 'INITIAL', 'NUMERICAL',
                'GRID_keys', 'KREL_keys', 'VFP_keys', 'FLUID_keys'])
    file = sr3.file.get_filepath().with_suffix('.dat')
    parser.process(file)
    bhp_depth = dat_run.get_well_key(parser.get(), keyword='BHPDEPTH', verbose=False)

    _check_bhp_depth_all_wells(wells, bhp_depth)
    _check_one_bhp_depth(wells, bhp_depth)

    data = {well: depth for _, well, depth in bhp_depth if well in wells}

    data_ = {'prod': {}, 'winj': {}, 'ginj': {}}
    prod, winj, ginj = _check_wells(sr3, verbose)
    for t, w_lst in {'prod': prod, 'winj': winj, 'ginj': ginj}.items():
        for well in w_lst:
            if well not in data:
                raise ValueError(f"Well {well} not found in dat file.")
            data_[t][well] = data[well]

    output_file = output_dir / "well_static.json"
    with output_file.open("w") as f:
        json.dump(data_, f, indent=4)
    if verbose:
        print(f"Data saved to {output_file.name}")


def _check_bhp_depth_all_wells(wells, bhp_depth):
    bhp_wells = [well for _, well, _ in bhp_depth]
    wells_not_ok = [well for well in wells if well not in bhp_wells]
    if len(wells_not_ok):
        msg = "The following wells do not have BHPDEPTH in the dat file: "
        msg += ', '.join(wells_not_ok)
        raise ValueError(msg)


def _check_one_bhp_depth(wells, bhp_depth):
    wells_ok = {well: [] for well in wells}
    for _, well, depth in bhp_depth:
        if well in wells:
            wells_ok[well].append(depth)
    for well, depth in wells_ok.items():
        depth = set(depth)
        if len(depth) > 1:
            msg = f"Well {well} has multiple BHPDEPTH values: {depth}. "
            msg += "This is not supported."
            raise ValueError(msg)


# MARK: Dynamic
def build_dynamic_well_data(
        sr3: Sr3Reader,
        output_dir: Union[str, Path],
        verbose: Optional[bool] = False,
        keep_indices: Optional[Sequence[int]] = None) -> None:
    """Read and save the dynamic well data to ``well_dynamic/*.pth``.

    Per output step and well: cumulative volumes at standard conditions (m3) ``np``, ``wp``,
    ``gp`` (producers), ``wi`` (water injectors), ``gi`` (gas injectors) and the BHP
    ``p_prod``, ``p_winj``, ``p_ginj`` (kgf/cm2). If `keep_indices` is given, only those
    grid outputs (indices into the SR3 grid days) are read.
    """
    output_dir = Path(output_dir)
    if verbose:
        print("\nReading dynamic well data.")

    wells = sr3.elements.get("well").keys()
    days = sr3.dates.get_days('grid')
    if keep_indices is not None:
        days = [days[i] for i in keep_indices]
    prod, winj, ginj = _check_wells(sr3, False)
    well_lists = {'prod': prod, 'winj': winj, 'ginj': ginj}

    for w_lst in well_lists.values():
        for well in w_lst:
            if well not in wells:
                raise ValueError(f"Well {well} not found in SR3 file.")

    properties = {
        'prod': {'np': 'OILVOLSC', 'wp': 'WATVOLSC', 'gp': 'GASVOLSC', 'p_prod': 'BHP'},
        'winj': {'wi': 'WATVOLSC', 'p_winj': 'BHP'},
        'ginj': {'gi': 'GASVOLSC', 'p_ginj': 'BHP'}
    }

    for t, well_lst in well_lists.items():
        if len(well_lst) == 0:
            if verbose:
                print(f"  No {t.upper()} wells found.")
            continue
        for file_name, prop in properties[t].items():
            values = sr3.data.get(
                element_type='well',
                properties=[prop],
                elements=well_lst,
                days=days
            )[prop].values

            values = np.nan_to_num(values, nan=0.0)
            output_path = output_dir / "well_dynamic" / f"{file_name}.pth"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.tensor(values, dtype=torch.float32), output_path)
            if verbose:
                print(f"  Saved {t.upper()} {prop} to {output_path.name}")
