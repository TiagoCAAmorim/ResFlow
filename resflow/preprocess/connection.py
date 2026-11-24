"""Well connection data (completions) from an IMEX SR3 file and its OUT file.

Copyright (c) Tiago Amorim. MIT License.
"""
import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch

from ..io.cmg.sr3reader import Sr3Reader
from ..io.cmg.outreader import wi

from .model import _check_wells


# MARK: Static
def build_static_con_data(
        sr3: Sr3Reader,
        output_dir: Union[str, Path],
        verbose: Optional[bool] = False) -> None:
    """Read and save the static connection data to ``con_static.json``.

    Layout: {'prod'|'winj'|'ginj': {well: {connection: {'cell', 'depth', 'wi'}}}}, with the
    1-based active cell number, the cell depth (m) and the connection well index (IMEX
    units, from the OUT file). Connections to inactive cells are removed.
    """
    output_dir = Path(output_dir)
    if verbose:
        print("\nReading static connection data.")

    wells = sr3.elements.get("well").keys()

    data = {}
    _get_con_list(sr3, verbose, wells, data)
    _get_con_cell(sr3, data)
    _prune_inactive_cons(sr3, verbose, data)
    _get_con_wi(sr3, verbose, data)

    for well, cons in data.items():
        for con in cons:
            for k in ['cell', 'depth', 'wi']:
                if k not in cons[con]:
                    raise ValueError(f"Key {k} not found in connection {con} for well {well}.")

    prod, winj, ginj = _check_wells(sr3, verbose)
    data_ = {'prod': {}, 'winj': {}, 'ginj': {}}
    for t, w_lst in {'prod': prod, 'winj': winj, 'ginj': ginj}.items():
        for well in w_lst:
            if well not in data:
                raise ValueError(f"Well {well} not found in SR3 file.")
            data_[t][well] = data[well]

    output_file = output_dir / "con_static.json"
    with output_file.open("w") as f:
        json.dump(data_, f, indent=4)
    if verbose:
        print(f"Data saved to {output_file.name}")


def _get_con_wi(sr3, verbose, data):
    if verbose:
        print("Reading out file.")
    out_wi = wi.OutWI(verbose=False, wi_only=True, encoding='utf-8')
    file_path = sr3.file.get_filepath().with_suffix('.out')
    out_wi.process(file_path, prune=True)
    table = out_wi.get_table()

    if len(table) == 0:
        raise ValueError(f"OUT file {file_path.name} does not contain any well connection data.")

    for well, cons in data.items():
        for con in cons:
            cell = con.split('{')[1][:-1]
            con_factor = table.loc[(table['well'] == well) & (table['cell'] == cell), 'wi'].values
            if len(con_factor) == 0:
                raise ValueError(f"Connection {con} not found in OUT file.")
            if len(con_factor) > 1 and verbose:
                print(f"  Connection {con} found multiple times in OUT file.")
            cons[con]['wi'] = float(con_factor[0])


def _prune_inactive_cons(sr3, verbose, data):
    bad_cons = {}
    # (n_active, 1) at day 0; ravel so that each depth is a plain number (numpy >= 2)
    depth = np.ravel(
        sr3.data.get('grid', 'BLOCKDEPTH', days=0, active_only=True)['BLOCKDEPTH'].values)
    for well, cons in data.items():
        bad_cons[well] = []
        for con in cons:
            cell_index = cons[con]['cell']
            if cell_index == 0:
                bad_cons[well].append(con)
            else:
                cons[con]['depth'] = float(depth[cell_index - 1])
        if len(bad_cons[well]) == 0:
            del bad_cons[well]

    i = 0
    for well, cons in bad_cons.items():
        for con in cons:
            del data[well][con]
            i += 1
        if len(data[well]) == 0:
            del data[well]
    if verbose:
        print(f"  Removed {i:,} connections to inactive cells.")


def _get_con_cell(sr3, data):
    cell_data = sr3.elements.get_layer_data('cell_act')
    for cons in data.values():
        for con in cons:
            cons[con]['cell'] = int(cell_data[con])


def _get_con_list(sr3, verbose, wells, data):
    for well in wells:
        data[well] = {}
        connections = sr3.elements.get_children('layer', well)
        if verbose:
            print(f"  Well {well} has {len(connections)} connections.")
        for con in connections:
            data[well][con] = {}


# MARK: Dynamic
def build_dynamic_con_data(
        sr3: Sr3Reader,
        output_dir: Union[str, Path],
        verbose: Optional[bool] = False,
        keep_indices: Optional[Sequence[int]] = None) -> None:
    """Read and save the dynamic connection data to ``con_dynamic/*.pth``.

    Per output step and connection (columns in the order of the wells of each type in
    con_static.json): step volumes at standard conditions (m3) ``np``, ``wp``, ``gp``
    (producers: oil, water, gas), ``wi`` (water injectors), ``gi`` (gas injectors), and
    the connection pressure ``p_prod``, ``p_winj``, ``p_ginj``. If `keep_indices` is
    given, only those grid outputs (indices into the SR3 grid days) are read.
    """
    output_dir = Path(output_dir)
    if verbose:
        print("\nReading dynamic connection data.")

    wells = sr3.elements.get("well").keys()

    data = {}
    _get_con_list(sr3, False, wells, data)

    days = sr3.dates.get_days('grid')
    if keep_indices is not None:
        days = [days[i] for i in keep_indices]
    prod, winj, ginj = _check_wells(sr3, verbose)
    well_lists = {'prod': prod, 'winj': winj, 'ginj': ginj}

    for w_lst in well_lists.values():
        for well in w_lst:
            if well not in wells:
                raise ValueError(f"Well {well} not found in SR3 file.")

    con_lists = {}
    for t, w_lst in well_lists.items():
        con_lists[t] = [con for well in w_lst for con in data[well].keys()]

    properties = {
        'prod': {'np': 'OILVOLSC', 'wp': 'WATVOLSC', 'gp': 'GASVOLSC', 'p_prod': 'WELLPRES'},
        'winj': {'wi': 'WATVOLSC', 'p_winj': 'WELLPRES'},
        'ginj': {'gi': 'GASVOLSC', 'p_ginj': 'WELLPRES'}
    }

    for t, con_lst in con_lists.items():
        if len(con_lst) == 0:
            if verbose:
                print(f"  No connections found for {t.upper()}.")
            continue
        for file_name, prop in properties[t].items():
            values = sr3.data.get(
                element_type='layer',
                properties=[prop],
                elements=con_lst,
                days=days
            )[prop].values

            values = np.nan_to_num(values, nan=0.0)
            output_path = output_dir / "con_dynamic" / f"{file_name}.pth"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.tensor(values, dtype=torch.float32), output_path)
            if verbose:
                print(f"  Saved {t.upper()} {prop} to {output_path.name}")
