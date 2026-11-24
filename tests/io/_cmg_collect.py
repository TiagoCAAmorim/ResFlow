"""Collect everything the preprocessing reads from a CMG deck, and compare two collections.

The collector takes the reader modules as arguments, so the same code runs on the vendored
reader (tests) and on the original library (golden generation, a one-off development step).
"""
from pathlib import Path

import numpy as np
import pandas as pd

# keep_indices variants: all grid outputs, and every 3rd one (exercises day selection)
DAY_SELECTIONS = {'all': None, 'every3': slice(None, None, 3)}

DAT_IGNORE_PVT = ['ROCKFLUID', 'INITIAL', 'NUMERICAL', 'RUN', 'GRID_keys', 'KREL_keys', 'VFP_keys']
DAT_IGNORE_WELL = ['TITLE1', 'GRID', 'ROCKFLUID', 'INITIAL', 'NUMERICAL',
                   'GRID_keys', 'KREL_keys', 'VFP_keys', 'FLUID_keys']

STATIC_GRID_PROPS = ['BLOCKDEPTH', 'BLOCKPVOL', 'POR', 'PERMI', 'PERMJ', 'PERMK']
CON_PROPS = ['OILVOLSC', 'WATVOLSC', 'GASVOLSC', 'WELLPRES']
WELL_PROPS = ['OILVOLSC', 'WATVOLSC', 'GASVOLSC', 'BHP']


def to_plain(obj):
    """Convert readers' outputs (incl. xarray datasets) into dicts, lists, arrays and scalars."""
    if isinstance(obj, (str, bool, int, float, type(None))):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if hasattr(obj, 'data_vars') and hasattr(obj, 'coords'):  # xarray.Dataset
        return {
            '__xarray__': True,
            # the sr3 path depends on where the repo lives: keep only the file name
            'attrs': {k: (Path(v).name if k == 'file' else to_plain(v)) for k, v in obj.attrs.items()},
            'coords': {k: (tuple(v.dims), np.asarray(v.values)) for k, v in obj.coords.items()},
            'vars': {k: (tuple(v.dims), np.asarray(v.values), to_plain(dict(v.attrs)))
                     for k, v in obj.data_vars.items()},
        }
    if hasattr(obj, 'keys') and hasattr(obj, '__getitem__'):
        return {k: to_plain(obj[k]) for k in obj.keys()}
    return repr(obj)


def deep_diff(a, b, path=''):
    """Return the list of differences between two plain structures (empty when identical)."""
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return [f'{path}: type {type(a).__name__} != {type(b).__name__}']
    if isinstance(a, dict):
        out = []
        if list(a.keys()) != list(b.keys()):
            return [f'{path}: keys {list(a.keys())[:6]} != {list(b.keys())[:6]}']
        for k in a:
            out += deep_diff(a[k], b[k], f'{path}/{k}')
        return out
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} != {len(b)}']
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += deep_diff(x, y, f'{path}[{i}]')
        return out
    if isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape:
            return [f'{path}: {a.dtype}{a.shape} != {b.dtype}{b.shape}']
        same = np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc')
        return [] if same else [f'{path}: array values differ']
    if isinstance(a, pd.DataFrame):
        same = a.equals(b) and list(a.columns) == list(b.columns) and list(a.dtypes) == list(b.dtypes)
        return [] if same else [f'{path}: dataframe differs']
    if isinstance(a, float) and np.isnan(a) and np.isnan(b):
        return []
    return [] if a == b else [f'{path}: {a!r} != {b!r}']


def collect(deck, Sr3Reader, wi, dat_parser, dat_run, dat_pvt):
    """Run every reader call used by the preprocessing on `deck` (path without suffix)."""
    deck = Path(deck)
    res = {}

    # --- sr3 ---
    sr3 = Sr3Reader(deck.with_suffix('.sr3'))
    res['filepath_name'] = Path(sr3.file.get_filepath()).name
    res['days_grid'] = np.asarray(sr3.dates.get_days('grid'))
    res['days_well'] = np.asarray(sr3.dates.get_days('well'))
    res['n_active'] = sr3.grid.get_size('n_active')
    n_active = res['n_active']
    res['ijk'] = sr3.grid.n2ijk(np.arange(1, n_active + 1), as_active=True)
    res['grid_props'] = list(sr3.properties.get('grid').keys())
    res['components'] = list(sr3.properties.get_components_list())
    grid_elements = list(sr3.elements.get('grid').keys())
    wells = list(sr3.elements.get('well').keys())
    res['grid_elements_n'] = len(grid_elements)
    res['grid_elements_head'] = grid_elements[:5]
    res['wells'] = wells
    children = {w: list(sr3.elements.get_children('layer', w)) for w in wells}
    res['well_layers'] = children
    res['layer_cell_act'] = {k: int(v) for k, v in sr3.elements.get_layer_data('cell_act').items()}

    n_krel = sr3.krel.get_number_of_tables()
    res['n_krel'] = n_krel
    res['krel'] = [to_plain(sr3.krel.get(i + 1)) for i in range(n_krel)]

    res['connections'] = np.asarray(sr3.connections.get_connections(as_active=True))
    res['transmissibilities'] = np.asarray(sr3.connections.get_transmissibilities(geom_mod=True))

    props = sr3.properties.get('grid').keys()
    res['static_grid'] = to_plain(
        sr3.data.get(element_type='grid', properties=STATIC_GRID_PROPS, days=0))
    res['blockdepth_active'] = to_plain(
        sr3.data.get('grid', 'BLOCKDEPTH', days=0, active_only=True))
    if 'KRSETN' in props:
        res['krsetn'] = to_plain(sr3.data.get(element_type='grid', properties=['KRSETN'], days=0))

    dyn = ['PRES', 'SO', 'RS'] + (['SG'] if 'SG' in props else ['SW'] if 'SW' in props else [])
    res['dyn_grid_all_days'] = to_plain(sr3.data.get(element_type='grid', properties=dyn))

    last_day = sr3.dates.get_days('well')[-1]
    res['well_cum_last_day'] = to_plain(sr3.data.get(
        element_type='well', properties=['NP', 'GP', 'WP'], elements=wells, days=last_day))

    all_days = sr3.dates.get_days('grid')
    all_cons = [c for w in wells for c in children[w]]
    for name, sel in DAY_SELECTIONS.items():
        days = list(all_days) if sel is None else list(all_days)[sel]
        res[f'well_dyn_{name}'] = {
            p: to_plain(sr3.data.get(
                element_type='well', properties=[p], elements=wells, days=days))
            for p in WELL_PROPS}
        res[f'con_dyn_{name}'] = {
            p: to_plain(sr3.data.get(
                element_type='layer', properties=[p], elements=all_cons, days=days))
            for p in CON_PROPS}
    sr3.close()

    # --- dat ---
    dat = deck.with_suffix('.dat')
    parser = dat_parser.DatParser(ignore=DAT_IGNORE_PVT)
    parser.process(dat)
    res['dat_pvt_parsed'] = to_plain(parser.get())
    res['pvt'] = to_plain(dat_pvt.get_from_dat_data(parser.get()))

    parser = dat_parser.DatParser(ignore=DAT_IGNORE_WELL)
    parser.process(dat)
    res['dat_well_parsed'] = to_plain(parser.get())
    res['bhp_depth'] = to_plain(dat_run.get_well_key(parser.get(), keyword='BHPDEPTH', verbose=False))

    # --- out ---
    out_wi = wi.OutWI(verbose=False, wi_only=True, encoding='utf-8')
    out_wi.process(deck.with_suffix('.out'), prune=True)
    res['wi_table'] = out_wi.get_table()
    return res
