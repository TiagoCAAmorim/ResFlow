"""Shared fixtures: convert (a subset of) the IMEX subset decks once per session.

All tests of this folder skip when `_data/imex` or `_data/golden` is absent.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[2]
IMEX = ROOT / '_data' / 'imex'
GOLDEN = ROOT / '_data' / 'golden'

SETS = {'spe10': 1.0, 'spe10_fluvial': 1.0, 'spe10_10d': 10.0, 'spe10_3d_10d': 10.0}

have_data = IMEX.is_dir() and (GOLDEN / 'raw').is_dir() and (GOLDEN / 'datasets').is_dir()
needs_data = pytest.mark.skipif(not have_data, reason='_data/imex or _data/golden absent')


def golden_order(name, split='train'):
    """Simulation names of a golden split in the order they were processed."""
    details = json.loads((GOLDEN / 'datasets' / name / split / 'data.json').read_text())
    names = []
    for k in sorted(details, key=int):
        stem = details[k]['name'][:-len('.sr3')]
        if stem not in names:
            names.append(stem)
    return names


def subset_decks(name):
    """Decks converted by the tests for each set (None: all). The spe10 subset has the
    first 3 train simulations of the golden build (their random draw is then reproduced
    exactly), a deck that fails (run ended at day 4), a val and a test deck."""
    if name != 'spe10':
        return None
    first = golden_order('spe10', 'train')[:3]
    index = pd.read_csv(IMEX / 'spe10' / 'decks_index.csv')
    val = index.loc[index['set'].isin(['validation', 'val']), 'deck'].tolist()[:1]
    test = index.loc[index['set'] == 'test', 'deck'].tolist()[:1]
    return first + ['L03_32x32_02_2P1I_variable'] + val + test


@pytest.fixture(scope='session')
def converted(tmp_path_factory):
    """{set: (raw root, list of failure messages, decks converted or None for all)}."""
    from resflow.data.convert import organize_sr3_dataset
    out = {}
    base = tmp_path_factory.mktemp('port')
    for name, step in SETS.items():
        if not have_data:
            pytest.skip('_data absent')
        index = pd.read_csv(IMEX / name / 'decks_index.csv')
        decks = subset_decks(name)
        if decks is not None:
            index = index[index['deck'].isin(decks)]
        index_path = base / f'{name}_index.csv'
        index.to_csv(index_path, index=False)
        root = base / 'raw' / name
        messages = organize_sr3_dataset(
            folder=IMEX / name / 'decks', output_dir=root, split_csv=index_path, step_size=step)
        out[name] = (root, messages, decks)
    return out
