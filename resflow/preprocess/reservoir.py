"""Static and dynamic reservoir (grid) data from an IMEX SR3 file.

Copyright (c) Tiago Amorim. MIT License.
"""
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch_geometric as tg

from ..io.cmg.sr3reader import Sr3Reader
from .model import get_sg


# MARK: Static
def build_static_grid_data(
        sr3: Sr3Reader,
        output_dir: Union[str, Path],
        verbose: Optional[bool] = False) -> None:
    """Read and save the static reservoir graph to ``res_static.pt``.

    The file holds a torch_geometric ``Data`` with:
    - stat [n_nodes, 7]: depth, reference pore volume, porosity, perm I, J, K, KREL table;
    - edge_index [2, 2 n_edges]: both directions of every cell connection;
    - edge_tran [2 n_edges, 1]: transmissibility; edge_dist, edge_dist_mod [2 n_edges]:
      distance between the cells, plain and modified by the geometry (from the SR3 file);
    - ijk [n_nodes, 3]: grid indices; mask [n_nodes, 1]: nodes with at least one connection.
    No dense all-pairs distance matrices are stored.
    """
    output_dir = Path(output_dir)

    properties = ['BLOCKDEPTH', 'BLOCKPVOL', 'POR', 'PERMI', 'PERMJ', 'PERMK']
    if verbose:
        print("\nReading static grid data.")
    node_values = sr3.data.get(element_type='grid', properties=properties, days=0)
    depth = node_values['BLOCKDEPTH'].sel(day=0.).values
    volume = node_values['BLOCKPVOL'].sel(day=0.).values
    por = node_values['POR'].sel(day=0.).values
    permi = node_values['PERMI'].sel(day=0.).values
    permj = node_values['PERMJ'].sel(day=0.).values
    permk = node_values['PERMK'].sel(day=0.).values
    krel = get_krel(sr3)

    connections = sr3.connections.get_connections(as_active=True)[:, :2]
    trans_data = np.asarray(sr3.connections.get_transmissibilities(geom_mod=True))
    if trans_data.ndim != 2 or trans_data.shape[1] < 3:
        raise ValueError(
            "Expected get_transmissibilities(geom_mod=True) to return an array "
            f"with shape [n_cell_pairs, 3], got {trans_data.shape}."
        )
    n_cells = sr3.grid.get_size("n_active")

    data_ = build_graph(
        depth=depth, volume=volume, por=por, permi=permi, permj=permj, permk=permk,
        krel=krel,
        connections=connections,
        transmissibility=trans_data[:, 0],
        cell_distance=trans_data[:, 1],
        cell_distance_mod=trans_data[:, 2],
        ijk=sr3.grid.n2ijk(np.arange(1, n_cells + 1), as_active=True),
    )

    if verbose:
        print("Graph summary:")
        print(f"  Number of nodes: {data_.num_nodes:,}")
        print(f"  Number of edges: {data_.num_edges:,}")
        print(f"  Has isolated nodes: {data_.has_isolated_nodes()}")

    output_path = output_dir / "res_static.pt"
    torch.save(data_, output_path)
    if verbose:
        print(f"Static grid graph saved to {output_path.name}")


def get_krel(sr3: Sr3Reader) -> np.ndarray:
    """KREL table index (1-based) of each active cell, shape (n_cells, 1)."""
    if 'KRSETN' in sr3.properties.get("grid"):
        krel_ = sr3.data.get(element_type='grid', properties=['KRSETN'], days=0)
        krel = krel_['KRSETN'].sel(day=0.).values.astype(int)
    else:
        if sr3.krel.get_number_of_tables() > 1:
            raise ValueError(
                'KRSETN not found in SR3 file and '
                f'model has {sr3.krel.get_number_of_tables():,} KREL tables. '
                'Cannot continue.')
        krel = np.ones(sr3.grid.get_size("n_active"), dtype=int)
    return np.expand_dims(krel, axis=1)


def build_graph(
        *,
        depth: np.ndarray,
        volume: np.ndarray,
        por: np.ndarray,
        permi: np.ndarray,
        permj: np.ndarray,
        permk: np.ndarray,
        krel: np.ndarray,
        connections: np.ndarray,
        transmissibility: np.ndarray,
        cell_distance: np.ndarray,
        cell_distance_mod: np.ndarray,
        ijk: np.ndarray) -> tg.data.Data:
    """Build the (undirected) pyg graph from node and edge values.

    `connections` is (n_edges, 2) with 1-based active cell numbers; each connection is
    stored in both directions. Raises ValueError for self loops or a directed graph.
    """
    stat = np.stack([depth, volume, por, permi, permj, permk, krel[:, 0]], axis=1)
    num_nodes = stat.shape[0]

    connections = np.vstack([connections, connections[:, [1, 0]]])
    edge_index = (connections - 1).T
    edge_tran = np.concat([transmissibility, transmissibility]).reshape(-1, 1)
    edge_dist = np.concat([cell_distance, cell_distance])
    edge_dist_mod = np.concat([cell_distance_mod, cell_distance_mod])

    graph_ = tg.data.Data(
        stat=torch.tensor(stat).to(torch.float32),
        edge_index=torch.tensor(edge_index).to(torch.int64),
        edge_tran=torch.tensor(edge_tran).to(torch.float32),
        edge_dist=torch.tensor(edge_dist).to(torch.float32),
        edge_dist_mod=torch.tensor(edge_dist_mod).to(torch.float32),
        ijk=torch.tensor(ijk, dtype=torch.int64),
        num_nodes=num_nodes,
        num_edges=edge_index.shape[1],
    )

    # Connected nodes are those that appear in at least one edge endpoint.
    connected = np.unique(edge_index.reshape(-1))
    mask = np.zeros(num_nodes, dtype=bool)
    mask[connected] = True
    graph_.mask = torch.tensor(mask).to(torch.bool).unsqueeze(-1)

    graph_.validate(raise_on_error=True)
    if graph_.has_self_loops():
        raise ValueError("Graph has self loops. Please check the data.")
    if graph_.is_directed():
        raise ValueError("Graph is directed. Please check the data.")
    return graph_


# MARK: Dynamic
def build_dynamic_grid_data(
        sr3: Sr3Reader,
        output_dir: Union[str, Path],
        verbose: Optional[bool] = False,
        keep_indices: Optional[Sequence[int]] = None) -> None:
    """Read and save the dynamic reservoir state to ``res_dynamic/{pres,so,rs,sg}.pth``.

    Each file is a float32 tensor [n_time_steps, n_active_cells]. If `keep_indices` is
    given, only those grid outputs (indices into the SR3 grid days) are saved.
    """
    output_dir = Path(output_dir)

    properties = ['PRES', 'SO', 'RS']
    for p in properties:
        if p not in sr3.properties.get("grid"):
            raise ValueError(f'Property {p} not found in SR3 file.')

    if 'SG' in sr3.properties.get("grid"):
        properties.append('SG')
    elif 'SW' in sr3.properties.get("grid"):
        properties.append('SW')
    else:
        print("Warning: Neither SG nor SW found in SR3 file. Assume SG=0.")

    if verbose:
        print("\nReading dynamic grid data.")
    values = sr3.data.get(element_type='grid', properties=properties)

    if 'SG' not in values:
        get_sg(values, missing_sw_safe=True)
        properties.append('SG')
        if 'SW' in properties:
            properties.remove('SW')

    n_days = len(sr3.dates.get_days('grid'))
    n_cells = sr3.grid.get_size("n_active")
    for p in properties:
        v = values[p].values
        v[v < 1e-15] = 0
        if verbose:
            print(f"  Property {p}: {v.shape}")
        if v.shape[0] != n_cells:
            print(f"Warning: Number of cells in {p} ({v.shape[0]:,}) "
                  f"does not match number of active cells ({n_cells:,}).")
        if v.shape[1] != n_days:
            print(f"Warning: Number of time steps in {p} ({v.shape[1]:,}) "
                  f"does not match number of grid time steps ({n_days:,}).")

        output_path = output_dir / "res_dynamic" / f"{p.lower()}.pth"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if keep_indices is not None:
            v = v[:, list(keep_indices)]
        torch.save(torch.tensor(v, dtype=torch.float32).T, output_path)
        if verbose:
            print(f"  Saved {p} to {output_path.name}")
