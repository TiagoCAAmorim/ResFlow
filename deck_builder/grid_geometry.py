"""Grid geometry (CORNERS.inc) for a flat, regular 2D grid."""
from pathlib import Path

import numpy as np


def _compact_list(data: list[float]) -> str:
    """Converts a list of numbers into a compact string representation."""
    if not data:
        return ""

    counts = []
    if len(data) > 1:
        count = 1
        for i in range(1, len(data)):
            if data[i] == data[i - 1]:
                count += 1
            else:
                counts.append((count, data[i - 1]))
                count = 1
        counts.append((count, data[-1]))
    else:
        counts.append((1, data[0]))

    parts = []
    for count, value in counts:
        if count > 1:
            parts.append(f"{count}*{value:.2f}")
        else:
            parts.append(f"{value:.2f}")
    return " ".join(parts)


def _write_long_string(file_handle, text: str, max_width: int = 80) -> None:
    """Writes a long string, breaking it into lines of max_width."""
    words = text.split()
    current_line = ""
    for word in words:
        if not current_line:
            current_line = word
        elif len(current_line) + 1 + len(word) <= max_width:
            current_line += " " + word
        else:
            file_handle.write(current_line + "\n")
            current_line = word
    if current_line:
        file_handle.write(current_line + "\n")


def _compact_and_write_coordinates(line_width, f, lines) -> None:
    """Compacts and writes coordinate data to the file."""
    all_coord_data_float = np.concatenate(lines).astype(float).tolist()
    compacted_string = _compact_list(all_coord_data_float)
    _write_long_string(f, compacted_string, line_width)


def _get_node_positions(cell_widths, num_cells: int) -> np.ndarray:
    """
    Generate node positions from cell widths.

    Args:
        cell_widths: Either a float (uniform cells) or array-like (non-uniform cells)
        num_cells: Number of cells in this direction

    Returns:
        Array of node positions (length = num_cells + 1)
    """
    if isinstance(cell_widths, (list, tuple, np.ndarray)):
        cell_widths = np.asarray(cell_widths, dtype=float)
        if len(cell_widths) != num_cells:
            raise ValueError(
                f"Length of cell widths ({len(cell_widths)}) must equal "
                f"number of cells ({num_cells})"
            )
        return np.concatenate([[0], np.cumsum(cell_widths)])
    else:
        return np.arange(num_cells + 1) * float(cell_widths)


def write_corners_file(
    dims: tuple[int, int, int],
    cell_sides: tuple,
    origin: tuple[float, float],
    top: float,
    output_path: str | Path,
    include_comments: bool = True,
    line_width: int = 80,
) -> None:
    """
    Generate a CORNERS file for a flat (no dip/azimuth/disalignment) corner-point grid.

    Args:
        dims: Number of cells in the i, j, and k directions.
        cell_sides: Cell widths for each direction. Each entry may be a float
            for uniform spacing or an array-like of floats for non-uniform widths.
        origin: (x, y) coordinates of the NW corner of cell (1, 1, 1).
        top: Depth of the top layer at the origin.
        output_path: Path where the output file will be written.
        include_comments: Whether to include comment lines in the file.
        line_width: Maximum width of lines in the output file.
    """
    ni, nj, nk = dims
    di, dj, dk = cell_sides

    x_nodes = _get_node_positions(di, ni)
    y_nodes = _get_node_positions(dj, nj)
    z_nodes = _get_node_positions(dk, nk)

    nodes = np.zeros((ni + 1, nj + 1, nk + 1, 3))
    nodes[..., 0] = x_nodes[:, np.newaxis, np.newaxis]
    nodes[..., 1] = y_nodes[np.newaxis, :, np.newaxis]
    nodes[..., 2] = z_nodes[np.newaxis, np.newaxis, :]

    # Node order: NW-T, NE-T, SW-T, SE-T, NW-B, NE-B, SW-B, SE-B
    cell_corners = np.zeros((ni, nj, nk, 8, 3))
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                cell_corners[i, j, k, 0, :] = nodes[i, j, k, :]
                cell_corners[i, j, k, 1, :] = nodes[i + 1, j, k, :]
                cell_corners[i, j, k, 2, :] = nodes[i, j + 1, k, :]
                cell_corners[i, j, k, 3, :] = nodes[i + 1, j + 1, k, :]
                cell_corners[i, j, k, 4, :] = nodes[i, j, k + 1, :]
                cell_corners[i, j, k, 5, :] = nodes[i + 1, j, k + 1, :]
                cell_corners[i, j, k, 6, :] = nodes[i, j + 1, k + 1, :]
                cell_corners[i, j, k, 7, :] = nodes[i + 1, j + 1, k + 1, :]

    cell_corners[..., 0] += origin[0]
    cell_corners[..., 1] += origin[1]
    cell_corners[..., 2] += top

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("** Input Parameters:\n")
        f.write(f"** Dimensions (ni, nj, nk): {dims}\n")
        f.write(f"** Cell Sides (dx, dy, dz): {cell_sides}\n")
        f.write(f"** Origin (x, y): {origin}\n")
        f.write(f"** Top Depth: {top}\n")
        f.write(f"** Line Width: {line_width}\n")
        f.write("\nCORNERS\n")

        coord_names = ["x-coord", "y-coord", "z-coord"]
        for coord_idx, coord_name in enumerate(coord_names):
            if include_comments:
                f.write(f"** {coord_name}\n")

            for k in range(nk):
                if include_comments:
                    f.write(f"** K={k + 1}, {coord_name}, Top face\n")
                lines = []
                for j in range(nj):
                    lines.append(cell_corners[:, j, k, [0, 1], coord_idx].flatten())
                    lines.append(cell_corners[:, j, k, [2, 3], coord_idx].flatten())
                _compact_and_write_coordinates(line_width, f, lines)
                if include_comments:
                    f.write("\n")

                if include_comments:
                    f.write(f"** K={k + 1}, {coord_name}, Bottom face\n")
                lines = []
                for j in range(nj):
                    lines.append(cell_corners[:, j, k, [4, 5], coord_idx].flatten())
                    lines.append(cell_corners[:, j, k, [6, 7], coord_idx].flatten())
                _compact_and_write_coordinates(line_width, f, lines)
                if include_comments:
                    f.write("\n")
            if not include_comments:
                f.write("\n")
