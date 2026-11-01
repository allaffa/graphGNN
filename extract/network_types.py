"""The objects of a per-slice grain-boundary network. Coordinates are cells: x along columns, y along rows."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Junction:
    """A point where three grains meet, possibly smeared over several adjacent grid corners."""
    jid: int
    triple: tuple                # the three grain ids meeting here (sorted)
    pos: np.ndarray              # (x, y), refined by the line fit
    pos_raw: np.ndarray          # (x, y), mean of the corner positions
    corners: list                # grid corners (i, j) forming this junction
    edges: list = field(default_factory=list)


@dataclass
class Edge:
    """One grain boundary between two junctions. Two junctions may share several edges (a lens grain)."""
    eid: int
    j1: int
    j2: int
    pair: tuple                  # the two grains this boundary separates (sorted)
    path: np.ndarray             # corner points (x, y) along the boundary, j1 -> j2, unwrapped relative to j1
    length_path: float           # along the pixel boundary (cells)
    length: float                # straight junction-to-junction distance (cells, periodic)


@dataclass
class Grain:
    """One grain of the slice as a polygon: junctions and edges in cyclic order, pixel area and centroid."""
    gid: int
    area: int                    # cells
    centroid: np.ndarray         # (x, y), mean of its junction positions (unwrapped), the legacy convention
    neighbours: set              # grain ids sharing an edge (the wall id included when walled)
    junctions: list              # cyclic order around the polygon
    edges: list                  # cyclic order around the polygon
    two_sided: bool              # exactly two neighbours: a lens grain


@dataclass
class SliceGraph:
    """The validated network of one slice, as returned by build_slice_graph."""
    shape: tuple                 # (H, W) of `cleaned`, padded by one cell on each walled axis
    periodic: tuple              # (x, y)
    junctions: dict              # jid -> Junction
    edges: dict                  # eid -> Edge
    grains: dict                 # gid -> Grain (the wall is not a grain)
    valid: bool
    report: dict
    cleaned: np.ndarray          # the image after cleanup, in the coordinates of all positions
