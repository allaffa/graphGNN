"""The per-run trajectory store written by layer 2 and read by the sample builder.

Everything is indexed by layer k and, for junctions, by slot. A slot is a row that a junction occupies for as long
as it is tracked; when the junction vanishes, the slot is handed to a junction created by the same event (or kept
free for a later one). This is the form GrainGNN consumes: a fixed set of junction rows whose occupants come and
go. The `track` array distinguishes successive occupants of the same slot. Grains are indexed by row in `gids`.
Positions are in cells of the (padded, when walled) slice image; heights are in um.
"""
from dataclasses import dataclass
import json
import numpy as np
from scipy.signal import savgol_filter
from .periodic_grid import wrap_delta, wrap_point

REPORT_KEYS = ('junctions', 'edges', 'grains', 'island_cells', 'corner_cells', 'split_clusters',
               'lenses', 'enclosed', 'duplicate_triples')
EVENT_DTYPE = np.dtype([('kind', 'U10'),            # eliminate, two_sided, enclosed, switch, appear
                        ('k', np.int32),            # first layer where the new state is seen
                        ('grain', np.int32),        # the grain concerned, -1 for a switch
                        ('pair_old', np.int32, 2),  # switch: grains separated by the edge before (-1 otherwise)
                        ('pair_new', np.int32, 2),  # switch: grains separated by the edge after
                        ('slots', np.int32, 2),     # switch: the two junction slots at the ends of the edge
                        ('n', np.int32)])           # eliminate: sides before; appear/two_sided/enclosed: sides now

ARRAY_FIELDS = ('valid', 'report', 'gids', 'theta', 'area', 'centroid', 'n_sides', 'pos', 'pos_raw', 'triple',
                'track', 'edge_offsets', 'edge_slots', 'edge_pair', 'edge_len_path', 'events')


@dataclass
class Trajectory:
    """Grain and junction tracks of one stack; see the module docstring for the indexing."""
    shape: tuple                 # (H, W) of the slice images used for positions
    periodic: tuple              # (x, y)
    cell: float                  # um per cell in the plane
    dz: float                    # um between layers
    z0: float                    # height of layer 0 (um)
    source: str
    valid: np.ndarray            # (K,) bool, layer-1 validation of each slice
    report: np.ndarray           # (K, len(REPORT_KEYS)) int, layer-1 counts per slice
    gids: np.ndarray             # (G,) grain id of each grain row
    theta: np.ndarray            # (G, 2) theta_x, theta_z in radians, nan when unknown
    area: np.ndarray             # (K, G) cells, 0 when the grain is absent
    centroid: np.ndarray         # (K, G, 2) mean of the junction positions, nan when absent
    n_sides: np.ndarray          # (K, G) number of neighbours, 0 when absent
    pos: np.ndarray              # (K, J, 2) refined junction position, nan when the slot is empty
    pos_raw: np.ndarray          # (K, J, 2) corner-based position
    triple: np.ndarray           # (K, J, 3) sorted grain ids, -1 when empty
    track: np.ndarray            # (K, J) id of the junction occupying the slot, -1 when empty
    edge_offsets: np.ndarray     # (K + 1,) edges of layer k are rows edge_offsets[k]:edge_offsets[k + 1]
    edge_slots: np.ndarray       # (E, 2) junction slots at the two ends
    edge_pair: np.ndarray        # (E, 2) grains separated by the edge
    edge_len_path: np.ndarray    # (E,) length along the pixel boundary (cells)
    events: np.ndarray           # structured, EVENT_DTYPE
    pos_smooth: np.ndarray = None   # (K, J, 2) after smooth_tracks, else None
    smooth_window: int = 0

    @property
    def n_layers(self):
        return self.area.shape[0]

    @property
    def n_slots(self):
        return self.pos.shape[1]

    def heights(self):
        return self.z0 + self.dz * np.arange(self.n_layers)

    def alive(self):
        """(K, G) bool: the grain has cells in the layer."""
        return self.area > 0

    def occupied(self):
        """(K, J) bool: the slot holds a junction in the layer."""
        return self.track >= 0

    def edges_of(self, k):
        """Rows of the edge arrays belonging to layer k."""
        return slice(int(self.edge_offsets[k]), int(self.edge_offsets[k + 1]))

    def events_of(self, kind):
        return self.events[self.events['kind'] == kind]

    def save(self, path):
        meta = dict(shape=list(self.shape), periodic=list(self.periodic), cell=self.cell, dz=self.dz, z0=self.z0,
                    source=self.source, smooth_window=self.smooth_window, report_keys=list(REPORT_KEYS))
        arrays = {name: getattr(self, name) for name in ARRAY_FIELDS}
        if self.pos_smooth is not None:
            arrays['pos_smooth'] = self.pos_smooth
        np.savez_compressed(path, meta=json.dumps(meta), **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['meta']))
            arrays = {name: data[name] for name in ARRAY_FIELDS}
            pos_smooth = data['pos_smooth'] if 'pos_smooth' in data else None
        return cls(shape=tuple(meta['shape']), periodic=tuple(meta['periodic']), cell=meta['cell'], dz=meta['dz'],
                   z0=meta['z0'], source=meta['source'], pos_smooth=pos_smooth,
                   smooth_window=meta['smooth_window'], **arrays)



def unwrap(points, shape, periodic):
    """Make a sequence of positions continuous across periodic seams: each step is taken the short way round."""
    H, W = shape
    steps = np.diff(points, axis=0)
    steps[:, 0] = wrap_delta(steps[:, 0], W, periodic[0])
    steps[:, 1] = wrap_delta(steps[:, 1], H, periodic[1])
    return points[0] + np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])


def track_segments(traj, slot):
    """Runs of layers [a, b) over which the slot holds the same junction with the same triple."""
    occupied = traj.track[:, slot] >= 0
    same = (traj.track[1:, slot] == traj.track[:-1, slot]) & np.all(traj.triple[1:, slot] == traj.triple[:-1, slot], axis=1)
    starts = np.flatnonzero(occupied & ~np.concatenate([[False], same]))
    ends = np.flatnonzero(occupied & ~np.concatenate([same, [False]])) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def smooth_tracks(traj, window):
    """Fill pos_smooth with a straight-line fit of each junction track along z.

    The raw positions sit on the pixel grid and so move in steps; a junction that drifts steadily shows a
    staircase. At every layer the smoothed position is the value of the line fitted to the track over `window`
    layers centred on that layer (Savitzky-Golay of order 1). At the ends of a track the line fitted to the last
    `window` layers is used, so a steady drift is reproduced exactly there too, which a running mean would not do.
    Tracks are cut where the junction changes its triple, so an event does not bleed into the layers around it.
    Shorter tracks than the window are fitted with a shorter odd window; tracks of fewer than 3 layers stay raw.
    """
    window = int(window)
    out = np.full_like(traj.pos, np.nan)
    for slot in range(traj.n_slots):
        for a, b in track_segments(traj, slot):
            points = unwrap(traj.pos[a:b, slot], traj.shape, traj.periodic)
            n = b - a
            w = min(window, n if n % 2 else n - 1)
            if w >= 3:
                points = savgol_filter(points, w, 1, axis=0, mode='interp')
            for k in range(n):
                out[a + k, slot] = wrap_point(points[k], traj.shape, traj.periodic)
    traj.pos_smooth = out
    traj.smooth_window = window
    return traj
