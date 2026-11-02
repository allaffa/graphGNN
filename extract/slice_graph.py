"""Per-slice grain-boundary network of a 2D grain-id image (layer 1 of the extractor).

build_slice_graph(S) runs, in order:
  cleanup      slice_cleanup.remove_islands and slice_cleanup.split_bad_corners (reported, never silent)
  junctions    boundary_tracing.find_junctions: junction corners, clusters, traced edges, lens handling
  positions    refine_junctions: intersect lines fitted to the start of the incident boundary paths
  grains       build_grains: polygon cycle, area (cells), centroid (mean of junction positions), neighbours
  validation   every junction degree 3, every grain one closed cycle, Euler characteristic of the domain
Coordinates are cells: x along columns, y along rows; corner (i, j) is at (x, y) = (j + 0.5, i + 0.5).
Grain ids must be >= 0. For a non-periodic axis pass wall_id: the image is padded by one cell of that id, all
positions refer to the padded image, and the wall appears as a neighbour of the grains touching it, not as a grain.
Two-sided grains (lenses, two junctions joined by two edges) and enclosed grains (one neighbour, no junctions) are
legal states of a shrinking grain and are reported, not rejected.
"""
from collections import defaultdict, Counter
import numpy as np
from .periodic_grid import roll, wrap_delta, wrap_point
from .slice_cleanup import remove_islands, split_bad_corners, corner_blocks
from .boundary_tracing import boundary_segments, find_junctions
from .network_types import Grain, SliceGraph


def refine_junctions(junctions, edges, shape, periodic, fit_len=8, max_shift=2.0, max_condition=1e6):
    """Move each junction from its grid corner to where its boundaries meet.

    Close to a junction each boundary is nearly straight, so a line fitted to the first fit_len corners of each
    incident path (8 cells: enough to average out the pixel staircase, short enough that curvature does not bias
    it) locates that boundary to a fraction of a cell. The junction becomes the least-squares intersection of the
    fitted lines. The raw position is kept when fewer than two paths have 3 or more points to fit, when the lines
    are nearly parallel (condition number above max_condition), or when the intersection lies more than max_shift
    cells from the corner, which only happens when a fit went wrong.
    """
    H, W = shape
    for junction in junctions.values():
        normal_matrix = np.zeros((2, 2))                       # sum of n n^T over the fitted lines
        rhs = np.zeros(2)                                      # sum of n n^T m over the fitted lines
        n_lines = 0
        for eid in junction.edges:
            edge = edges[eid]
            if edge.j1 == junction.jid:
                path = edge.path
            else:                                              # stored from the other end: reverse and re-anchor here
                path = edge.path[::-1]
                start = np.array([junction.pos_raw[0] + wrap_delta(path[0, 0] - junction.pos_raw[0], W, periodic[0]),
                                  junction.pos_raw[1] + wrap_delta(path[0, 1] - junction.pos_raw[1], H, periodic[1])])
                path = path - path[0] + start
            points = path[:min(len(path), fit_len)]
            if len(points) < 3:
                continue
            centre = points.mean(0)
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov((points - centre).T))
            normal = eigenvectors[:, 0]                        # direction of least spread: the line's normal
            normal_matrix += np.outer(normal, normal)
            rhs += np.outer(normal, normal) @ centre
            n_lines += 1
        if n_lines < 2 or np.linalg.cond(normal_matrix) >= max_condition:
            continue
        intersection = np.linalg.solve(normal_matrix, rhs)
        if np.hypot(*(intersection - junction.pos_raw)) <= max_shift:
            junction.pos = wrap_point(intersection, shape, periodic)


def polygon_cycle(grain_edges, edges):
    """Walk a grain's edges from junction to junction around its polygon.

    At each junction of a trivalent network a grain borders exactly two edges, so the walk is forced. Returns
    (junctions in cyclic order, edges in cyclic order, closed); closed is False when the walk does not return to
    its start or leaves edges unused, which means a hole or a second piece.
    """
    if not grain_edges:
        return [], [], True
    used = set()
    start = edges[grain_edges[0]].j1
    current = start
    cycle_junctions = [current]
    cycle_edges = []
    while True:
        candidates = [eid for eid in grain_edges if eid not in used and current in (edges[eid].j1, edges[eid].j2)]
        if not candidates:
            break
        eid = candidates[0]
        used.add(eid)
        cycle_edges.append(eid)
        current = edges[eid].j2 if edges[eid].j1 == current else edges[eid].j1
        if current == start:
            break
        cycle_junctions.append(current)
    closed = current == start and len(used) == len(grain_edges)
    return cycle_junctions, cycle_edges, closed


def image_neighbours(S, gid, periodic):
    """Ids sharing a cell side with grain gid, read from the image. Used for a grain that has no junctions, whose
    boundary is a closed loop inside a single neighbour and therefore carries no edge."""
    mask = S == gid
    ids = set()
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        ids.update(np.unique(roll(S, shift, axis, periodic)[mask]).tolist())
    ids.discard(gid)
    ids.discard(-1)
    return ids


def build_grains(S, junctions, edges, periodic, wall_id=None):
    """One Grain per id in the cleaned image, the wall excluded.

    Neighbours and incident edges come from the edge pairs; the area is the cell count; the centroid is the mean
    of the polygon's junction positions, unwrapped across periodic seams, which is the legacy GrainGNN definition.
    A grain without junctions (enclosed by one neighbour) gets its neighbour from the image and its centroid from
    its cells. Returns (grains: gid -> Grain, number of grains whose polygon walk did not close).
    """
    H, W = S.shape
    neighbours = defaultdict(set)
    edges_of_grain = defaultdict(list)
    for edge in edges.values():
        a, b = edge.pair
        neighbours[a].add(b)
        neighbours[b].add(a)
        edges_of_grain[a].append(edge.eid)
        edges_of_grain[b].append(edge.eid)
    ids, counts = np.unique(S, return_counts=True)
    for gid in ids.tolist():
        if gid != wall_id and gid not in edges_of_grain:                 # enclosed grain
            for other in image_neighbours(S, gid, periodic):
                neighbours[gid].add(other)
                neighbours[other].add(gid)
    grains = {}
    n_open = 0
    for gid, area in zip(ids.tolist(), counts.tolist()):
        if gid == wall_id:
            continue
        cycle_junctions, cycle_edges, closed = polygon_cycle(edges_of_grain[gid], edges)
        n_open += not closed
        if cycle_junctions:
            points = np.array([junctions[jid].pos for jid in cycle_junctions])
            ref = points[0]
            dx = wrap_delta(points[:, 0] - ref[0], W, periodic[0])
            dy = wrap_delta(points[:, 1] - ref[1], H, periodic[1])
            unwrapped = ref + np.column_stack([dx, dy])
            centroid = wrap_point(unwrapped.mean(0), S.shape, periodic)
        else:                                                  # enclosed grain, or one grain filling the slice
            yy, xx = np.where(S == gid)
            centroid = np.array([xx.mean(), yy.mean()])
        two_sided = len(neighbours[gid]) == 2
        grains[gid] = Grain(gid, area, centroid, neighbours[gid], cycle_junctions, cycle_edges, two_sided)
    return grains, n_open


def validate(junctions, edges, grains, n_open_cycles, periodic, wall_id=None):
    """Consistency checks and summary statistics. Returns (valid, report).

    valid requires every junction to have degree 3, every grain's polygon to close, and the Euler characteristic
    V - E + F of the domain: 0 on the torus (both axes periodic), 2 on a disk (both axes walled, the wall counted
    as the outer face); a cylinder (one periodic axis) is not checked. The Euler count catches global defects,
    such as a lost junction or a ring-shaped grain, that the local checks can miss. Enclosed grains have no
    junctions and are not faces of the junction network, so they are left out of F and counted separately.
    """
    degree_hist = Counter(len(junction.edges) for junction in junctions.values())
    V = len(junctions)
    E = len(edges)
    enclosed = sum(not grain.junctions for grain in grains.values())
    F = len(grains) - enclosed + (1 if wall_id is not None else 0)
    euler = V - E + F
    if all(periodic):
        euler_ok = euler == 0
    elif wall_id is not None and not any(periodic):
        euler_ok = euler == 2
    else:
        euler_ok = True
    triple_counts = Counter(junction.triple for junction in junctions.values())
    report = dict(junctions=V, edges=E, grains=len(grains), euler=euler, degree_hist=dict(degree_hist),
                  bad_cycles=n_open_cycles,
                  lenses=sum(grain.two_sided for grain in grains.values()), enclosed=enclosed,
                  duplicate_triples=sum(count > 1 for count in triple_counts.values()))
    valid = euler_ok and n_open_cycles == 0 and set(degree_hist) <= {3}
    return valid, report


def build_slice_graph(S, periodic=(True, True), wall_id=None, cleanup=True, fit_len=8, refine=True):
    """Build the grain-boundary network of one slice (see the module docstring).

    S          2D integer grain ids, S[i, j] with i = y and j = x
    periodic   (x, y) periodicity flags
    wall_id    id used to pad each non-periodic axis by one cell (required when an axis is not periodic)
    cleanup    run remove_islands and split_bad_corners first (False only for diagnostics)
    fit_len    corners per boundary used by the line fit of refine_junctions
    refine     replace the corner-based junction positions by the line-fit intersections
    """
    S = np.asarray(S).astype(np.int32)
    if S.min() < 0:
        raise ValueError('grain ids must be >= 0 (-1 marks the outside of a non-periodic axis)')
    if wall_id is not None:
        pad_rows = (0, 0) if periodic[1] else (1, 1)
        pad_cols = (0, 0) if periodic[0] else (1, 1)
        S = np.pad(S, [pad_rows, pad_cols], constant_values=wall_id)
    report = {}
    if cleanup:
        S, n_island_cells = remove_islands(S, periodic)
        S, n_corner_cells = split_bad_corners(S, periodic)
        report.update(island_cells=n_island_cells, corner_cells=n_corner_cells)
    blocks = corner_blocks(S, periodic)
    adj = boundary_segments(S, periodic)
    junctions, edges, n_split = find_junctions(S, blocks, adj, periodic)
    report['split_clusters'] = n_split
    if refine:
        refine_junctions(junctions, edges, S.shape, periodic, fit_len)
    grains, n_open = build_grains(S, junctions, edges, periodic, wall_id)
    valid, stats = validate(junctions, edges, grains, n_open, periodic, wall_id)
    report.update(stats)
    return SliceGraph(S.shape, periodic, junctions, edges, grains, valid, report, S)
