"""From a cleaned slice to junctions and traced boundaries.

Corner (i, j) is the grid vertex shared by cells (i,j), (i+1,j), (i,j+1), (i+1,j+1); it sits at (x, y) = (j+0.5, i+0.5).
A grain boundary runs along cell edges, so it is a chain of unit segments between neighbouring corners.

boundary_segments   the unit segments as an undirected graph on corners, each carrying the pair of grains it separates
junction_corners    corners whose 2x2 block holds three or more distinct ids, with the sorted tuple of those ids
cluster_junctions   adjacent junction corners with the same triple are one junction smeared over the pixel grid
trace_edges         walk the segment graph from junction corner to junction corner; every walk is one edge
find_junctions      cluster + trace, re-clustering with any self-looping cluster split apart (a merged one-cell lens)
"""
from collections import defaultdict
import numpy as np
from .periodic_grid import roll, wrap_delta, wrap_point
from .network_types import Junction, Edge


def boundary_segments(S, periodic):
    """The boundary as a graph on corners.

    Returns adj: corner -> list of (neighbouring corner, (grain a, grain b)) with a < b, one entry per unit
    segment, stored from both of its corners. A corner has 0 entries inside a grain, 2 on a boundary and 3 at a
    junction (4 cannot occur after split_bad_corners).
    """
    H, W = S.shape
    adj = defaultdict(list)
    below = roll(S, -1, 0, periodic)
    right = roll(S, -1, 1, periodic)
    # The bottom edge of cell (i, j) is a boundary segment when the cell below differs; it joins corner (i, j-1)
    # to corner (i, j). On a non-periodic axis a segment whose corner would fall outside the image is skipped.
    for i, j in zip(*np.where((S != below) & (below >= 0))):
        i, j = int(i), int(j)
        left_corner = (i, (j - 1) % W if periodic[0] else j - 1)
        if left_corner[1] < 0:
            continue
        pair = tuple(sorted((int(S[i, j]), int(below[i, j]))))
        adj[left_corner].append(((i, j), pair))
        adj[(i, j)].append((left_corner, pair))
    # The right edge of cell (i, j) is a boundary segment when the cell to the right differs; it joins
    # corner (i-1, j) to corner (i, j).
    for i, j in zip(*np.where((S != right) & (right >= 0))):
        i, j = int(i), int(j)
        upper_corner = ((i - 1) % H if periodic[1] else i - 1, j)
        if upper_corner[0] < 0:
            continue
        pair = tuple(sorted((int(S[i, j]), int(right[i, j]))))
        adj[upper_corner].append(((i, j), pair))
        adj[(i, j)].append((upper_corner, pair))
    return adj


def junction_corners(blocks):
    """{(i, j): ids} for every corner whose 2x2 block (blocks, shape (H, W, 4)) holds three or more distinct ids
    and no outside cell; ids is the sorted tuple of the distinct ids, a triple after cleanup."""
    ordered = np.sort(blocks, -1)
    n_distinct = 1 + (ordered[..., 1:] != ordered[..., :-1]).sum(-1)
    inside = blocks.min(-1) >= 0
    return {(int(i), int(j)): tuple(int(g) for g in np.unique(blocks[i, j]))
            for i, j in zip(*np.where((n_distinct >= 3) & inside))}


def cluster_junctions(corner_triples, split, shape, periodic):
    """Group adjacent junction corners with the same triple into junctions.

    The pixel grid cannot place a junction exactly, so the three grains often show in the blocks of two or three
    neighbouring corners; those corners are one physical junction. Corners listed in `split` are never grouped
    (see find_junctions). The junction position is the mean of its corners, unwrapped across a periodic seam.
    Returns (junctions: jid -> Junction, corner_to_jid).
    """
    H, W = shape
    corner_to_jid = {}
    junctions = {}
    next_jid = 0
    for corner, triple in corner_triples.items():
        if corner in corner_to_jid:
            continue
        stack = [corner]
        members = []
        while stack:                                           # flood fill over the 8 neighbouring corners
            q = stack.pop()
            if q in corner_to_jid:
                continue
            corner_to_jid[q] = next_jid
            members.append(q)
            if q in split:
                continue
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni = (q[0] + di) % H if periodic[1] else q[0] + di
                    nj = (q[1] + dj) % W if periodic[0] else q[1] + dj
                    n = (ni, nj)
                    if n in corner_triples and n not in corner_to_jid and n not in split and corner_triples[n] == triple:
                        stack.append(n)
        points = np.array([[m[1] + 0.5, m[0] + 0.5] for m in members], float)     # corner (i, j) -> (x, y)
        ref = points[0]
        dx = wrap_delta(points[:, 0] - ref[0], W, periodic[0])
        dy = wrap_delta(points[:, 1] - ref[1], H, periodic[1])
        unwrapped = ref + np.column_stack([dx, dy])
        position = wrap_point(unwrapped.mean(0), shape, periodic)
        junctions[next_jid] = Junction(next_jid, triple, position.copy(), position.copy(), members)
        next_jid += 1
    return junctions, corner_to_jid


def trace_edges(junctions, corner_to_jid, adj, shape, periodic):
    """Follow the boundary from every junction corner to the next junction corner.

    Each unit segment is used once: `used` holds it as an unordered pair of corners, so reaching it from either
    end counts. A segment between two corners of the same cluster is internal and skipped. Otherwise the walk
    continues through degree-2 corners, which leave exactly one way forward, until it reaches a corner that
    belongs to a junction. The walk becomes an Edge whose path is unwrapped across periodic seams.
    Returns (edges: eid -> Edge, self_looping: the jids whose walk came back to themselves).
    """
    H, W = shape
    edges = {}
    next_eid = 0
    used = set()
    self_looping = set()
    for jid, junction in junctions.items():
        for corner in junction.corners:
            for neighbour, pair in adj[corner]:
                segment = frozenset((corner, neighbour))
                if segment in used:
                    continue
                used.add(segment)
                if corner_to_jid.get(neighbour) == jid:
                    continue                                   # internal to this cluster
                path = [corner, neighbour]
                prev, cur = corner, neighbour
                while cur not in corner_to_jid:
                    onward = [m for (m, _) in adj[cur] if m != prev]
                    if len(onward) != 1:
                        break                                  # a corner of degree 0 or 4: defect, abandon the walk
                    nxt = onward[0]
                    used.add(frozenset((cur, nxt)))
                    path.append(nxt)
                    prev, cur = cur, nxt
                if cur not in corner_to_jid:
                    continue
                end_jid = corner_to_jid[cur]
                if end_jid == jid:
                    self_looping.add(jid)
                points = np.array([[q[1] + 0.5, q[0] + 0.5] for q in path], float)
                steps_x = wrap_delta(np.diff(points[:, 0]), W, periodic[0])
                steps_y = wrap_delta(np.diff(points[:, 1]), H, periodic[1])
                points = points[0] + np.column_stack([np.cumsum(np.r_[0, steps_x]), np.cumsum(np.r_[0, steps_y])])
                end = junctions[end_jid].pos_raw
                chord = np.array([wrap_delta(end[0] - junction.pos_raw[0], W, periodic[0]),
                                  wrap_delta(end[1] - junction.pos_raw[1], H, periodic[1])])
                edges[next_eid] = Edge(next_eid, jid, end_jid, pair, points, float(len(path) - 1), float(np.hypot(*chord)))
                junction.edges.append(next_eid)
                junctions[end_jid].edges.append(next_eid)
                next_eid += 1
    return edges, self_looping


def find_junctions(S, blocks, adj, periodic, max_attempts=3):
    """Junctions and edges of a cleaned slice.

    A one-cell lens grain has its two junctions on adjacent corners with the same triple, so cluster_junctions
    merges them into one cluster, which then traces its two boundaries back to itself. A real junction never has
    an edge to itself, so each self-looping cluster is dissolved into single-corner junctions and the slice is
    clustered and traced again. One retry has always sufficed; max_attempts bounds it.
    Returns (junctions, edges, number of corners split apart).
    """
    corner_triples = junction_corners(blocks)
    split = set()
    for _ in range(max_attempts):
        junctions, corner_to_jid = cluster_junctions(corner_triples, split, S.shape, periodic)
        edges, self_looping = trace_edges(junctions, corner_to_jid, adj, S.shape, periodic)
        if not self_looping:
            break
        for jid in self_looping:
            split.update(junctions[jid].corners)
    return junctions, edges, len(split)
