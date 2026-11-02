"""Helpers for 2D grain-id images on a grid that is periodic in x, y, both or neither.

Conventions used throughout the package:
  S[i, j]     the grain id of the cell in row i (y) and column j (x)
  periodic    the pair (x, y); array axis 0 is y, array axis 1 is x
  -1          marks "outside the image" in arrays of neighbour values on a non-periodic axis
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components as _csgraph_components


def roll(S, shift, axis, periodic):
    """np.roll along one array axis, with the row or column that wrapped around set to -1 when that axis is
    not periodic. roll(S, 1, 0) therefore holds, at every cell, the id of the cell above it (or -1)."""
    R = np.roll(S, shift, axis)
    axis_is_periodic = periodic[1] if axis == 0 else periodic[0]
    if not axis_is_periodic:
        wrapped = [slice(None), slice(None)]
        wrapped[axis] = 0 if shift > 0 else -1
        R[tuple(wrapped)] = -1
    return R


def wrap_delta(d, L, periodic_axis):
    """Shortest signed difference along an axis of length L: d is reduced into [-L/2, L/2) when the axis is
    periodic and returned unchanged otherwise. Works elementwise on arrays."""
    if not periodic_axis:
        return d
    return d - np.round(d / L) * L


def wrap_point(p, shape, periodic):
    """Bring a point (x, y) back into the image on the periodic axes; shape is (H, W)."""
    H, W = shape
    x = p[0] % W if periodic[0] else p[0]
    y = p[1] % H if periodic[1] else p[1]
    return np.array([x, y])


def window(c, n, periodic_axis):
    """Indices c-1, c, c+1 along an axis of length n: wrapped when the axis is periodic, clipped to 0..n-1
    otherwise, so that a 3x3 neighbourhood never reaches across a wall."""
    candidates = (c - 1, c, c + 1)
    if periodic_axis:
        return [k % n for k in candidates]
    return [k for k in candidates if 0 <= k < n]


def connected_components(S, periodic):
    """Label the 4-connected regions of equal id in one pass.

    Every cell is a node; two 4-neighbours with the same id are linked, including the pairs that face each other
    across a periodic seam. The components of that graph are the pieces of the grains.
    Returns (labels, n): labels[i, j] in 0..n-1, numbered in raster order of each piece's first cell.
    """
    H, W = S.shape
    node = np.arange(H * W).reshape(H, W)
    links = [(node[:-1, :], node[1:, :], S[:-1, :] == S[1:, :]),          # vertical neighbours
             (node[:, :-1], node[:, 1:], S[:, :-1] == S[:, 1:])]          # horizontal neighbours
    if periodic[1]:
        links.append((node[-1, :], node[0, :], S[-1, :] == S[0, :]))      # bottom row to top row
    if periodic[0]:
        links.append((node[:, -1], node[:, 0], S[:, -1] == S[:, 0]))      # last column to first column
    rows = np.concatenate([a[same] for a, b, same in links])
    cols = np.concatenate([b[same] for a, b, same in links])
    graph = sparse.coo_matrix((np.ones(len(rows), bool), (rows, cols)), shape=(H * W, H * W))
    n, labels = _csgraph_components(graph, directed=False)
    return labels.reshape(H, W), n
