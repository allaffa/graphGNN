"""Pixel cleanup of one grain-id slice: the smallest edits that make it representable as a trivalent polygon network.

remove_islands      every grain keeps its largest 4-connected piece; the other pieces are repainted from the
                    outside in, each cell taking the id of the surrounding grain it touches most
split_bad_corners   a corner where four boundary segments meet (four ids, or a grain touching itself across the
                    diagonal) cannot be a trivalent junction; the one cell with least support in its 3x3
                    neighbourhood is changed to the best-supported other id of the block
Both return (image, number of changed cells). No other cell is touched: this is not a smoothing filter.
On the ExaCA runs seen so far the two together change about 0.05% of the cells of a slice.
"""
import numpy as np
from .periodic_grid import roll, window, connected_components

SIDES = ((1, 0), (-1, 0), (0, 1), (0, -1))             # (dy, dx) offsets of the four 4-neighbours
DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))      # (dy, dx) offsets of the four diagonal neighbours


def island_mask(S, periodic):
    """True for every cell that is not in the largest 4-connected piece of its grain.
    Among equal-sized pieces the one whose first cell comes first in raster order is kept."""
    labels, n = connected_components(S, periodic)
    size = np.bincount(labels.ravel(), minlength=n)
    grain_of = np.empty(n, S.dtype)
    grain_of[labels.ravel()] = S.ravel()                       # the grain id of every piece
    by_grain_then_size = np.lexsort((-size, grain_of))          # pieces grouped by grain, largest first (stable on ties)
    sorted_grains = grain_of[by_grain_then_size]
    first_of_grain = np.r_[True, sorted_grains[1:] != sorted_grains[:-1]]
    keep = np.zeros(n, bool)
    keep[by_grain_then_size[first_of_grain]] = True
    return ~keep[labels]


def flood_islands(S, island, periodic):
    """Repaint the island cells from the outside in.

    In each pass every island cell that touches a stable cell (one that is not an island, or was repainted in an
    earlier pass) takes an id from its stable 4-neighbours: the id held by most of them; on a tie the id also
    held by most stable diagonal neighbours; on a second tie the smaller id. Cells deeper inside a piece wait for
    the next pass, so a multi-cell piece is shared between the grains around it in proportion to contact.
    """
    S = S.copy()
    island = island.copy()
    n_ids = int(S.max()) + 1                                   # ids are 0..n_ids-1; used to pack the tie-break key

    def neighbour_values(A, offsets):
        """A at the given (dy, dx) offsets, taken at the island cells: shape (n_island, len(offsets)).
        Positions outside a non-periodic image read -1."""
        shifted = []
        for dy, dx in offsets:
            R = roll(A, dy, 0, periodic) if dy else A
            shifted.append(roll(R, dx, 1, periodic) if dx else R)
        return np.stack(shifted, -1)[island]

    while island.any():
        stable = (~island).astype(np.int8)                     # 1 where the id is final
        sides = neighbour_values(S, SIDES)
        sides = np.where(neighbour_values(stable, SIDES) == 1, sides, -1)            # only stable ids may vote
        diagonals = neighbour_values(S, DIAGONALS)
        diagonals = np.where(neighbour_values(stable, DIAGONALS) == 1, diagonals, -1)
        # For each of the four side slots: how many stable sides and how many stable diagonals hold the same id.
        side_votes = (sides[:, :, None] == sides[:, None, :]).sum(-1)
        diagonal_votes = (sides[:, :, None] == diagonals[:, None, :]).sum(-1)
        # One integer per slot that sorts by (side votes, diagonal votes, smaller id): both vote counts are at
        # most 4, so side_votes * 5 + diagonal_votes orders them lexicographically; multiplying by n_ids and
        # adding (n_ids - 1 - id) then prefers the smaller id among equal votes.
        score = (side_votes * 5 + diagonal_votes) * n_ids + (n_ids - 1 - sides)
        score[sides < 0] = -1                                  # empty or unstable slots never win
        best_slot = score.argmax(1)
        ready = score.max(1) >= 0                              # island cells with at least one stable 4-neighbour
        if not ready.any():
            break
        yy, xx = np.where(island)
        new_ids = sides[np.arange(len(best_slot)), best_slot]
        S[yy[ready], xx[ready]] = new_ids[ready]
        island[yy[ready], xx[ready]] = False
    return S


def remove_islands(S, periodic, max_iter=10):
    """Keep the largest component of every grain and repaint the other pieces from the surrounding grains.

    Repainting joins every island cell to a kept piece, so new islands cannot appear; the loop normally runs
    twice (repaint, then confirm that nothing is left). max_iter only guards against a pathological image.
    Returns (cleaned image, number of repainted cells).
    """
    S = S.copy()
    changed = 0
    for _ in range(max_iter):
        island = island_mask(S, periodic)
        if not island.any():
            break
        changed += int(island.sum())
        S = flood_islands(S, island, periodic)
    return S, changed


def corner_blocks(S, periodic):
    """The 2x2 block of ids around every corner.

    Corner (i, j) is the grid vertex shared by cells (i, j), (i+1, j), (i, j+1), (i+1, j+1). The result has shape
    (H, W, 4) with those four ids along the last axis, -1 where the block reaches outside a non-periodic image.
    """
    here = S
    below = roll(S, -1, 0, periodic)
    right = roll(S, -1, 1, periodic)
    below_right = roll(below, -1, 1, periodic)
    return np.stack([here, below, right, below_right], -1)


def corner_degree(blocks):
    """Number of boundary segments meeting at each corner: 0 inside a grain, 2 on a boundary, 3 at a junction,
    4 at a defect (four ids, or two ids in a checkerboard). blocks is the (H, W, 4) array of corner_blocks."""
    here, below, right, below_right = (blocks[..., k] for k in range(4))
    return (here != below).astype(int) + (below != below_right) + (below_right != right) + (right != here)


def split_bad_corners(S, periodic, max_pass=4):
    """Resolve every corner of degree 4 by changing one cell of its 2x2 block.

    The cell changed is the one whose own id has the least support (fewest cells of that id) in its 3x3
    neighbourhood; it takes the other id of the block with the most support there. Support windows wrap only on
    periodic axes. One change can create a new degree-4 corner next door, hence up to max_pass sweeps; one or two
    are needed in practice. Returns (image, number of changed cells).
    """
    S = S.copy()
    changed = 0
    H, W = S.shape

    def support(cell, g):
        """Cells of id g in the 3x3 neighbourhood of `cell`, the cell itself excluded."""
        ys = window(cell[0], H, periodic[1])
        xs = window(cell[1], W, periodic[0])
        return int(np.sum(S[np.ix_(ys, xs)] == g)) - int(S[cell] == g)

    for _ in range(max_pass):
        blocks = corner_blocks(S, periodic)
        inside = blocks.min(-1) >= 0
        bad = list(zip(*np.where((corner_degree(blocks) == 4) & inside)))
        if not bad:
            break
        for i, j in bad:
            i_below = (i + 1) % H if periodic[1] else i + 1
            j_right = (j + 1) % W if periodic[0] else j + 1
            cells = [(i, j), (i_below, j), (i, j_right), (i_below, j_right)]
            ids = [int(S[c]) for c in cells]
            still_bad = (ids[0] != ids[1]) + (ids[1] != ids[3]) + (ids[3] != ids[2]) + (ids[2] != ids[0]) == 4
            if not still_bad:
                continue                                       # resolved by an earlier change in this sweep
            weakest = min(range(4), key=lambda q: support(cells[q], ids[q]))
            other_ids = [g for q, g in enumerate(ids) if q != weakest and g != ids[weakest]]
            S[cells[weakest]] = max(other_ids, key=lambda g: support(cells[weakest], g))
            changed += 1
    return S, changed
