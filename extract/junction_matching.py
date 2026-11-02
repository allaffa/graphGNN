"""Match the junctions of two consecutive slices (layer 2 of the extractor).

A junction keeps its identity for as long as the same three grains meet at it, so most junctions of a slice are
found again in the next slice by their triple alone. The junctions that are not are the ones touched by an event
in between the two slices:
  neighbour switch    a boundary shrinks to a point and a new one opens between the other two grains; the two
                      junctions at its ends each trade one grain of their triple for another
  grain elimination   a grain with n sides disappears; its n junctions go and n - 2 new ones appear among its
                      former neighbours (a two-sided lens leaves none)
In both cases an old junction and the new one that replaces it share two grains and sit close together, so the
leftovers are paired by an assignment that allows exactly such pairs and minimises the total distance. Old
junctions still left over end here, new ones still left over start here. Which of the two junctions of a switch
inherits which identity is decided by distance alone; GrainGNN masks these junctions in its loss, so only the
pairing itself matters there.
"""
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from .periodic_grid import wrap_delta

FORBIDDEN = 1e9                     # cost of a pair the assignment must not make


@dataclass
class Matching:
    """How the junctions of a slice map to those of the next slice."""
    matched: dict                    # jid in prev -> jid in cur
    ended: list                      # jids of prev with no successor
    started: list                    # jids of cur with no predecessor
    changed: set                     # jids of prev whose successor has a different triple


def distance(p, q, shape, periodic):
    """Shortest distance between two points (x, y), across periodic seams where the axis is periodic."""
    H, W = shape
    return np.hypot(wrap_delta(q[0] - p[0], W, periodic[0]), wrap_delta(q[1] - p[1], H, periodic[1]))


def assign(old, new, cost):
    """Minimum-cost pairing of old and new items; cost is len(old) x len(new), forbidden pairs are dropped."""
    if cost.size == 0:
        return []
    rows, cols = linear_sum_assignment(cost)
    return [(old[i], new[j]) for i, j in zip(rows, cols) if cost[i, j] < FORBIDDEN]


def match_junctions(prev, cur, max_jump=None):
    """Link the junctions of SliceGraph prev to those of SliceGraph cur (same shape and periodicity).

    Stage 1 matches by triple. A triple seen once in each slice is a match. A triple seen several times (the two
    junctions of a lens) is resolved by distance. Stage 2 pairs the leftovers whose triples share two grains,
    again by minimum total distance; max_jump (cells) optionally forbids pairs farther apart than that.
    """
    shape, periodic = cur.shape, cur.periodic
    matched, changed, taken = {}, set(), set()
    prev_by_triple, cur_by_triple = defaultdict(list), defaultdict(list)
    for junction in prev.junctions.values():
        prev_by_triple[junction.triple].append(junction.jid)
    for junction in cur.junctions.values():
        cur_by_triple[junction.triple].append(junction.jid)

    leftover_prev, leftover_cur = [], []
    for triple, olds in prev_by_triple.items():
        news = cur_by_triple.get(triple, [])
        if len(olds) == 1 and len(news) == 1:
            matched[olds[0]] = news[0]
            taken.add(news[0])
        elif olds and news:
            cost = np.array([[distance(prev.junctions[o].pos, cur.junctions[n].pos, shape, periodic)
                              for n in news] for o in olds])
            for o, n in assign(olds, news, cost):
                matched[o] = n
                taken.add(n)
            leftover_prev += [o for o in olds if o not in matched]
            leftover_cur += [n for n in news if n not in taken]
        else:
            leftover_prev += olds
    for triple, news in cur_by_triple.items():
        if triple not in prev_by_triple:
            leftover_cur += news

    if leftover_prev and leftover_cur:
        cost = np.full((len(leftover_prev), len(leftover_cur)), FORBIDDEN)
        for a, o in enumerate(leftover_prev):
            old = prev.junctions[o]
            for b, n in enumerate(leftover_cur):
                new = cur.junctions[n]
                if len(set(old.triple) & set(new.triple)) == 2:
                    d = distance(old.pos, new.pos, shape, periodic)
                    if max_jump is None or d <= max_jump:
                        cost[a, b] = d
        for o, n in assign(leftover_prev, leftover_cur, cost):
            matched[o] = n
            taken.add(n)
            changed.add(o)

    ended = sorted(jid for jid in prev.junctions if jid not in matched)
    started = sorted(jid for jid in cur.junctions if jid not in taken)
    return Matching(matched, ended, started, changed)
