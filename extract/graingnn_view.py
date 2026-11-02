"""Reduce a grain-id stack to what GrainGNN can represent (layer 3, part 1).

GrainGNN deletes a grain as soon as it is down to two junctions and never brings a grain back, so a two-sided
(lens) or enclosed grain has no state in the model, and the training data must follow the same rule or the model
is trained on states it can never produce. The rule applied here (option 1 of the layer-2 report):
  dead_schedule   the layer from which a grain counts as dead: the first layer in which it has at most two
                  neighbours, or the first layer in which it has no cells, whichever comes first
  paint_dead      the cells of the dead grains of a slice are handed to the live neighbours that grow into them,
                  one shell at a time, each cell taking the id most of its live neighbours carry
  view_field      the painted stack, ready for track_stack
  track_view      raw tracking -> schedule -> painting -> tracking again, repeated while the painted stack still
                  shows a lens or enclosed grain (painting can turn a neighbour into one)
  schedule_kinds  how the grains of a stack end under the schedule, for reports
Layers 1 and 2 on the painted stack then give exactly the network GrainGNN would hold: a lens becomes one boundary
between its two neighbours, a three-sided remnant becomes one junction, an enclosed grain is absorbed, and a
remnant that flickers after its first disappearance stays gone. The elimination of such a grain is seen at the
layer of its onset, with the number of sides it had the layer before.
"""
import numpy as np
from .grain_field import Field
from .periodic_grid import roll
from .stack_tracking import track_stack


def dead_schedule(traj, max_sides=2):
    """gid -> first layer in which the grain counts as dead (n_layers when it never does), from a raw Trajectory."""
    K = traj.n_layers
    schedule = {}
    for r, gid in enumerate(traj.gids.tolist()):
        alive = traj.area[:, r] > 0
        if not alive.any():
            schedule[gid] = 0
            continue
        first = int(np.argmax(alive))
        gaps = np.flatnonzero(~alive[first:])
        low = np.flatnonzero(alive & (traj.n_sides[:, r] <= max_sides))
        k_gap = first + int(gaps[0]) if len(gaps) else K
        k_low = int(low[0]) if len(low) else K
        schedule[gid] = min(k_gap, k_low)
    return schedule


def paint_dead(S, dead, periodic, wall_id=None):
    """S with the cells of the grains in `dead` handed to their live neighbours, shell by shell: every dead cell
    with at least one live 4-neighbour takes the id carried by most of them, ties going to the smaller id, and the
    pass is repeated until no dead cell is left. The wall id never spreads."""
    S = np.array(S, dtype=np.int32, copy=True)
    dead = np.asarray(sorted(set(dead)), dtype=np.int32)
    if dead.size == 0:
        return S
    is_dead = np.isin(S, dead)
    for _ in range(max(S.shape)):
        if not is_dead.any():
            return S
        cand = [roll(S, 1, 0, periodic), roll(S, -1, 0, periodic), roll(S, 1, 1, periodic), roll(S, -1, 1, periodic)]
        valid = [(c >= 0) & ~np.isin(c, dead) & (c != wall_id if wall_id is not None else True) for c in cand]
        best_score = np.full(S.shape, -1)
        best_id = np.full(S.shape, np.iinfo(np.int32).max, np.int32)
        for ci, vi in zip(cand, valid):
            score = sum((vj & (cj == ci)).astype(int) for cj, vj in zip(cand, valid))
            better = vi & ((score > best_score) | ((score == best_score) & (ci < best_id)))
            best_score = np.where(better, score, best_score)
            best_id = np.where(better, ci, best_id)
        fill = is_dead & (best_score > 0)
        S[fill] = best_id[fill]
        is_dead = np.isin(S, dead)
    raise RuntimeError('paint_dead did not converge: a dead grain has no live neighbour')


def view_field(field, schedule, wall_id=None):
    """The stack with every grain painted out from its scheduled layer on. Returns (Field, painted cells per layer)."""
    ids = np.empty_like(field.ids)
    painted = np.zeros(field.nk, int)
    for k in range(field.nk):
        dead = [g for g, k0 in schedule.items() if k0 <= k]
        ids[k] = paint_dead(field.ids[k], dead, field.periodic, wall_id)
        painted[k] = int((ids[k] != field.ids[k]).sum())
    view = Field(ids=ids, cell=field.cell, dz=field.dz, periodic=field.periodic, source=field.source,
                 theta=field.theta, z0=field.z0)
    return view, painted


def track_view(field, raw=None, wall_id=None, rounds=4, verbose=False, **track_kwargs):
    """Track the GrainGNN view of `field`. Returns (view Trajectory, view Field, schedule, painted cells per layer).

    raw           the raw Trajectory of `field` if already computed (else track_stack is run first)
    track_kwargs  passed to track_stack for the view (smooth, max_jump, slice_kwargs)
    """
    raw = raw if raw is not None else track_stack(field, wall_id=wall_id, **track_kwargs)
    schedule = dead_schedule(raw)
    for _ in range(rounds):
        view, painted = view_field(field, schedule, wall_id)
        traj = track_stack(view, wall_id=wall_id, verbose=verbose, **track_kwargs)
        onsets = {}
        for ev in traj.events:
            if ev['kind'] in ('two_sided', 'enclosed'):
                onsets[int(ev['grain'])] = min(int(ev['k']), onsets.get(int(ev['grain']), traj.n_layers))
        again = {g: min(k, schedule.get(g, traj.n_layers)) for g, k in onsets.items()}
        if not again or all(schedule.get(g) == k for g, k in again.items()):
            return traj, view, schedule, painted
        schedule.update(again)
    raise RuntimeError('track_view: the painted stack still shows lenses or enclosed grains')


def schedule_kinds(raw, schedule):
    """Counts of grains by how the schedule ends them: 'onset' (lens or enclosed state), 'gap' (no cells),
    'absent' (never present), 'never' (alive to the top)."""
    kinds = dict(onset=0, gap=0, absent=0, never=0)
    for r, gid in enumerate(raw.gids.tolist()):
        alive = raw.area[:, r] > 0
        k0 = schedule[gid]
        kinds['absent' if not alive.any() else 'never' if k0 >= raw.n_layers else 'gap' if not alive[k0] else 'onset'] += 1
    return kinds
