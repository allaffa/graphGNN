"""Track grains and junctions through a stack of slices (layer 2 of the extractor).

track_stack builds the layer-1 network of every slice and links consecutive slices with junction_matching. The
result is a Trajectory in slot form: a junction keeps its slot for as long as it is matched, and a slot freed by a
vanished junction is handed to the junction that the same event created, preferring the one that shares two
grains with it (the GrainGNN convention of a new junction inheriting an old vertex). Grains are tracked by their
id, which is global in ExaCA and cuPF fields. Events are recorded at the first layer where the new state is seen:
  eliminate    a grain present in layer k - 1 has no cells in layer k
  two_sided    a grain has exactly two neighbours in layer k and had another number (or did not exist) in k - 1
  enclosed     a grain has one neighbour and no junctions in layer k and had not in k - 1
  switch       two grains that shared a boundary in layer k - 1 no longer do in layer k while both still exist;
               slots are the junctions at the ends of that boundary and pair_new the boundary now joining those
               two slots, which is the one that opened between the other two grains in a clean switch, or (-1, -1)
               when overlapping events in one step left no boundary between them
  appear       a grain id has cells in layer k > 0 and had none in layer k - 1
Two-sided and enclosed grains are the end states of a shrinking grain that GrainGNN cannot represent; the sample
builder treats their onset as the elimination. An `appear` after a gap is a remnant of one or two cells that the
corner cleanup erased in some layers and not in others; the sample builder keeps the first disappearance.
"""
import numpy as np
from .slice_graph import build_slice_graph
from .junction_matching import match_junctions, distance
from .trajectory import Trajectory, REPORT_KEYS, EVENT_DTYPE, smooth_tracks

NONE2 = (-1, -1)


def event(kind, k, grain=-1, pair_old=NONE2, pair_new=NONE2, slots=NONE2, n=-1):
    """One row of Trajectory.events, in the field order of EVENT_DTYPE."""
    return (kind, k, grain, pair_old, pair_new, slots, n)


class SlotTable:
    """Hands out junction slots: freed slots are reused before new ones are opened."""

    def __init__(self):
        self.n_slots = 0
        self.free = []                                     # slots freed in earlier layers, ascending
        self.next_track = 0

    def open(self):
        if self.free:
            return self.free.pop(0)
        self.n_slots += 1
        return self.n_slots - 1

    def release(self, slots):
        self.free = sorted(self.free + list(slots))

    def new_track(self):
        self.next_track += 1
        return self.next_track - 1


def place_started(started, cur, freed, slots, shape, periodic):
    """Choose a slot for every junction that has no predecessor.

    freed holds (slot, triple, pos) of the junctions that ended in this same transition. A started junction takes
    the nearest freed slot whose old triple shares two grains with its own, else the nearest freed slot, else a
    slot from the free list or a new one.
    """
    chosen = {}
    freed = list(freed)
    for jid in started:
        new = cur.junctions[jid]
        if freed:
            def rank(item):
                slot, triple, pos = item
                return (len(set(triple) & set(new.triple)) != 2, distance(pos, new.pos, shape, periodic))
            best = min(freed, key=rank)
            freed.remove(best)
            chosen[jid] = best[0]
        else:
            chosen[jid] = slots.open()
    slots.release(slot for slot, _, _ in freed)
    return chosen


def assign_slots(m, prev, cur, prev_slot, prev_track, slots, shape, periodic):
    """Slot and track of every junction of cur from the Matching m against prev: a matched junction keeps the
    slot and track of its predecessor, an ended junction frees its slot for this transition, a started junction
    is placed by place_started and gets a new track."""
    slot = {n: prev_slot[o] for o, n in m.matched.items()}
    track = {n: prev_track[o] for o, n in m.matched.items()}
    freed = [(prev_slot[o], prev.junctions[o].triple, prev.junctions[o].pos) for o in m.ended]
    placed = place_started(m.started, cur, freed, slots, shape, periodic)
    slot.update(placed)
    track.update({jid: slots.new_track() for jid in placed})
    return slot, track


def switch_events(k, prev, cur, prev_slot, slot, alive):
    """Switch events between layers k - 1 and k: boundaries of prev whose two grains are both alive in cur but no
    longer share a boundary. See the module docstring for the slots and pair_new recorded."""
    cur_pairs = {edge.pair for edge in cur.edges.values()}
    between_slots = {}
    for edge in cur.edges.values():
        between_slots.setdefault(frozenset((slot[edge.j1], slot[edge.j2])), set()).add(edge.pair)
    events = []
    for edge in prev.edges.values():
        a, b = edge.pair
        if edge.pair in cur_pairs or not (alive[a] and alive[b]):
            continue
        p, q = prev_slot[edge.j1], prev_slot[edge.j2]
        four = set(prev.junctions[edge.j1].triple) | set(prev.junctions[edge.j2].triple)
        opened = tuple(sorted(four - {a, b}))
        if opened not in between_slots.get(frozenset((p, q)), ()):
            opened = NONE2
        events.append(event('switch', k, pair_old=edge.pair, pair_new=opened, slots=(p, q)))
    return events


def grain_events(k, gids, area, n_sides):
    """Grain events at layer k from the per-layer area and side counts: eliminate, appear, two_sided, enclosed."""
    events = []
    if k > 0:
        for r in np.flatnonzero((area[k - 1] > 0) & (area[k] == 0)):
            events.append(event('eliminate', k, grain=gids[r], n=n_sides[k - 1, r]))
        for r in np.flatnonzero((area[k - 1] == 0) & (area[k] > 0)):
            events.append(event('appear', k, grain=gids[r], n=n_sides[k, r]))
    before = n_sides[k - 1] if k > 0 else np.zeros_like(n_sides[k])
    for n, kind in ((2, 'two_sided'), (1, 'enclosed')):
        for r in np.flatnonzero((n_sides[k] == n) & (before != n) & (area[k] > 0)):
            events.append(event(kind, k, grain=gids[r], n=n))
    return events


def layer_rows(cur, slot, track):
    """The junction arrays of one layer in jid order, and its edge rows in eid order, both in slot terms."""
    order = sorted(cur.junctions)
    junctions = dict(slot=np.array([slot[j] for j in order], int),
                     track=np.array([track[j] for j in order], int),
                     pos=np.array([cur.junctions[j].pos for j in order]).reshape(-1, 2),
                     pos_raw=np.array([cur.junctions[j].pos_raw for j in order]).reshape(-1, 2),
                     triple=np.array([cur.junctions[j].triple for j in order], int).reshape(-1, 3))
    edges = [(slot[e.j1], slot[e.j2], e.pair[0], e.pair[1], e.length_path)
             for e in sorted(cur.edges.values(), key=lambda e: e.eid)]
    return junctions, edges


def pack_layers(layers, edge_rows, n_slots):
    """Scatter the per-layer junction rows into (K, J, ...) arrays and concatenate the edge rows with offsets."""
    K, J = len(layers), n_slots
    pos, pos_raw = np.full((K, J, 2), np.nan), np.full((K, J, 2), np.nan)
    triple, track = np.full((K, J, 3), -1, np.int32), np.full((K, J), -1, np.int32)
    for k, layer in enumerate(layers):
        s = layer['slot']
        pos[k, s], pos_raw[k, s], triple[k, s], track[k, s] = layer['pos'], layer['pos_raw'], layer['triple'], layer['track']
    counts = [len(rows) for rows in edge_rows]
    flat = np.array([row for rows in edge_rows for row in rows], float).reshape(-1, 5)
    return dict(pos=pos, pos_raw=pos_raw, triple=triple, track=track,
                edge_offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
                edge_slots=flat[:, :2].astype(np.int32), edge_pair=flat[:, 2:4].astype(np.int32),
                edge_len_path=flat[:, 4])


def track_stack(field, wall_id=None, max_jump=None, smooth=0, verbose=False, **slice_kwargs):
    """Layer-1 network of every slice of `field`, linked into a Trajectory.

    field         extract.grain_field.Field
    wall_id       passed to build_slice_graph for non-periodic axes
    max_jump      cells; the farthest a junction may move between layers while changing its triple (None: no limit;
                  sharing two grains already confines the candidates to one grain boundary)
    smooth        window in layers for smooth_tracks (0 or 1: no smoothing)
    slice_kwargs  further arguments of build_slice_graph (cleanup, fit_len, refine)
    """
    ids, K, periodic = field.ids, field.nk, field.periodic
    gids = np.unique(ids)
    if wall_id is not None:
        gids = gids[gids != wall_id]
    row = {int(g): r for r, g in enumerate(gids)}
    G = len(gids)
    theta = np.full((G, 2), np.nan)
    for g, r in row.items():
        if g in field.theta:
            theta[r] = field.theta[g]

    area = np.zeros((K, G), np.int32)
    centroid = np.full((K, G, 2), np.nan)
    n_sides = np.zeros((K, G), np.int16)
    valid = np.zeros(K, bool)
    report = np.zeros((K, len(REPORT_KEYS)), np.int32)
    layers, edge_rows, events = [], [], []
    slots = SlotTable()
    prev, prev_slot, prev_track, shape = None, None, None, None

    for k in range(K):
        cur = build_slice_graph(ids[k], periodic, wall_id=wall_id, **slice_kwargs)
        shape = cur.shape
        valid[k] = cur.valid
        report[k] = [cur.report.get(key, 0) for key in REPORT_KEYS]
        for grain in cur.grains.values():
            r = row[grain.gid]
            area[k, r], centroid[k, r], n_sides[k, r] = grain.area, grain.centroid, len(grain.neighbours)

        if prev is None:                                                   # layer 0: one slot and track per junction
            slot = {jid: slots.open() for jid in sorted(cur.junctions)}
            track = {jid: slots.new_track() for jid in sorted(cur.junctions)}
        else:
            m = match_junctions(prev, cur, max_jump)
            slot, track = assign_slots(m, prev, cur, prev_slot, prev_track, slots, shape, periodic)
            alive = {gid: area[k, row[gid]] > 0 for gid in row}
            events += switch_events(k, prev, cur, prev_slot, slot, alive)
        events += grain_events(k, gids, area, n_sides)

        junctions, edges = layer_rows(cur, slot, track)
        layers.append(junctions)
        edge_rows.append(edges)
        prev, prev_slot, prev_track = cur, slot, track
        if verbose:
            print(f'layer {k}: {len(cur.junctions)} junctions, {len(cur.grains)} grains, '
                  f'{"valid" if cur.valid else "INVALID"}, slots {slots.n_slots}')

    events = np.array(events, dtype=EVENT_DTYPE) if events else np.zeros(0, EVENT_DTYPE)
    traj = Trajectory(shape=shape, periodic=periodic, cell=field.cell, dz=field.dz, z0=field.z0, source=field.source,
                      valid=valid, report=report, gids=gids.astype(np.int32), theta=theta, area=area,
                      centroid=centroid, n_sides=n_sides, events=events, **pack_layers(layers, edge_rows, slots.n_slots))
    if smooth and smooth > 1:
        smooth_tracks(traj, smooth)
    return traj


def main():
    import argparse
    from .exaca_reader import read_vtk, exaca_theta
    parser = argparse.ArgumentParser(description='Track an ExaCA GrainID stack into a Trajectory (.npz).')
    parser.add_argument('vtk')
    parser.add_argument('out')
    parser.add_argument('--orientations', help='ExaCA orientation csv for theta_x, theta_z')
    parser.add_argument('--smooth', type=int, default=0, help='window in layers for smooth_tracks')
    parser.add_argument('--max-jump', type=float, default=None)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    field = read_vtk(args.vtk)
    if args.orientations:
        field.theta = exaca_theta(field.ids, args.orientations)
    traj = track_stack(field, max_jump=args.max_jump, smooth=args.smooth, verbose=args.verbose)
    traj.save(args.out)
    kinds, counts = np.unique(traj.events['kind'], return_counts=True)
    print(f'{traj.n_layers} layers, {len(traj.gids)} grains, {traj.n_slots} junction slots, '
          f'valid {int(traj.valid.sum())}/{traj.n_layers}, events {dict(zip(kinds.tolist(), counts.tolist()))}')


if __name__ == '__main__':
    main()
