"""Tests for extract.stack_tracking (layer 2): synthetic stacks with a steady drift, a shrinking grain and moving
Voronoi seeds. Run with pytest or `python tests/test_stack_tracking.py`."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from extract.grain_field import Field
from extract.slice_graph import build_slice_graph
from extract.stack_tracking import track_stack
from extract.trajectory import Trajectory, smooth_tracks
from extract.periodic_grid import wrap_delta


def voronoi_image(pts, n=120):
    yy, xx = np.mgrid[0:n, 0:n]
    best = np.full((n, n), np.inf); lab = np.zeros((n, n), int)
    for g, (px, py) in enumerate(pts, start=1):
        dx = (xx - px + n / 2) % n - n / 2; dy = (yy - py + n / 2) % n - n / 2
        d = dx * dx + dy * dy; m = d < best; best[m] = d[m]; lab[m] = g
    return lab


def seeds(n=120, ngrains=30, seed=5):
    return np.random.default_rng(seed).random((ngrains, 2)) * n


def check_against_slices(traj, ids):
    """The tracker must reproduce every slice: same triples alive in each layer, same edge count."""
    for k in range(traj.n_layers):
        G = build_slice_graph(ids[k])
        want = sorted(tuple(j.triple) for j in G.junctions.values())
        have = sorted(tuple(t) for t in traj.triple[k][traj.track[k] >= 0])
        assert want == have, k
        assert traj.edge_offsets[k + 1] - traj.edge_offsets[k] == len(G.edges)
        assert np.isfinite(traj.pos[k][traj.track[k] >= 0]).all()
        assert (traj.area[k] > 0).sum() == len(G.grains)


def test_static_stack_has_constant_tracks_and_no_events():
    S = voronoi_image(seeds()); ids = np.stack([S] * 4)
    traj = track_stack(Field(ids=ids, cell=1.0, dz=1.0))
    check_against_slices(traj, ids)
    assert len(traj.events) == 0
    assert traj.n_slots == 2 * len(np.unique(S))
    assert (traj.track == traj.track[0]).all() and (traj.triple == traj.triple[0]).all()
    assert np.allclose(traj.pos, traj.pos[0])


def test_drifting_stack_tracks_every_junction_and_smoothing_is_exact():
    S = voronoi_image(seeds()); n = S.shape[0]
    ids = np.stack([np.roll(np.roll(S, k, 0), 2 * k, 1) for k in range(6)])      # each layer moves by (2, 1) cells
    traj = track_stack(Field(ids=ids, cell=1.0, dz=1.0), smooth=5)
    check_against_slices(traj, ids)
    assert len(traj.events) == 0 and (traj.track >= 0).all()
    step = np.diff(traj.pos, axis=0)
    assert np.allclose(wrap_delta(step[..., 0], n, True), 2) and np.allclose(wrap_delta(step[..., 1], n, True), 1)
    assert traj.smooth_window == 5
    d = traj.pos_smooth - traj.pos
    assert np.allclose(wrap_delta(d[..., 0], n, True), 0, atol=1e-8) and np.allclose(wrap_delta(d[..., 1], n, True), 0, atol=1e-8)


def shrinking_stack(layers=8):
    """A Voronoi image in which one grain is eroded towards its seed until it is gone in the last layer."""
    pts = seeds(); S = voronoi_image(pts); n = S.shape[0]
    g = int(np.argmax(np.bincount(S.ravel())))                                   # the largest grain
    without = voronoi_image(np.delete(pts, g - 1, 0)); without[without >= g] += 1  # same ids, g absent
    yy, xx = np.mgrid[0:n, 0:n]; px, py = pts[g - 1]
    r = np.hypot((xx - px + n / 2) % n - n / 2, (yy - py + n / 2) % n - n / 2)
    rmax = r[S == g].max()
    stack = []
    for k in range(layers):
        keep = (S == g) & (r <= rmax * (1 - k / (layers - 1)) - 0.5)
        stack.append(np.where(S == g, np.where(keep, S, without), S))
    return np.stack(stack), g


def test_shrinking_grain_is_eliminated_and_slots_are_reused():
    ids, g = shrinking_stack()
    traj = track_stack(Field(ids=ids, cell=1.0, dz=1.0))
    check_against_slices(traj, ids)
    assert traj.valid.all()
    elim = traj.events_of('eliminate')
    assert len(elim) == 1 and elim[0]['grain'] == g and elim[0]['k'] == traj.n_layers - 1
    assert elim[0]['n'] == traj.n_sides[-2, list(traj.gids).index(g)] >= 1
    onsets = [(ev['kind'], int(ev['k'])) for ev in traj.events if ev['kind'] in ('two_sided', 'enclosed')]
    assert onsets, 'the eroded disk must pass through a two-sided or enclosed state before it vanishes'
    assert all(ev['grain'] == g for ev in traj.events if ev['kind'] in ('two_sided', 'enclosed'))
    assert (traj.area[:, list(traj.gids).index(g)] > 0).tolist() == [True] * (traj.n_layers - 1) + [False]
    assert len(traj.events_of('appear')) == 0
    for k in range(traj.n_layers):                           # torus: junctions = 2 * polygons (enclosed grains are none)
        assert (traj.track[k] >= 0).sum() == 2 * (traj.n_sides[k] >= 2).sum()
    assert traj.n_slots == (traj.track[0] >= 0).sum(), 'no slot was opened beyond the first layer'
    for ev in traj.events_of('switch'):                                          # every switch names real pairs
        k = ev['k']
        pairs_after = {tuple(p) for p in traj.edge_pair[traj.edges_of(k)]}
        pairs_before = {tuple(p) for p in traj.edge_pair[traj.edges_of(k - 1)]}
        assert tuple(ev['pair_old']) in pairs_before and tuple(ev['pair_old']) not in pairs_after
        assert ev['pair_new'][0] < 0 or tuple(ev['pair_new']) in pairs_after
        assert (traj.track[k - 1, ev['slots']] >= 0).all(), 'the slots name the junctions at the ends of the lost boundary'


def moving_seeds_stack(speed, layers=10, ngrains=24, seed=11, n=120):
    pts = seeds(ngrains=ngrains, seed=seed); vel = np.random.default_rng(1).normal(size=pts.shape) * speed
    return np.stack([voronoi_image((pts + k * vel) % n, n) for k in range(layers)])


def lost_pairs(traj):
    lost = 0
    for k in range(1, traj.n_layers):
        before = {tuple(p) for p in traj.edge_pair[traj.edges_of(k - 1)]}
        after = {tuple(p) for p in traj.edge_pair[traj.edges_of(k)]}
        lost += len(before - after)
    return lost


def test_moving_seeds_switches_match_lost_boundaries():
    ids = moving_seeds_stack(speed=0.5)
    traj = track_stack(Field(ids=ids, cell=1.0, dz=1.0))
    check_against_slices(traj, ids)
    assert len(traj.events_of('eliminate')) == 0 and len(traj.events_of('appear')) == 0
    switches = traj.events_of('switch')
    assert lost_pairs(traj) > 0 and len(switches) == lost_pairs(traj)
    assert (traj.track >= 0).all(), 'every junction is carried through the switches'
    resolved = switches[switches['pair_new'][:, 0] >= 0]
    assert len(resolved) >= len(switches) // 2, 'most switches are clean: one boundary closes, one opens'
    for ev in resolved:                                      # the two slots trade one grain each and share the new pair
        k, (p, q) = ev['k'], ev['slots']
        four = set(traj.triple[k - 1, p]) | set(traj.triple[k - 1, q])
        assert set(ev['pair_new']) == four - set(ev['pair_old'])
        for slot in (p, q):
            assert len(set(traj.triple[k - 1, slot]) & set(traj.triple[k, slot])) == 2
            assert set(ev['pair_new']) <= set(traj.triple[k, slot])


def test_fast_seeds_overlapping_switches_keep_identities():
    ids = moving_seeds_stack(speed=1.5)
    traj = track_stack(Field(ids=ids, cell=1.0, dz=1.0))
    check_against_slices(traj, ids)
    switches = traj.events_of('switch')
    assert len(switches) == lost_pairs(traj)
    assert (traj.track >= 0).all()
    assert (switches['pair_new'] < 0).any(), 'fast seeds: some switches overlap within one layer step'


def test_save_load_roundtrip(tmp_path=None):
    ids, _ = shrinking_stack(layers=4)
    traj = track_stack(Field(ids=ids, cell=0.5, dz=0.25, z0=2.0, source='synthetic'), smooth=3)
    path = os.path.join(tmp_path or os.environ.get('TMPDIR', '/tmp'), 'traj_test.npz')
    traj.save(path); back = Trajectory.load(path)
    for name in ('area', 'triple', 'track', 'edge_slots', 'edge_pair', 'valid'):
        assert np.array_equal(getattr(traj, name), getattr(back, name)), name
    assert np.allclose(traj.pos, back.pos, equal_nan=True) and np.allclose(traj.pos_smooth, back.pos_smooth, equal_nan=True)
    assert back.events.dtype == traj.events.dtype and len(back.events) == len(traj.events)
    assert back.cell == 0.5 and back.dz == 0.25 and back.z0 == 2.0 and back.smooth_window == 3
    assert np.allclose(back.heights(), 2.0 + 0.25 * np.arange(4))
    os.remove(path)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
