"""Tests for layer 3: the GrainGNN view (schedule and painting) and the sample builder.
Run with pytest or `python tests/test_samples.py`."""
import sys, os, types, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
from test_stack_tracking import voronoi_image, seeds, shrinking_stack, moving_seeds_stack
from extract.grain_field import Field
from extract.slice_graph import build_slice_graph
from extract.stack_tracking import track_stack
from extract.trajectory import EVENT_DTYPE
from extract.graingnn_view import dead_schedule, paint_dead, track_view
from extract.samples import build_samples, calibrate_span, span_choices, summary, SCALING, NO_EDGE, Z_RANGE_UM

PER = (True, True)


def test_paint_dead_hands_cells_to_neighbours_only():
    S = voronoi_image(seeds()); G0 = build_slice_graph(S)
    g = max(G0.grains.values(), key=lambda x: x.area).gid
    P = paint_dead(S, [g], PER)
    assert not (P == g).any() and (P[S != g] == S[S != g]).all()
    assert set(P[S == g].tolist()) <= G0.grains[g].neighbours
    # an enclosed disk goes back to its host entirely
    yy, xx = np.mgrid[0:S.shape[0], 0:S.shape[1]]; cx, cy = G0.grains[g].centroid
    S2 = S.copy(); S2[(np.hypot(xx - cx, yy - cy) <= 3) & (S2 == g)] = 999
    assert (paint_dead(S2, [999], PER) == S).all()
    assert (paint_dead(S, [], PER) == S).all()


def test_schedule_is_lens_onset_or_first_gap():
    ids, g = shrinking_stack(layers=8)
    raw = track_stack(Field(ids=ids, cell=1.0, dz=1.0))
    r = raw.gids.tolist().index(g)
    k_low = next(k for k in range(raw.n_layers) if raw.area[k, r] > 0 and raw.n_sides[k, r] <= 2)
    sched = dead_schedule(raw)
    assert sched[g] == k_low and all(sched[x] == raw.n_layers for x in raw.gids.tolist() if x != g)
    # a grain missing from one middle layer only is dead from that layer on
    other = max((x for x in raw.gids.tolist() if x != g), key=lambda x: raw.area[0, raw.gids.tolist().index(x)])
    ids2 = ids.copy(); ids2[3] = paint_dead(ids2[3], [other], PER)
    sched2 = dead_schedule(track_stack(Field(ids=ids2, cell=1.0, dz=1.0)))
    assert sched2[other] == 3


def test_view_shows_the_elimination_at_the_onset():
    ids, g = shrinking_stack(layers=8); field = Field(ids=ids, cell=1.0, dz=1.0)
    raw = track_stack(field)
    traj, vfield, sched, painted = track_view(field, raw=raw)
    k0 = sched[g]
    assert traj.valid.all()
    assert len(traj.events_of('two_sided')) == 0 and len(traj.events_of('enclosed')) == 0
    elim = traj.events_of('eliminate')
    assert len(elim) == 1 and elim[0]['grain'] == g and elim[0]['k'] == k0 and elim[0]['n'] >= 3
    has_cells = (ids == g).reshape(len(ids), -1).any(1)
    assert ((painted > 0) == (has_cells & (np.arange(len(ids)) >= k0))).all()
    assert painted.tolist() == (vfield.ids != ids).reshape(len(ids), -1).sum(1).tolist()
    r = traj.gids.tolist().index(g)
    assert (traj.area[:, r] > 0).tolist() == [k < k0 for k in range(traj.n_layers)]
    for k in range(traj.n_layers):                                   # torus with no lens: junctions = 2 * grains
        assert traj.occupied()[k].sum() == 2 * traj.alive()[k].sum()
        assert traj.area[k].sum() == ids[k].size


def test_static_samples():
    S = voronoi_image(seeds()); ids = np.stack([S] * 6)
    traj = track_stack(Field(ids=ids, cell=0.5, dz=0.25))
    samples = build_samples(traj, G=5.0, R=1.0, span=2)
    assert len(samples) == 4 and [s.physical_params['height'] for s in samples] == [0, 1, 2, 3]
    s = samples[0]; nG, J = len(traj.gids), traj.n_slots
    assert s.features == {'grain': ['x', 'y', 'z', 'area', 'cosx', 'sinx', 'cosz', 'sinz', 'span', 'darea'],
                          'joint': ['x', 'y', 'z', 'G', 'R', 'span', 'dx', 'dy']}
    assert s.feature_dicts['grain'].shape == (nG, 10) and s.feature_dicts['joint'].shape == (J, 8)
    assert s.target_dicts['grain'].shape == (nG, 1) and s.target_dicts['joint'].shape == (J, 2)
    assert np.allclose(s.target_dicts['grain'], 0) and np.allclose(s.target_dicts['joint'], 0)
    assert (s.mask['grain'] == 1).all() and (s.mask['joint'] == 1).all() and (s.mask['edge'] == 1).all()
    assert (s.target_dicts['edge_event'] == 0).all() and (s.target_dicts['grain_event'] == 0).all()
    assert np.allclose(s.target_dicts['edge'], 0)
    assert np.isclose(s.feature_dicts['grain'][:, 3].sum(), 1.0)
    x = s.feature_dicts['joint'][:, :2]; assert (x >= 0).all() and (x < 1).all()
    assert np.allclose(s.feature_dicts['joint'][:, 3], 0.5) and np.allclose(s.feature_dicts['joint'][:, 4], 0.5)
    assert np.allclose(s.feature_dicts['joint'][:, 5], 2 * 0.25 / Z_RANGE_UM) and np.allclose(s.feature_dicts['grain'][:, 8], 2 * 0.25 / Z_RANGE_UM)
    assert np.allclose(samples[1].feature_dicts['joint'][:, 2], 0.25 / Z_RANGE_UM)
    assert (s.feature_dicts['grain'][:, 4:8] == 0).all(), 'unknown orientation -> zeros'
    E = s.edge_index_dicts[('joint', 'connect', 'joint')]
    assert E.shape == (2, 3 * J) and s.edge_index_dicts[('grain', 'push', 'joint')].shape == (2, 3 * J)
    assert (s.edge_index_dicts[('grain', 'push', 'joint')][::-1] == s.edge_index_dicts[('joint', 'pull', 'grain')]).all()
    assert all(len(t) == 3 and all(1 <= g <= nG for g in t) for t in s.vertex2joint.values()) and len(s.vertex2joint) == J
    w = s.edge_weight_dicts[('joint', 'connect', 'joint')]; assert w.shape == (3 * J, 1) and (w > 0).all() and (w < 0.5).all()
    assert np.allclose(samples[2].feature_dicts['joint'][:, 6:8], 0) and np.allclose(samples[2].feature_dicts['grain'][:, 9], 0)
    assert len(s.target_dicts['edge']) == 3 * J and len(s.mask['edge']) == 3 * J


def test_drifting_samples_have_exact_labels_and_history():
    S = voronoi_image(seeds()); n = S.shape[0]
    ids = np.stack([np.roll(np.roll(S, k, 0), 2 * k, 1) for k in range(8)])
    traj = track_stack(Field(ids=ids, cell=1.0, dz=1.0))
    span = 3
    samples = build_samples(traj, 1.0, 1.0, span=span)
    assert [s.physical_params['height'] for s in samples] == [0, 1, 2, 3, 4]
    for s in samples:
        assert (s.mask['joint'] == 1).all() and (s.target_dicts['edge_event'] == 0).all()
        assert np.allclose(s.target_dicts['joint'][:, 0], SCALING['joint'] * 2 * span / n)
        assert np.allclose(s.target_dicts['joint'][:, 1], SCALING['joint'] * span / n)
        assert np.allclose(s.target_dicts['edge'], 0, atol=1e-9) and np.allclose(s.target_dicts['grain'], 0)
        h = s.feature_dicts['joint'][:, 6:8]
        if s.physical_params['height'] >= span:
            assert np.allclose(h[:, 0], SCALING['joint'] * 2 * span / n) and np.allclose(h[:, 1], SCALING['joint'] * span / n)
        else:
            assert np.allclose(h, 0)


def test_moving_seeds_samples_masks_and_events():
    ids = moving_seeds_stack(speed=0.5); n = ids.shape[1]
    traj = track_stack(Field(ids=ids, cell=1.0, dz=1.0))
    span = 3
    samples = build_samples(traj, 1.0, 1.0, span=span)
    n_switch = 0
    for s in samples:
        k = s.physical_params['height']; m = k + span
        mask = s.mask['joint'][:, 0].astype(bool)
        same = (traj.track[k] >= 0) & (traj.track[k] == traj.track[m]) & np.all(traj.triple[k] == traj.triple[m], axis=1)
        assert (mask == same).all()
        ev = s.target_dicts['edge_event']; E = s.edge_index_dicts[('joint', 'connect', 'joint')]
        assert (s.mask['edge'] == (ev >= 0)).all() and (ev[:len(ev) // 2] == ev[len(ev) // 2:]).all()
        joined = {frozenset(p) for p in traj.edge_slots[traj.edges_of(m)].tolist()}
        for i in range(len(ev)):
            a, b = int(E[0, i]), int(E[1, i])
            kept = traj.track[k, a] == traj.track[m, a] and traj.track[k, b] == traj.track[m, b] and frozenset((a, b)) in joined
            assert (ev[i] >= 0) == kept
        n_switch += int((ev == 1).sum()) // 2
        gone = (traj.track[k] >= 0) & (traj.track[m] != traj.track[k])
        assert np.allclose(s.target_dicts['joint'][gone], -SCALING['joint'] * traj.pos[k, gone] / n)
    assert n_switch > 0
    # at span 1 every resolved layer-2 switch is a switching edge of the sample
    by_layer = {}
    for s in build_samples(traj, 1.0, 1.0, span=1, stride=1):
        ev = s.target_dicts['edge_event']; E = s.edge_index_dicts[('joint', 'connect', 'joint')]
        by_layer[s.physical_params['height'] + 1] = {frozenset((int(E[0, i]), int(E[1, i]))) for i in np.flatnonzero(ev == 1)}
    resolved = [e for e in traj.events_of('switch') if e['pair_new'][0] >= 0]
    assert resolved and all(frozenset(e['slots'].tolist()) in by_layer[int(e['k'])] for e in resolved)


def test_calibrate_span_follows_the_legacy_rule():
    def fake(n_switch, n_elim, n_layers):
        ev = np.zeros(n_switch + n_elim, EVENT_DTYPE); ev['kind'][:n_switch] = 'switch'; ev['kind'][n_switch:] = 'eliminate'
        return types.SimpleNamespace(events=ev, n_layers=n_layers)
    choices = span_choices(0.4, 121)
    assert choices == [6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]
    assert calibrate_span(fake(134, 75, 121), choices) == 6           # cuPF seed 10020: the paper's span
    assert calibrate_span(fake(30, 10, 121), choices) == 30
    assert calibrate_span(fake(0, 0, 121), choices) == 120
    assert span_choices(0.25, 200) == [10, 13, 16, 19, 24, 32, 38, 48, 64, 96, 192]


def test_samples_feed_the_model():
    try:
        import torch
        from data_loader import DynamicHeteroGraphTemporalSignal
        from models import GrainNN_regressor
        from parameters import regressor
    except ImportError as err:
        print('skipped (', err, ')'); return
    ids = moving_seeds_stack(speed=0.5)
    traj = track_stack(Field(ids=ids, cell=1.0, dz=1.0))
    samples = build_samples(traj, 2.0, 1.0, span=3)
    ds = DynamicHeteroGraphTemporalSignal(samples); b = ds[0]
    hp = regressor(0); hp.features, hp.targets, hp.device, hp.metadata = samples[0].features, samples[0].targets, 'cpu', b.metadata()
    model = GrainNN_regressor(hp, False, True); model.eval()
    with torch.no_grad():
        out = model(b.x_dict, b.edge_index_dict, b.edge_attr_dict)
    J, nG = traj.n_slots, len(traj.gids)
    assert out['joint'].shape == (J, 2) and out['grain'].shape[0] == nG and out['edge'].shape == (b['mask']['edge'].shape[0],)
    loss = torch.mean(b['mask']['joint'] * (b.y_dict['joint'] - out['joint']) ** 2) + torch.mean(b['mask']['edge'] * (b.y_dict['edge'] - out['edge']) ** 2)
    assert torch.isfinite(loss)
    print('   ', summary(samples, traj).replace('\n', '\n    '))


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
