"""Tests for extract.slice_graph (layer 1): synthetic images with a lens, a diagonal self-contact, a wrap shift and a
walled domain. Run with pytest or `python tests/test_slice_graph.py`."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from extract.slice_graph import build_slice_graph


def voronoi_image(n=160, seed=3, ngrains=40, periodic=True):
    rng = np.random.default_rng(seed); pts = rng.random((ngrains, 2)) * n
    yy, xx = np.mgrid[0:n, 0:n]
    best = np.full((n, n), np.inf); lab = np.zeros((n, n), int)
    for g, (px, py) in enumerate(pts, start=1):
        dx, dy = xx - px, yy - py
        if periodic: dx = (dx + n / 2) % n - n / 2; dy = (dy + n / 2) % n - n / 2
        d = dx * dx + dy * dy
        m = d < best; best[m] = d[m]; lab[m] = g
    return lab


def test_periodic_voronoi_is_trivalent():
    S = voronoi_image(); G = build_slice_graph(S)
    assert G.valid, G.report
    assert G.report['junctions'] == 2 * G.report['grains'], G.report
    assert set(G.report['degree_hist']) == {3}
    assert all(len(g.junctions) >= 3 for g in G.grains.values())


def test_lens_grain_is_allowed():
    S = voronoi_image()
    # carve a lens: a thin strip of a new grain along part of the boundary between two neighbouring grains
    G0 = build_slice_graph(S); E = max(G0.edges.values(), key=lambda e: e.length)       # longest boundary
    a, b = E.pair; P = E.path[len(E.path) // 4: 3 * len(E.path) // 4]                   # middle half of that boundary
    S2 = G0.cleaned.copy(); lens_id = S2.max() + 1; H, W = S2.shape
    for x, y in P:
        for dy in (-1, 0):
            for dx in (-1, 0):
                yy, xx = (int(np.floor(y)) + dy) % H, (int(np.floor(x)) + dx) % W
                if S2[yy, xx] in (a, b): S2[yy, xx] = lens_id
    G = build_slice_graph(S2)
    assert G.valid, G.report
    lens = G.grains[lens_id]
    assert lens.two_sided and lens.neighbours == {a, b}, (lens.neighbours, lens.two_sided)
    assert len(lens.junctions) == 2 and len(lens.edges) == 2
    assert G.junctions[lens.junctions[0]].triple == G.junctions[lens.junctions[1]].triple, 'both lens junctions carry the same triple'
    assert G.report['duplicate_triples'] >= 1


def test_diagonal_self_contact_is_resolved():
    S = voronoi_image().copy(); G0 = build_slice_graph(S, cleanup=False)
    J = next(iter(G0.junctions.values())); i, j = J.corners[0]; H, W = S.shape
    a, b, c = S[i, j], S[(i + 1) % H, j], S[i, (j + 1) % W]
    S[(i + 1) % H, j] = S[i, (j + 1) % W]                                           # make one grain touch itself across the diagonal
    G = build_slice_graph(S)
    assert G.valid and set(G.report['degree_hist']) == {3}, G.report


def test_wrap_shift_invariance():
    S = voronoi_image(); G = build_slice_graph(S); sx, sy = 37, 91
    G2 = build_slice_graph(np.roll(np.roll(S, sy, 0), sx, 1))
    assert G2.valid and G2.report['junctions'] == G.report['junctions'] and G2.report['edges'] == G.report['edges']
    p1 = sorted((J.triple, tuple(np.round(J.pos, 2))) for J in G.junctions.values())
    p2 = sorted((J.triple, tuple(np.round((J.pos - np.array([sx, sy])) % np.array(S.shape[::-1]), 2))) for J in G2.junctions.values())
    assert [t for t, _ in p1] == [t for t, _ in p2]
    d = np.array([np.hypot(*(((np.array(q1) - np.array(q2)) + 80) % 160 - 80)) for (_, q1), (_, q2) in zip(p1, p2)])
    assert d.max() < 1e-6, d.max()


def test_walled_domain_is_a_disk():
    S = voronoi_image(periodic=False)                                                # ids 1..40; 0 is the wall
    G = build_slice_graph(S, periodic=(False, False), wall_id=0)
    assert G.valid, G.report
    assert G.report['euler'] == 2 and set(G.report['degree_hist']) == {3}
    assert G.report['junctions'] == 2 * (G.report['grains'] - 1)                    # trivalent planar: V = 2 (F - 2)
    assert 0 not in G.grains and any(0 in g.neighbours for g in G.grains.values())
    assert G.shape == (S.shape[0] + 2, S.shape[1] + 2)                               # positions refer to the padded image
    P = np.array([J.pos for J in G.junctions.values()])
    assert (P >= 0.5 - 1e-9).all() and (P[:, 0] <= S.shape[1] + 1.5).all() and (P[:, 1] <= S.shape[0] + 1.5).all()   # wall junctions sit on the wall line


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
