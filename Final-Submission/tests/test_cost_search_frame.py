import numpy as np
from Systems import Core as C
from Systems.plans import get_config
from Systems.plans.plan_8_astar.graph import Graph


def test_graph_uses_configured_rotating_frame_for_rock_prediction():
    C.configure()
    st = dict(r=C.S.copy(), v=5*C.SH, burning=False, main=False, fuel=20., t=60.)
    rp = np.array([C.S + 100*C.SH])
    rv = np.array([[3., 1., -2.]])
    rocks = (rp, rv, np.array([10.]), np.array([1.]), np.array([.03]))
    for enabled in (False, True):
        graph = Graph(st, rocks, 8.4, cfg={**get_config(8), 'frame_accel': enabled})
        np.testing.assert_allclose(graph.g, C.accel(rp, rv, frame=enabled))
