import json

import numpy as np
import pytest
from PIL import Image

from Systems.TrainingCapture import CaptureSpec, TrainingRecorder


class Source:
    def __init__(self):
        self.t = 0.
        self.calls = []
        self.fail = False
    def time(self): return self.t
    def point(self, spec): self.calls.append(spec.camera)
    def snapshot(self, spec): return {'sim_s': self.t, 'radar_raw_3x3': np.zeros((1, 3, 3)), 'target_position': np.array([1., 2., 3.])}
    def capture(self, spec):
        self.t += .25
        if self.fail: return None
        return {c: np.arange(spec.width * spec.height, dtype=np.uint8) + i for i, c in enumerate('RGB')}


def test_collector_keeps_non_square_raw_buffers_and_bracketed_metadata(tmp_path):
    source = Source()
    recorder = TrainingRecorder(tmp_path, interval_s=2, specs=[CaptureSpec(width=5, height=3)])
    assert not recorder.directory.exists()
    event = recorder.poll(source)
    assert event['status'] == 'saved'
    meta = json.loads((recorder.directory / event['metadata']).read_text())
    assert Image.open(recorder.directory / meta['image']).size == (5, 3)
    assert np.array_equal(np.load(recorder.directory / meta['raw_channels'])['R'], np.arange(15))
    assert meta['capture_start_sim_s'] == 0 and meta['capture_end_sim_s'] == .25
    assert meta['labels']['debris'] == 'noisy_radar_only'
    assert meta['labels']['bounding_boxes'] is None
    assert meta['before']['target_position'] == [1., 2., 3.]
    assert recorder.poll(source) is None
    source.t = 100
    recorder.poll(source)
    assert recorder.poll(source) is None   # no catch-up burst after long gaps


def test_camera_fairness_and_failed_capture_are_logged_without_partial_images(tmp_path):
    source = Source()
    recorder = TrainingRecorder(tmp_path)
    recorder.poll(source)
    source.fail = True
    assert recorder.poll(source)['status'] == 'failed'
    assert source.calls == ['top', 'bottom']
    assert len(list(recorder.directory.glob('*.png'))) == 1
    events = [json.loads(x) for x in (recorder.directory / 'manifest.jsonl').read_text().splitlines()]
    assert [e['status'] for e in events] == ['saved', 'failed']
    assert recorder.poll(source) is None


def test_camera_limits_are_validated():
    for kw in ({'camera': 'left'}, {'pitch_deg': 1}, {'camera': 'bottom', 'pitch_deg': -1},
               {'fov_deg': 0}, {'width': 0}, {'yaw_deg': float('nan')}):
        with pytest.raises(ValueError): CaptureSpec(**kw)


def test_engine_adapter_uses_upstream_pointing_helper_and_requested_capture_settings(monkeypatch):
    from types import SimpleNamespace
    from Systems import BasicController as BC
    seen = {}
    class Params:
        disable_bloom = False
    def capture(camera, params):
        seen.update(camera=camera, width=params.ResolutionX, height=params.ResolutionY,
                    fov=params.FOV, ev=params.nonphys_EV, bloom_disabled=params.disable_bloom)
        return SimpleNamespace(as_RGB8=lambda: SimpleNamespace(PixelsR=[1]*15, PixelsG=[2]*15, PixelsB=[3]*15))
    monkeypatch.setattr(BC, 'st', SimpleNamespace(CaptureImageProperties=Params,
                                                camera=SimpleNamespace(CaptureImageBlocking=capture)))
    source = BC.SpaceTeamsTrainingSource.__new__(BC.SpaceTeamsTrainingSource)
    source.api = SimpleNamespace(topCamera='top_entity', bottomCamera='bottom_entity',
                                 PanTiltCamera=lambda *args: seen.update(point=args))
    spec = CaptureSpec(camera='bottom', pitch_deg=20, width=5, height=3, exposure_ev=11, disable_bloom=True)
    source.point(spec)
    planes = source.capture(spec)
    assert seen == dict(point=('bottom_entity', 0., 20), camera='bottom_entity', width=5, height=3,
                        fov=60., ev=11, bloom_disabled=True)
    assert planes['R'].shape == (15,)
