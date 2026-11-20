"""Camera dataset writer shared by the raw sensor adapter and the training scene."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import uuid

import numpy as np

from Systems.Interfaces import TrainingCaptureSource


@dataclass(frozen=True)
class CaptureSpec:
    camera: str = 'top'
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    width: int = 512
    height: int = 512
    fov_deg: float = 60.0
    exposure_ev: float = 10.0
    disable_bloom: bool = False

    def __post_init__(self):
        if self.camera not in ('top', 'bottom'):
            raise ValueError('camera must be top or bottom')
        if not all(math.isfinite(x) for x in (self.yaw_deg, self.pitch_deg, self.fov_deg, self.exposure_ev)):
            raise ValueError('camera settings must be finite')
        if not -89 <= self.yaw_deg <= 89:
            raise ValueError('yaw outside camera limits')
        lo, hi = (-50, 0) if self.camera == 'top' else (0, 50)
        if not lo <= self.pitch_deg <= hi:
            raise ValueError('pitch outside camera limits')
        if not 0 < self.fov_deg < 180 or min(self.width, self.height) < 1:
            raise ValueError('invalid FOV or image dimensions')


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class TrainingRecorder:
    """Capture one due camera per poll, skipping missed slots instead of bursting.

    Every camera has its own interval, measured in sim seconds. Capture calls may
    block; a future integrator must decide when the flight can tolerate that.
    Metadata brackets the exposure; it does not invent an exact exposure timestamp.
    """
    def __init__(self, directory, interval_s=2.0, specs=None, *, debris_labels='noisy_radar_only'):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError('interval_s must be positive and finite')
        self.directory = Path(directory) / f'capture_{uuid.uuid4().hex[:12]}'
        self.interval_s = float(interval_s)
        self.debris_labels = debris_labels
        self.specs = tuple(specs if specs is not None else (CaptureSpec(), CaptureSpec(camera='bottom')))
        if not self.specs:
            raise ValueError('at least one camera specification is required')
        self.due = [float('-inf')] * len(self.specs)
        self.next_slot = 0
        self.sequence = 0

    def poll(self, source: TrainingCaptureSource):
        now = float(source.time())
        slot = next((i for k in range(len(self.specs)) for i in [(self.next_slot + k) % len(self.specs)]
                     if now >= self.due[i]), None)
        if slot is None:
            return None
        self.due[slot] = now + self.interval_s
        self.next_slot = (slot + 1) % len(self.specs)
        spec = self.specs[slot]
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = f'{self.sequence:08d}_{spec.camera}'
        self.sequence += 1
        files = []
        try:
            source.point(spec)
            before = source.snapshot(spec)
            t0 = float(source.time())
            shot = source.capture(spec)
            t1 = float(source.time())
            after = source.snapshot(spec)
            if shot is None:
                raise RuntimeError('CaptureImageBlocking returned no image')
            planes = {c: np.asarray(shot[c], dtype=np.uint8).ravel() for c in ('R', 'G', 'B')}
            if any(p.size != spec.width * spec.height for p in planes.values()):
                raise ValueError('received image dimensions differ from requested dimensions')
            # Retain original channel buffers, even if the engine's pixel layout
            # later needs flipping/transposing during calibration.
            raw = self.directory / f'{stem}.npz'
            files.append(raw)
            np.savez_compressed(raw, **planes)
            from PIL import Image
            rgb = np.stack([planes[c].reshape(spec.height, spec.width) for c in ('R', 'G', 'B')], axis=2)
            png = self.directory / f'{stem}.png'
            files.append(png)
            Image.fromarray(rgb).save(png)
            metadata = dict(schema_version=1, camera=asdict(spec), image=png.name, raw_channels=raw.name,
                            capture_start_sim_s=t0, capture_end_sim_s=t1,
                            exposure_time_known=False, pixel_layout='row_major_assumed_unverified',
                            before=before, after=after,
                            labels=dict(debris=self.debris_labels, bounding_boxes=None, segmentation_masks=None,
                                        occlusion_known=False, radar_identity_known=False))
            path = self.directory / f'{stem}.json'
            files.append(path)
            path.write_text(json.dumps(metadata, default=_jsonable, allow_nan=False, indent=2) + '\n')
            event = dict(status='saved', metadata=path.name, camera=spec.camera, sim_s=t1)
        except Exception as exc:
            for p in files:
                p.unlink(missing_ok=True)
            event = dict(status='failed', camera=spec.camera, sim_s=now, error=f'{type(exc).__name__}: {exc}')
        # A blocked capture may overrun several slots: don't immediately retry it.
        self.due[slot] = max(self.due[slot], float(source.time()) + self.interval_s)
        with (self.directory / 'manifest.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, allow_nan=False) + '\n')
        return event
