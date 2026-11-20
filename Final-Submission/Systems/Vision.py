"""
Systems/Vision.py - camera geometry, the segmenters, and the uses of the camera.
Spec: docs/plans/VISION.md. No sim calls: images arrive through the Sim adapter (Interfaces.Sim.capture).
Checked offline against an emulated camera (tools/offline_bench.py --camera consensus); not tested in the sim.

Two plug points (Systems/Interfaces.py):
    SEGMENTERS  how rocks are found in an image: classical now, a pretrained or the team's trained model later
    CAMERAS     what the camera is used for: "consensus" fuses bearings into the tracker, "inspect" measures outlines
"""
from __future__ import annotations
import numpy as np
from Systems.Interfaces import pick

SIZE = 512                                    # pixels, square image
FOV = 60.0                                    # degrees, the sim's default
TOP_MOUNT = np.array([1.27, 0.0, 0.9]); BOTTOM_MOUNT = np.array([1.675, 0.0, -1.025])
# How the sim lays out the image: (sign of u, sign of v, rows and columns swapped). Not known until seen in the
# sim. Consensus tries all eight on every image and stores the one the rocks agree with here.
LAYOUTS = [(su, sv, swap) for swap in (False, True) for su in (1.0, -1.0) for sv in (1.0, -1.0)]
LAYOUT = (-1.0, -1.0, False)                  # the real sim's layout: u = 256 - f*y/x, v = 256 - f*z/x (camera axes, x along
#                                               the optical axis), solved from ground truth on real images 2026-11-13
SIG_PX = 1.0                                  # pixels, error of a detected rock centre. REAL images measured 2026-11-13:
#                                               median 3 px, p90 6.7 px (irregular, part-lit rocks); raise to 3 before
#                                               flying CAMERA = "consensus" in the sim. 1.0 matches the emulated test camera.
SIG_SHAPE = 0.2                               # plus this fraction of the rock's radius: outlines are not balls


def focal(fov_deg=FOV, size=SIZE):
    return size / 2 / np.tan(np.radians(fov_deg) / 2)


FOCAL = focal()                               # pixels at the default field of view


# ---------------------------------------------------------------------------------------------------------------
# segmenters (Interfaces.Segmenter)
# ---------------------------------------------------------------------------------------------------------------
class ClassicalSegmenter:
    """Bright rock on dark sky: threshold, then connected components. No model needed."""

    def __init__(self, min_area=6, most=200):
        self.min_area = min_area; self.most = most            # drops stars; caps the work on a cluttered image

    def segment(self, image, prompts=None):
        gray = image.max(axis=2).astype(float)
        from scipy import ndimage
        labels, _ = ndimage.label(gray > max(30.0, _otsu(gray)))
        if prompts is None:
            area = np.bincount(labels.ravel())[1:]
            big = np.flatnonzero(area >= self.min_area)
            return [labels == k + 1 for k in big[np.argsort(-area[big])][:self.most]]
        vv, uu = np.mgrid[:gray.shape[0], :gray.shape[1]]
        out = []
        for u, v, r in prompts:
            near = labels[(np.hypot(uu - u, vv - v) <= max(r, 3.0)) & (labels > 0)]
            out.append(labels == np.bincount(near).argmax() if near.size else None)
        return out


class PromptedSegmenter:
    """Pretrained segment-anything model through the `ultralytics` package (for example "mobile_sam.pt").
    Each radar point is given to the model as a prompt; with no prompts it segments everything.
    Untested here: needs `pip install ultralytics`."""

    def __init__(self, weights="mobile_sam.pt"):
        from ultralytics import SAM
        self.model = SAM(weights)

    def _masks(self, res):
        return [] if not res or res[0].masks is None else list(res[0].masks.data.cpu().numpy().astype(bool))

    def segment(self, image, prompts=None):
        if prompts is None:
            return self._masks(self.model.predict(image, verbose=False))
        out = []
        for u, v, _ in prompts:
            masks = self._masks(self.model.predict(image, points=[[float(u), float(v)]], labels=[1], verbose=False))
            out.append(masks[0] if masks else None)
        return out


class DetectorSegmenter:
    """An instance-segmentation network that needs no prompt: the slot for the team's custom-trained model
    (any `ultralytics` YOLO segmentation weights). A prompt takes the mask whose centre is nearest to it.
    Untested here: needs `pip install ultralytics` and trained weights."""

    def __init__(self, weights="weights/rocks-seg.pt"):
        from ultralytics import YOLO
        self.model = YOLO(weights)

    def segment(self, image, prompts=None):
        res = self.model.predict(image, verbose=False)
        masks = [] if not res or res[0].masks is None else list(res[0].masks.data.cpu().numpy().astype(bool))
        if prompts is None:
            return masks
        centres = [np.array([np.nonzero(m)[1].mean(), np.nonzero(m)[0].mean()]) for m in masks]
        out = []
        for u, v, r in prompts:
            d = [np.linalg.norm(c - (u, v)) for c in centres]
            out.append(masks[int(np.argmin(d))] if d and min(d) <= 2 * max(r, 5.0) else None)
        return out


# The registry. Models load only when chosen. To add one: a class with segment(image, prompts=None), one line here.
SEGMENTERS = {"classical": ClassicalSegmenter, "sam": PromptedSegmenter, "yolo": DetectorSegmenter}


def _otsu(gray):
    hist, edges = np.histogram(gray, bins=64); p = hist / max(hist.sum(), 1); mid = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p); m0 = np.cumsum(p * mid); mt = m0[-1]
    var = (mt * w0 - m0) ** 2 / np.maximum(w0 * (1 - w0), 1e-12)
    return mid[int(np.argmax(var))]


# ---------------------------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------------------------
def body_to_camera(yaw_deg, pitch_deg):
    """Body-to-camera rotation, as defined on the competition wiki (3-2-1 sequence, zero roll)."""
    ps, th = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.array([[np.cos(th), 0, -np.sin(th)], [0, 1, 0], [np.sin(th), 0, np.cos(th)]]) @ \
        np.array([[np.cos(ps), np.sin(ps), 0], [-np.sin(ps), np.cos(ps), 0], [0, 0, 1]])


def aim(p_body, top=None):
    """Which camera and which (yaw, pitch) look at the body-frame point p_body. Positive pitch is down."""
    def angles(mount):
        d = p_body - mount
        return np.degrees(np.arctan2(d[1], d[0])), -np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1])))
    top = angles(TOP_MOUNT)[1] <= 0 if top is None else top
    yaw, pitch = angles(TOP_MOUNT if top else BOTTOM_MOUNT)
    lo, hi = (-50.0, 0.0) if top else (0.0, 50.0)
    cy, cp = float(np.clip(yaw, -89, 89)), float(np.clip(pitch, lo, hi))
    return top, cy, cp, max(abs(cy - yaw), abs(cp - pitch))          # last value: degrees lost to the limits


def pixels(q, f, size=SIZE, layout=None):
    """Points in camera axes (k x 3, x along the optical axis) -> their pixels (k x 2, as u, v) and the
    Jacobians d(u, v)/dq (k x 2 x 3)."""
    su, sv, swap = LAYOUT if layout is None else layout
    x = q[:, 0]; a = su * f * q[:, 1] / x; b = sv * f * q[:, 2] / x; o = np.zeros_like(x)
    ja = np.column_stack([-a / x, su * f / x, o]); jb = np.column_stack([-b / x, o, sv * f / x])
    uv = size / 2 + (np.column_stack([b, a]) if swap else np.column_stack([a, b]))
    return uv, np.stack([jb, ja] if swap else [ja, jb], 1)


def project(p_body, top, yaw_deg, pitch_deg, f=FOCAL):
    """Body-frame point -> (u, v, range along the optical axis), or None if behind the camera."""
    q = body_to_camera(yaw_deg, pitch_deg) @ (p_body - (TOP_MOUNT if top else BOTTOM_MOUNT))
    if q[0] <= 1.0:
        return None
    (u, v), = pixels(q[None], f)[0]
    return u, v, q[0]


def measure(mask, u0, v0, rng, f=FOCAL):
    """Outline radius (m, across the line of sight) of the masked rock, or None if the mask is not trustworthy."""
    rows, cols = np.nonzero(mask)
    if not rows.size or rows.min() == 0 or cols.min() == 0 or rows.max() == mask.shape[0] - 1 or cols.max() == mask.shape[1] - 1:
        return None                                                  # empty, or cut off by the image border
    radius = rng * np.hypot(cols - u0, rows - v0).max() / f
    offset = rng * np.hypot(cols.mean() - u0, rows.mean() - v0) / f
    return float(radius) if offset < 0.5 * radius + 3.0 else None    # a far-off blob is a different object


def circle(mask):
    """Smallest circle around a mask, as (u, v, radius) in pixels. The sunlit limb of a ball spans its full
    diameter whatever the sun angle, so this gives the rock's centre where the centroid would sit toward the sun."""
    rows, cols = np.nonzero(mask); p = np.column_stack([cols, rows]).astype(float)
    a = b = p[0]
    for _ in range(2):                                               # two sweeps settle on a far pair: nearly a diameter
        a = p[np.argmax(((p - b) ** 2).sum(1))]; b = p[np.argmax(((p - a) ** 2).sum(1))]
    c = (a + b) / 2; rad = np.linalg.norm(b - a) / 2
    for _ in range(3):                                               # then take in whatever still sticks out
        d = np.linalg.norm(p - c, axis=1); k = int(np.argmax(d))
        if d[k] <= rad:
            break
        c = c + (p[k] - c) * (d[k] - rad) / (2 * d[k]); rad = (rad + d[k]) / 2
    return c[0], c[1], rad + 0.5                                     # pixel centres to pixel edges


# ---------------------------------------------------------------------------------------------------------------
# matching radar tracks to detected rocks
# ---------------------------------------------------------------------------------------------------------------
def consensus(pred, gate, det, pred_r=None, det_r=None, max_shift=30.0, cell=3.0):
    """Match projected radar tracks to rocks detected in the image, when both lists hold strangers (rocks the
    segmenter missed; stars, the mothership, merged blobs) and the whole image may be shifted by a pointing or
    timing error.
    pred: m x 2 pixels, gate: m pixels (3 sigma of each projection), det: n x 2 pixels, *_r: radii in pixels.
    Returns (pairs k x 2 of [pred index, det index], shift (2,) in pixels).

    1. Every pairing closer than max_shift votes for the shift it implies. Each real match votes for the true
       shift; wrong pairings scatter. The fullest cell of the vote histogram wins.
    2. Around that shift a pair is kept only if the detection is the one detection inside the projection's gate,
       and that projection the one that claims it. Ambiguous rocks are left to the radar."""
    none = np.zeros((0, 2), int), np.zeros(2)
    if not len(pred) or not len(det):
        return none
    d = det[None, :, :] - pred[:, None, :]
    fits = np.ones(d.shape[:2], bool)
    if pred_r is not None and det_r is not None:                     # the size must agree within a factor or 2 px
        ratio = det_r[None, :] / np.maximum(pred_r[:, None], 1e-6)
        fits = ((ratio > 0.5) & (ratio < 2.0)) | (np.abs(det_r[None, :] - pred_r[:, None]) < 2.0)
    votes = d[fits & (np.abs(d).max(2) < max_shift)]
    if not len(votes):
        return none
    bins = int(2 * max_shift / cell)
    hist = np.histogram2d(votes[:, 0], votes[:, 1], bins=bins, range=[[-max_shift, max_shift]] * 2)[0]
    from scipy import ndimage
    peak = np.unravel_index(np.argmax(ndimage.uniform_filter(hist, 3, mode="constant")), hist.shape)
    shift = (np.array(peak) + 0.5) * cell - max_shift
    shift = np.median(votes[np.abs(votes - shift).max(1) < 1.5 * cell], axis=0)
    for _ in range(3):
        inside = fits & (np.linalg.norm(d - shift, axis=2) < gate[:, None])
        pairs = np.argwhere(inside & (inside.sum(1) == 1)[:, None] & (inside.sum(0) == 1)[None, :])
        if not len(pairs):
            return none
        shift = np.median(d[pairs[:, 0], pairs[:, 1]], axis=0)
    return pairs, shift


# ---------------------------------------------------------------------------------------------------------------
# camera uses (Interfaces.CameraUse)
# ---------------------------------------------------------------------------------------------------------------
class Consensus:
    """One wide image of the rocks ahead, every 2 s of the coast. Each rock the radar tracks inside the view is
    projected into the image; consensus() matches projections to detected rocks; each matched rock's filter then
    takes the camera's bearing, which is finer across the line of sight than the radar's 2 m.
    Spec: docs/plans/VISION.md section 10."""

    def __init__(self, segmenter, fov=100.0, top=True, yaw=0.0, pitch=-50.0, period=2.0, near=20.0, min_px=1.5):
        self.segmenter = segmenter; self.view = (top, yaw, pitch, fov); self.period = period
        self.near = near; self.min_px = min_px                # use rocks beyond `near` m that show at least `min_px`
        self.next = 0.0; self.last = None; self.used = 0      # last: report of the latest image (for logs and tests)

    def due(self, t, coasting):
        return coasting and t >= self.next

    def __call__(self, sim, world, t, r, v):
        global LAYOUT
        self.next = t + self.period; self.last = None
        top, yaw, pitch, fov = self.view
        B = np.column_stack([sim.to_body(e) for e in np.eye(3)])     # world -> body, read before the capture blocks
        image = sim.capture(top, yaw, pitch, fov)
        if image is None or not world.n:
            return None
        det = np.array([circle(m) for m in self.segmenter.segment(image)], float).reshape(-1, 3)
        size = image.shape[0]; f = focal(fov, size)
        cam = body_to_camera(yaw, pitch); M = cam @ B
        rp, _, _, sp, _ = world.predict(t)
        q = (rp - r) @ M.T - cam @ (TOP_MOUNT if top else BOTTOM_MOUNT)      # camera axes, x along the optical axis
        x = np.maximum(q[:, 0], 1.0); pr = f * world.radius() / x
        ahead = np.flatnonzero((q[:, 0] > self.near) & (pr > self.min_px))
        sig = np.sqrt((f * sp / x) ** 2 + SIG_PX ** 2 + (SIG_SHAPE * pr) ** 2)
        tries = []
        for layout in LAYOUTS:                                       # the true image layout is the one the rocks agree with
            uv, J = pixels(q[ahead], f, size, layout)
            seen = np.abs(uv - size / 2).max(1) < size / 2 - pr[ahead]       # the whole rock is inside the image
            idx = ahead[seen]
            pairs, shift = consensus(uv[seen], 3 * sig[idx], det[:, :2], pr[idx], det[:, 2])
            tries.append((len(pairs), layout, pairs, shift, idx, uv[seen], J[seen]))
        tries.sort(key=lambda a: -a[0])
        k, layout, pairs, shift, idx, uv, J = tries[0]
        self.last = dict(t=t, detected=len(det), in_view=len(idx), matched=k, shift=shift, layout=layout, used=False)
        if k < max(5, 0.3 * min(len(idx), len(det))) or k < 2 * tries[1][0]:
            return self.last                                         # no clear agreement: the radar picture stands
        i, j = pairs[:, 0], pairs[:, 1]
        e = det[j, :2] - uv[i] - shift                               # what the camera saw minus what the filter expected
        var = SIG_PX ** 2 + (SIG_SHAPE * det[j, 2]) ** 2 + (e ** 2).sum() / (2 * k) / k     # last term: the shift is itself estimated
        world.sight(idx[i], t, J[i] @ M, e, var)
        LAYOUT = layout; self.used += 1
        self.last.update(used=True, rms_px=float(np.sqrt((e ** 2).sum() / k)))
        return self.last


class Inspect:
    """Point a camera at the next rock that will pass close and measure its outline radius, every 2 s of the
    coast. The radius replaces the blanket shape margin for that rock. Spec: docs/plans/VISION.md sections 3-6."""

    def __init__(self, segmenter, period=2.0):
        self.segmenter = segmenter; self.period = period; self.next = 0.0
        self.tried = set()                                           # one attempt per rock, successful or not

    def due(self, t, coasting):
        return coasting and t >= self.next

    def target(self, world, t, r, v):
        """Track index of the next unmeasured rock that passes close, or None."""
        rp, rv, hard, _, _ = world.predict(t)
        dr = rp - r; w = v - rv; tca = (dr * w).sum(1) / np.maximum((w * w).sum(1), 1e-9)
        miss = np.linalg.norm(dr - w * tca[:, None], axis=1)
        tried = np.zeros(world.n, bool); tried[list(self.tried)] = True
        ok = (tca > 8.0) & (tca < 60.0) & (miss < hard + 25.0) & ~tried
        return int(np.flatnonzero(ok)[np.argmin(tca[ok])]) if ok.any() else None

    def __call__(self, sim, world, t, r, v):
        self.next = t + self.period
        i = self.target(world, t, r, v) if world.n else None
        if i is None:
            return None
        self.tried.add(i)
        p_body = sim.to_body(world.predict(t)[0][i] - r)
        top, yaw, pitch, lost = aim(p_body)
        px = project(p_body, top, yaw, pitch)
        if lost > 5.0 or px is None:
            return None                                              # outside what the cameras can reach
        image = sim.capture(top, yaw, pitch, FOV)
        if image is None:
            return None
        u, v_px, rng = px
        mask = self.segmenter.segment(image, [(u, v_px, FOCAL * world.radius()[i] / rng)])[0]
        radius = None if mask is None else measure(mask, u, v_px, np.linalg.norm(p_body))
        if radius is not None:
            world.set_radius(i, radius)
        return i, radius


class ForwardXConsensus(Consensus):
    def __init__(self, segmenter):
        super().__init__(segmenter, yaw=0.0, pitch=0.0)


class ForwardYConsensus(Consensus):
    def __init__(self, segmenter):
        super().__init__(segmenter, yaw=89.0, pitch=0.0)


CAMERAS = {"forward_x": ForwardXConsensus, "forward_y": ForwardYConsensus, "off": None, "consensus": Consensus, "inspect": Inspect}     # the registry: add a use with one line here


def camera_use(camera="off", segmenter="classical"):
    """Build the camera use named `camera` around the segmenter named `segmenter`. "off" gives None."""
    if camera in (None, "off"):
        return None
    return pick(CAMERAS, camera, "camera use", pick(SEGMENTERS, segmenter, "segmenter"))
