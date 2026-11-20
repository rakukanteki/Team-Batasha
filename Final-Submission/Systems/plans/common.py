"""
Systems/plans/common.py - shared planning helpers across Plans 1..9.
"""
from __future__ import annotations
import json
import os
import re
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
from Systems import Core as C


def _coerce_types(raw: dict, defaults: dict) -> dict:
    out = {}
    for k, v in raw.items():
        if isinstance(defaults.get(k), tuple) and isinstance(v, list):
            out[k] = tuple(v)
        else:
            out[k] = v
    return out


def extract_plan_overrides(raw: dict, pname: str, defaults: dict | None = None) -> dict:
    """Resolve either flat overrides or { '*', '<plan_name>' } parameterized JSON into a plan dict."""
    if not isinstance(raw, dict):
        return {}
    merged: dict = {}
    has_scoped = any(isinstance(v, dict) for v in raw.values())
    if has_scoped:
        if isinstance(raw.get("*"), dict):
            merged.update(raw["*"])
        if isinstance(raw.get(pname), dict):
            merged.update(raw[pname])
    else:
        merged.update(raw)
    return _coerce_types(merged, defaults or {})


def read_config_json(path: Path) -> dict:
    """config.json is a git symlink to configs/<variant>.json. A Windows checkout without symlink support writes it
    as a text file holding only that relative path: follow it instead of failing to parse it."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text.startswith("{"):
        text = (Path(path).parent / text).read_text(encoding="utf-8")
    return json.loads(text)


def load_plan_config(config_file: str, defaults: dict, extra_override: dict | str | Path | None = None) -> dict:
    """Load plan config with precedence:
    1) defaults (config.py fallback)
    2) <plan_dir>/config.json (follows symlink e.g. config.json -> configs/<variant>.json)
    3) Parameterized JSON (Systems/plans/parameterized.json, BATASHA_CONFIG_JSON env, or extra_override)
    """
    cfg = dict(defaults)
    plan_dir = Path(config_file).resolve().parent
    pname = str(cfg.get("name", plan_dir.name))

    # 2. Per-plan config.json (resolving symlink transparently)
    cfg_json = plan_dir / "config.json"
    if cfg_json.is_file():
        data = read_config_json(cfg_json)
        cfg.update(extract_plan_overrides(data, pname, defaults))

    # 3a. Optional workspace parameterized.json
    param_file = Path(__file__).resolve().parent / "parameterized.json"
    if param_file.is_file():
        data = json.loads(param_file.read_text(encoding="utf-8"))
        cfg.update(extract_plan_overrides(data, pname, defaults))

    # 3b. Env var BATASHA_CONFIG_JSON
    env_path = os.environ.get("BATASHA_CONFIG_JSON", "").strip()
    if env_path and Path(env_path).is_file():
        data = json.loads(Path(env_path).read_text(encoding="utf-8"))
        cfg.update(extract_plan_overrides(data, pname, defaults))

    # 3c. Explicit runtime override (dict or path)
    if extra_override is not None:
        if isinstance(extra_override, (str, Path)):
            data = json.loads(Path(extra_override).read_text(encoding="utf-8"))
        else:
            data = extra_override
        cfg.update(extract_plan_overrides(data, pname, defaults))

    return cfg


def active_variant_info(plan_dir: Path) -> dict:
    """Inspect <plan_dir>/config.json symlink target and available presets under configs/."""
    cfg_json = plan_dir / "config.json"
    configs_dir = plan_dir / "configs"
    target = None
    is_symlink = cfg_json.is_symlink()
    pointed = cfg_json.resolve() if cfg_json.exists() else None
    if pointed is not None and not is_symlink:               # a symlink written as text (Windows): follow it
        text = cfg_json.read_text(encoding="utf-8").strip()
        if not text.startswith("{"):
            pointed = (plan_dir / text).resolve()
    if pointed is not None:
        resolved = pointed
        try:
            target = str(resolved.relative_to(plan_dir))
        except ValueError:
            target = str(resolved)
    variants = []
    if configs_dir.is_dir():
        for p in sorted(configs_dir.glob("*.json")):
            variants.append({
                "name": p.stem,
                "path": str(p.relative_to(plan_dir)),
                "active": (pointed is not None and pointed == p.resolve()),
            })
    return {
        "config_json": str(cfg_json),
        "is_symlink": is_symlink,
        "target": target,
        "variants": variants,
    }


def save_variant(plan_dir: Path, variant_name: str, cfg: dict, activate: bool = False, overwrite: bool = True) -> dict:
    """Save cfg JSON under <plan_dir>/configs/<variant_name>.json and optionally symlink config.json -> it."""
    safe = _variant_name(variant_name)
    configs_dir = plan_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    out_path = configs_dir / f"{safe}.json"
    serializable = {k: (list(v) if isinstance(v, tuple) else float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v) for k, v in cfg.items()}
    contents = json.dumps(serializable, indent=2, allow_nan=False) + "\n"
    with TemporaryDirectory(dir=configs_dir) as temp:
        staged = Path(temp) / "preset.json"
        staged.write_text(contents, encoding="utf-8")
        if overwrite:
            staged.replace(out_path)
        else:
            os.link(staged, out_path)
    if activate:
        activate_variant(plan_dir, safe)
    return {"variant": safe, "path": str(out_path), **active_variant_info(plan_dir)}


def _variant_name(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError("Use 1–64 letters, digits, hyphens or underscores for a preset name.")
    return name


def activate_variant(plan_dir: Path, variant_name: str) -> dict:
    """Point <plan_dir>/config.json symlink to configs/<variant_name>.json."""
    target_rel = Path("configs") / f"{_variant_name(variant_name)}.json"
    target_abs = plan_dir / target_rel
    if not target_abs.is_file():
        raise FileNotFoundError(f"Variant not found: {target_abs}")
    data = json.loads(target_abs.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Preset must contain a JSON object.")
    with TemporaryDirectory(dir=plan_dir) as temp:
        staged = Path(temp) / "config.json"
        try:
            staged.symlink_to(target_rel)
        except OSError:                      # Windows without symlink privilege: the same pointer as a text file
            staged.write_text(target_rel.as_posix(), encoding="utf-8")
        staged.replace(plan_dir / "config.json")
    return {"variant": variant_name, "config": data, **active_variant_info(plan_dir)}


def _one(plan, st, rocks, action, cfg=None):
    ev = C.score_plans([plan], st, rocks, cfg=cfg)
    return ev, C.info([plan], ev, 0, action)


def _best(plans, ev, action, extra=None):
    c = int(np.argmin(ev['cost']))
    info = C.info(plans, ev, c, action)
    if extra is not None and extra[c]:
        info.update(extra[c])
    return plans[c], info


def _insert(nodes, W):
    """Add an intermediate point, keeping along-track order and the aim point last."""
    return sorted(list(nodes[:-1]) + [W], key=lambda p: (p - C.S) @ C.SH) + [nodes[-1]]


def _beside(ev, c, rocks, st, top=None, cfg=None):
    """Eight points around each of the worst `top` blockers of candidate c, at the moment the path passes it."""
    if top is None:
        top = cfg.get("top_blockers", 2) if cfg else 2
    p_hit_min = cfg.get("p_hit_min", 1e-4) if cfg else 1e-4
    min_along = cfg.get("min_along_ahead", 60.0) if cfg else 60.0   # v5 NODE_AHEAD: the guard covers closer
    sig_mult = cfg.get("safety_sigma", 3.0) if cfg else 3.0
    margin_m = cfg.get("safety_margin_m", 3.0) if cfg else 3.0
    dt_plan = cfg.get("dt_plan", C.DT_PLAN) if cfg else C.DT_PLAN

    rp, rv, hard, sp, sv = rocks
    ph = ev['ph'][c]
    out = []
    for i in np.argsort(-ph)[:top]:
        if ph[i] < p_hit_min:
            break
        tau = ev['kmin'][c, i] * dt_plan
        pc = rp[i] + rv[i] * tau + 0.5 * C.accel(rp[i], rv[i]) * tau * tau
        if (pc - st['r']) @ C.SH < min_along:
            continue
        off = hard[i] + sig_mult * np.sqrt(sp[i] ** 2 + (tau * sv[i]) ** 2) + margin_m
        out += [pc + off * d for d in C.DIRS]
    return out


def _early(W, A, r):
    """The same pass beside the rock, but turning halfway there: the line aim -> W extended back toward us.
    A turn made while the main burn is still running costs almost nothing."""
    s0 = (r - C.S) @ C.SH
    sw = (W - C.S) @ C.SH
    sa = (A - C.S) @ C.SH
    return W + (W - A) * (0.5 * (sw - s0)) / max(sa - sw, 1.0)


def _settle(plan, cur, new, ni, committed, simplify_margin=None, cfg=None):
    """Foundation section 8: when a freshly computed plan may replace the one being flown."""
    p_safe = cfg.get("p_safe", C.P_SAFE) if cfg else C.P_SAFE
    sw = cfg.get("switch_margin", C.SWITCH) if cfg else C.SWITCH
    if simplify_margin is None:
        simplify_margin = cfg.get("simplify_margin", 0.5) if cfg else 0.5

    if committed and cur['ptot'] <= (cfg.get("keep_band", 3.0) if cfg else 3.0) * p_safe:   # still safe (v5 hysteresis)
        if ni['ptot'] <= p_safe and ni['cost'] < cur['cost'] - simplify_margin:
            ni['action'] = 'simplify'
            return new, ni
        return plan, cur
    if committed:                                       # no longer safe: take the repair if it helps
        return (new, ni) if (ni['ptot'] <= p_safe or ni['cost'] < cur['cost']) else (plan, cur)
    if ni['cost'] < cur['cost'] - sw or (ni['ptot'] <= p_safe < cur['ptot']):
        return new, ni
    return plan, cur


def _reference(st, base_plan, cfg=None):
    """Fly base_plan once: along-track position, time, speed and burnout step of that flight."""
    dt_plan = cfg.get("dt_plan", C.DT_PLAN) if cfg else C.DT_PLAN
    tmax = cfg.get("tmax", 420.0) if cfg else 420.0
    nodes, nn, v = C.pack([base_plan])
    path, spd, arr, _ = C.fly(st['r'], st['v'], st['burning'], st['main'], nodes, nn, v, dt=dt_plan, tmax=tmax)
    ka = arr[0] if arr[0] >= 0 else len(path) - 1
    return path[:ka + 1, 0], (path[:ka + 1, 0] - C.S) @ C.SH, np.arange(ka + 1) * dt_plan, spd[:ka + 1, 0]


def _start_slope(st):
    if np.linalg.norm(st['v']) <= 0.5:
        return None                                     # at rest: the burn sets the first direction freely
    return np.array([st['v'] @ C.E1, st['v'] @ C.E2]) / max(st['v'] @ C.SH, 0.5)


def _to_nodes(ys, ss, slope_eps=1e-6):
    """Sideways offsets at stations -> node list, keeping only stations where the direction changes."""
    pts = [C.S + ss[j] * C.SH + ys[j, 0] * C.E1 + ys[j, 1] * C.E2 for j in range(1, len(ss))]
    keep = [pts[j - 1] for j in range(1, len(ss) - 1)
            if np.linalg.norm((ys[j + 1] - ys[j]) / (ss[j + 1] - ss[j]) - (ys[j] - ys[j - 1]) / (ss[j] - ss[j - 1])) > slope_eps]
    return keep + [pts[-1]]


def _station_points(ys, ss):
    """All 3D points along stations 1..J (including collinear stations)."""
    return [C.S + ss[j] * C.SH + ys[j, 0] * C.E1 + ys[j, 1] * C.E2 for j in range(1, len(ss))]
