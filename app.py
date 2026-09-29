import streamlit as st
import numpy as np
import cv2
import os
import tempfile
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import mediapipe as mp
except Exception:
    mp = None

st.set_page_config(
    page_title="Multi-Person Whole-Body Command Arbitration",
    page_icon="🤖",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Constants and humanoid model
# ----------------------------------------------------------------------------

WIDTH = 960
PANEL_W = WIDTH // 2
PANEL_H = 470
DASH_H = 70
HEIGHT = PANEL_H + DASH_H

FPS_DEFAULT = 15
DURATION_DEFAULT = 60

OPS = ("A", "B")
MODE_OPTIONS = ["Auto (demo script)", "Exclusive", "Blended", "Handoff"]

JOINT_NAMES = [
    "pelvis",
    "spine",
    "chest",
    "neck",
    "head",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
    "l_hip",
    "l_knee",
    "l_ankle",
    "r_hip",
    "r_knee",
    "r_ankle",
]

PARENTS = np.array(
    [-1, 0, 1, 2, 2, 2, 5, 6, 2, 8, 9, 0, 11, 12, 0, 14, 15],
    dtype=int,
)

REST_OFFSETS = np.array(
    [
        [0.00, 0.00, 0.00],   # pelvis
        [0.00, 0.18, 0.00],   # spine
        [0.00, 0.18, 0.00],   # chest
        [0.00, 0.16, 0.00],   # neck
        [0.00, 0.12, 0.00],   # head
        [-0.18, 0.00, 0.00],  # l_shoulder
        [0.00, -0.28, 0.00],  # l_elbow
        [0.00, -0.26, 0.00],  # l_wrist
        [0.18, 0.00, 0.00],   # r_shoulder
        [0.00, -0.28, 0.00],  # r_elbow
        [0.00, -0.26, 0.00],  # r_wrist
        [-0.09, 0.00, 0.00],  # l_hip
        [0.00, -0.42, 0.00],  # l_knee
        [0.00, -0.40, 0.00],  # l_ankle
        [0.09, 0.00, 0.00],   # r_hip
        [0.00, -0.42, 0.00],  # r_knee
        [0.00, -0.40, 0.00],  # r_ankle
    ],
    dtype=float,
)

REST_LENGTHS = np.linalg.norm(REST_OFFSETS, axis=1)
EDGES = [(int(PARENTS[i]), i) for i in range(1, 17)]

JOINT_MASSES = np.array(
    [
        0.15, 0.10, 0.15, 0.05, 0.08,
        0.04, 0.03, 0.02,
        0.04, 0.03, 0.02,
        0.06, 0.05, 0.04,
        0.06, 0.05, 0.04,
    ],
    dtype=float,
)
JOINT_MASSES /= JOINT_MASSES.sum()

MP_IDX = {
    "nose": 0,
    "l_shoulder": 11,
    "r_shoulder": 12,
    "l_elbow": 13,
    "r_elbow": 14,
    "l_wrist": 15,
    "r_wrist": 16,
    "l_hip": 23,
    "r_hip": 24,
    "l_knee": 25,
    "r_knee": 26,
    "l_ankle": 27,
    "r_ankle": 28,
}

MP_EDGES = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (0, 11),
    (0, 12),
]

COLLISION_PAIRS = [
    (7, 4),
    (10, 4),
    (7, 2),
    (10, 2),
    (6, 12),
    (10, 15),
    (7, 16),
    (10, 13),
    (12, 15),
]


@dataclass
class Config:
    blend_exponent: float = 2.0
    handoff_margin: float = 0.15
    handoff_duration: float = 1.0
    token_duration: float = 12.0
    token_claim_time: float = 1.0
    balance_gain: float = 0.45
    collision_min_dist: float = 0.11


DEFAULT_PARAMS = {
    "duration_s": float(DURATION_DEFAULT),
    "fps": int(FPS_DEFAULT),
    "mode_override": "Auto (demo script)",
    "blend_exponent": 2.0,
    "handoff_margin": 0.15,
    "handoff_duration": 1.0,
    "token_duration": 12.0,
    "balance_gain": 0.45,
    "collision_min_dist": 0.11,
    "seed": 7,
}


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def smoothstep(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def initial_humanoid_pose(root_x: float = 0.0, root_y: float = 0.95, root_z: float = 0.0) -> np.ndarray:
    pos = np.zeros((17, 3), dtype=float)
    pos[0] = [root_x, root_y, root_z]
    for i in range(1, 17):
        pos[i] = pos[PARENTS[i]] + REST_OFFSETS[i]
    return pos


def weighted_blend(commands: Dict[str, np.ndarray], weights: Dict[str, float]) -> Optional[np.ndarray]:
    total = np.zeros((17, 3), dtype=float)
    wsum = 0.0
    for op, w in weights.items():
        if w <= 1e-6:
            continue
        cmd = commands.get(op)
        if cmd is None:
            continue
        total += float(w) * cmd
        wsum += float(w)
    if wsum <= 1e-6:
        return None
    return total / wsum


# ----------------------------------------------------------------------------
# Synthetic multi-person pose stream in MediaPipe-like format
# ----------------------------------------------------------------------------

def generate_landmarks(op_id: str, t: float, rng) -> Optional[Tuple[np.ndarray, float]]:
    if op_id == "A":
        quality = 0.96 + 0.03 * math.sin(0.3 * t)
        cx = 0.34 + 0.012 * math.sin(0.55 * t)
    else:
        enter = smoothstep((t - 10.5) / 3.5)
        if t < 10.0:
            return None
        quality = 0.22 + 0.74 * enter
        cx = 0.84 - 0.16 * enter

    quality = clamp(quality, 0.0, 1.0)
    cy = 0.56 + 0.012 * math.sin(1.1 * t + (0.0 if op_id == "A" else 1.7))
    wave = math.sin(2.0 * math.pi * 0.75 * t + (0.0 if op_id == "A" else 1.1))
    wave2 = math.sin(2.0 * math.pi * 1.35 * t + (0.4 if op_id == "A" else 2.0))

    raise_amt = 0.0
    if op_id == "A":
        if 38.5 <= t < 39.5:
            raise_amt = smoothstep((t - 38.5) / 0.7)
        elif 39.5 <= t < 41.0:
            raise_amt = 1.0
        elif 41.0 <= t < 42.0:
            raise_amt = 1.0 - smoothstep((t - 41.0) / 1.0)

    pts: Dict[str, Tuple[float, float]] = {}
    pts["nose"] = (cx, cy - 0.28 + 0.006 * wave)
    pts["neck"] = (cx, cy - 0.22 + 0.004 * wave)
    pts["chest"] = (cx, cy - 0.12)
    pts["pelvis"] = (cx, cy + 0.05)

    sh_y = cy - 0.10
    hip_y = cy + 0.05

    pts["l_shoulder"] = (cx - 0.085, sh_y)
    pts["r_shoulder"] = (cx + 0.085, sh_y)

    if raise_amt > 0.01:
        el_y = sh_y - (0.04 + 0.08 * raise_amt) + 0.006 * wave2
        wr_y = sh_y - (0.08 + 0.20 * raise_amt) + 0.010 * math.sin(6.0 * t)
        pts["l_elbow"] = (cx - 0.125 - 0.02 * raise_amt, el_y)
        pts["r_elbow"] = (cx + 0.125 + 0.02 * raise_amt, el_y)
        pts["l_wrist"] = (cx - 0.155 - 0.03 * raise_amt, wr_y)
        pts["r_wrist"] = (cx + 0.155 + 0.03 * raise_amt, wr_y)
    else:
        amp = 0.07 if op_id == "A" else 0.09
        pts["l_elbow"] = (cx - 0.13, sh_y + 0.09 + 0.025 * wave)
        pts["l_wrist"] = (cx - 0.17, sh_y + 0.18 + amp * wave)
        pts["r_elbow"] = (cx + 0.13, sh_y + 0.09 - 0.025 * wave)
        pts["r_wrist"] = (cx + 0.17, sh_y + 0.18 - amp * wave)

    step = math.sin(2.0 * math.pi * 0.85 * t + (0.0 if op_id == "A" else 0.9))
    pts["l_hip"] = (cx - 0.045, hip_y)
    pts["r_hip"] = (cx + 0.045, hip_y)
    pts["l_knee"] = (cx - 0.055 + 0.012 * step, hip_y + 0.18)
    pts["r_knee"] = (cx + 0.055 - 0.012 * step, hip_y + 0.18)
    pts["l_ankle"] = (cx - 0.065 + 0.025 * step, hip_y + 0.36)
    pts["r_ankle"] = (cx + 0.065 - 0.025 * step, hip_y + 0.36)

    lm = np.zeros((33, 3), dtype=float)
    for name, idx in MP_IDX.items():
        if name in pts:
            lm[idx, 0] = pts[name][0]
            lm[idx, 1] = pts[name][1]
            lm[idx, 2] = 1.0

    for i in range(33):
        if lm[i, 2] == 0.0:
            if i < 10:
                lm[i] = lm[0]
            elif i < 23:
                lm[i] = lm[MP_IDX["l_shoulder"]]
            else:
                lm[i] = lm[MP_IDX["l_hip"]]
            lm[i, 2] = 0.20

    visible = lm[:, 2] > 0.25
    if np.any(visible):
        noise = 0.004 + 0.025 * (1.0 - quality)
        n_vis = int(np.count_nonzero(visible))
        lm[visible, 0:2] += rng.normal(0.0, noise, size=(n_vis, 2))
        lm[visible, 2] = np.clip(
            quality * (0.82 + 0.18 * rng.random(n_vis)),
            0.05,
            1.0,
        )

    return lm, quality


def landmarks_to_command(lm: np.ndarray, op_id: str, t: float) -> Optional[np.ndarray]:
    if lm is None:
        return None

    cmd = np.full((17, 3), np.nan, dtype=float)
    vis_thresh = 0.25

    xs = []
    for nm in ("l_hip", "r_hip", "l_shoulder", "r_shoulder"):
        idx = MP_IDX[nm]
        if lm[idx, 2] > vis_thresh:
            xs.append(float(lm[idx, 0]))
    center_x = float(np.mean(xs)) if xs else 0.5

    world_x = -0.58 if op_id == "A" else 0.58
    world_z = 0.00 if op_id == "A" else 0.18

    def conv(xn: float, yn: float) -> np.ndarray:
        x = world_x + (float(xn) - center_x) * 0.82
        y = 0.96 - (float(yn) - 0.50) * 1.95
        z = world_z + (0.50 - float(yn)) * 0.07 + (float(xn) - center_x) * 0.04
        return np.array([x, y, z], dtype=float)

    def put(joint: int, mp_name: str, dx: float = 0.0, dy: float = 0.0) -> None:
        idx = MP_IDX[mp_name]
        if lm[idx, 2] > vis_thresh:
            cmd[joint] = conv(lm[idx, 0] + dx, lm[idx, 1] + dy)

    put(5, "l_shoulder")
    put(8, "r_shoulder")
    put(6, "l_elbow")
    put(9, "r_elbow")
    put(7, "l_wrist")
    put(10, "r_wrist")
    put(11, "l_hip")
    put(14, "r_hip")
    put(12, "l_knee")
    put(15, "r_knee")
    put(13, "l_ankle")
    put(16, "r_ankle")

    hip_vecs = [cmd[j] for j in (11, 14) if np.all(np.isfinite(cmd[j]))]
    if hip_vecs:
        cmd[0] = np.mean(hip_vecs, axis=0)

    sh_vecs = [cmd[j] for j in (5, 8) if np.all(np.isfinite(cmd[j]))]
    if sh_vecs:
        cmd[2] = np.mean(sh_vecs, axis=0) + np.array([0.0, 0.02, 0.0])

    if lm[MP_IDX["nose"], 2] > vis_thresh:
        nose = conv(lm[MP_IDX["nose"], 0], lm[MP_IDX["nose"], 1])
        if not np.all(np.isfinite(cmd[3])):
            cmd[3] = nose + np.array([0.0, 0.07, 0.0])
        if not np.all(np.isfinite(cmd[4])):
            cmd[4] = nose + np.array([0.0, 0.16, 0.0])

    if not np.all(np.isfinite(cmd[0])):
        cmd[0] = np.array([world_x, 0.96, world_z], dtype=float)
    if not np.all(np.isfinite(cmd[2])):
        cmd[2] = cmd[0] + np.array([0.0, 0.36, 0.0])
    if not np.all(np.isfinite(cmd[3])):
        cmd[3] = cmd[2] + np.array([0.0, 0.16, 0.0])
    if not np.all(np.isfinite(cmd[4])):
        cmd[4] = cmd[3] + np.array([0.0, 0.12, 0.0])
    if not np.all(np.isfinite(cmd[1])):
        cmd[1] = 0.5 * (cmd[0] + cmd[2])

    for i in range(1, 17):
        if not np.all(np.isfinite(cmd[i])):
            p = int(PARENTS[i])
            if np.all(np.isfinite(cmd[p])):
                cmd[i] = cmd[p] + REST_OFFSETS[i]

    base = initial_humanoid_pose(float(cmd[0, 0]), float(cmd[0, 1]), float(cmd[0, 2]))
    cmd = np.where(np.isfinite(cmd), cmd, base)

    cmd[0, 1] = clamp(float(cmd[0, 1]), 0.82, 1.12)
    for j in (13, 16):
        cmd[j, 1] = max(float(cmd[j, 1]), 0.03)

    return cmd


def hands_raised(lm: Optional[np.ndarray]) -> bool:
    if lm is None:
        return False
    ids = [
        MP_IDX["l_shoulder"],
        MP_IDX["r_shoulder"],
        MP_IDX["l_wrist"],
        MP_IDX["r_wrist"],
    ]
    if any(lm[i, 2] < 0.35 for i in ids):
        return False
    sh_y = 0.5 * (float(lm[MP_IDX["l_shoulder"], 1]) + float(lm[MP_IDX["r_shoulder"], 1]))
    wr_y = 0.5 * (float(lm[MP_IDX["l_wrist"], 1]) + float(lm[MP_IDX["r_wrist"], 1]))
    return wr_y < sh_y - 0.065


def scripted_confidence_base(op_id: str, t: float) -> float:
    if op_id == "A":
        if t < 12.0:
            return 0.91
        if t < 27.0:
            return 0.61
        if t < 38.5:
            return 0.70
        if t < 40.0:
            return 0.70 + 0.27 * smoothstep((t - 38.5) / 1.5)
        if t < 52.0:
            return 0.97
        return 0.88

    if t < 12.0:
        return 0.00
    if t < 18.0:
        return 0.20 + 0.75 * smoothstep((t - 12.0) / 6.0)
    if t < 27.0:
        return 0.94
    if t < 38.5:
        return 0.82
    if t < 42.0:
        return 0.82 - 0.39 * smoothstep((t - 38.5) / 3.5)
    if t < 52.0:
        return 0.43
    return 0.58


def compute_confidence(
    op_id: str,
    t: float,
    lm: Optional[np.ndarray],
    cmd: Optional[np.ndarray],
    hist: deque,
    quality: float,
) -> float:
    base = scripted_confidence_base(op_id, t)
    if base <= 0.01 or lm is None or cmd is None:
        return 0.0

    jitter = 0.0
    if len(hist) >= 2:
        arr = list(hist)
        diffs = []
        for i in range(1, len(arr)):
            diffs.append(float(np.nanmean(np.abs(arr[i] - arr[i - 1]))))
        if diffs:
            jitter = float(np.mean(diffs[-6:]))

    stability = math.exp(-jitter * 28.0)
    vis_q = clamp(quality, 0.0, 1.0)
    conf = base * (0.58 + 0.42 * vis_q) * (0.66 + 0.34 * stability)
    return float(np.clip(conf, 0.02, 0.99))


def scripted_mode(t: float) -> str:
    if t < 12.0:
        return "Exclusive"
    if t < 27.0:
        return "Handoff"
    if t < 40.0:
        return "Blended"
    if t < 52.0:
        return "Exclusive"
    return "Handoff"


# ----------------------------------------------------------------------------
# Arbitration
# ----------------------------------------------------------------------------

class Arbitrator:
    def __init__(self) -> None:
        self.current: Optional[str] = None
        self.handoff_from: Optional[str] = None
        self.handoff_to: Optional[str] = None
        self.handoff_progress: float = 1.0
        self.token_holder: Optional[str] = None

    def update(
        self,
        t: float,
        dt: float,
        commands: Dict[str, np.ndarray],
        confs: Dict[str, float],
        mode: str,
        cfg: Config,
        token_holder: Optional[str] = None,
        token_expiry: float = -1.0,
    ) -> Tuple[Optional[np.ndarray], Dict[str, float], str, str, str]:
        self.token_holder = token_holder if (token_holder is not None and token_expiry > t) else None

        present = [
            op
            for op in OPS
            if op in commands and commands[op] is not None and confs.get(op, 0.0) > 0.04
        ]

        if not present:
            if self.current is not None and self.current in commands and commands[self.current] is not None:
                return (
                    commands[self.current].copy(),
                    {self.current: 1.0},
                    f"Operator {self.current} (hold)",
                    "hold-last",
                    mode,
                )
            any_op = next(iter(commands), None)
            if any_op is not None and commands[any_op] is not None:
                self.current = any_op
                return (
                    commands[any_op].copy(),
                    {any_op: 1.0},
                    f"Operator {any_op} (fallback)",
                    "fallback",
                    mode,
                )
            return None, {}, "No operator", "idle", mode

        if self.token_holder is not None and self.token_holder in present:
            self.current = self.token_holder
            self.handoff_from = None
            self.handoff_to = None
            self.handoff_progress = 1.0
            return (
                commands[self.current].copy(),
                {self.current: 1.0},
                f"Operator {self.current} (priority token)",
                "priority-token",
                mode,
            )

        if mode == "Exclusive":
            best = max(present, key=lambda o: confs.get(o, 0.0))
            self.current = best
            self.handoff_to = None
            self.handoff_progress = 1.0
            return (
                commands[best].copy(),
                {best: 1.0},
                f"Operator {best}",
                "exclusive-confidence",
                mode,
            )

        if mode == "Blended":
            raw = {op: max(confs.get(op, 0.0), 0.01) ** cfg.blend_exponent for op in present}
            s = sum(raw.values())
            weights = {op: float(v / s) for op, v in raw.items()}
            cmd = weighted_blend(commands, weights)
            if cmd is None:
                best = max(present, key=lambda o: confs.get(o, 0.0))
                cmd = commands[best].copy()
                weights = {best: 1.0}

            max_op = max(weights, key=lambda k: weights[k])
            if weights[max_op] > 0.85:
                active = f"Operator {max_op}"
            else:
                parts = [f"{op}:{weights[op]:.2f}" for op in sorted(weights, key=lambda k: -weights[k]) if weights[op] > 0.02]
                active = "Blended " + " / ".join(parts)
            return cmd, weights, active, "confidence-blend", mode

        # Handoff mode
        best = max(present, key=lambda o: confs.get(o, 0.0))

        if self.current not in present:
            self.current = best
            self.handoff_to = None
            self.handoff_progress = 1.0

        if self.handoff_to is not None and self.handoff_to not in present:
            self.handoff_to = None
            self.handoff_progress = 1.0

        if self.handoff_to is None and best != self.current:
            if confs.get(best, 0.0) > confs.get(self.current, 0.0) + cfg.handoff_margin:
                self.handoff_from = self.current
                self.handoff_to = best
                self.handoff_progress = 0.0

        if (
            self.handoff_to is not None
            and self.handoff_from is not None
            and self.handoff_from in commands
            and self.handoff_to in commands
        ):
            self.handoff_progress = min(
                1.0,
                self.handoff_progress + dt / max(cfg.handoff_duration, 1e-3),
            )
            a = smoothstep(self.handoff_progress)
            cmd = (1.0 - a) * commands[self.handoff_from] + a * commands[self.handoff_to]
            weights = {self.handoff_from: float(1.0 - a), self.handoff_to: float(a)}
            active = f"Handoff {self.handoff_from}->{self.handoff_to} {a:.0%}"
            status = "smooth-handoff"

            if self.handoff_progress >= 1.0:
                self.current = self.handoff_to
                self.handoff_from = None
                self.handoff_to = None
                self.handoff_progress = 1.0

            return cmd, weights, active, status, mode

        self.current = best
        return (
            commands[best].copy(),
            {best: 1.0},
            f"Operator {best}",
            "handoff-held",
            mode,
        )


# ----------------------------------------------------------------------------
# Safety and physics twin
# ----------------------------------------------------------------------------

def apply_safety(cmd: Optional[np.ndarray], cfg: Config) -> Optional[np.ndarray]:
    if cmd is None:
        return None

    c = np.array(cmd, dtype=float, copy=True)
    root = c[0].copy()
    if not np.all(np.isfinite(root)):
        root = np.array([0.0, 0.95, 0.0], dtype=float)

    base = initial_humanoid_pose(float(root[0]), float(root[1]), float(root[2]))
    c = np.where(np.isfinite(c), c, base)

    # Balance projection
    feet = c[[13, 16]]
    if np.all(np.isfinite(feet)):
        support = np.mean(feet, axis=0)
        desired_y = clamp(float(support[1] + 0.92), 0.82, 1.12)
        c[0, 1] += cfg.balance_gain * (desired_y - c[0, 1])

        com = np.average(c, axis=0, weights=JOINT_MASSES)
        dx = float(support[0] - com[0])
        dz = float(support[2] - com[2])
        shift = np.array([dx, 0.0, dz], dtype=float) * cfg.balance_gain * 0.35
        shift[0] = clamp(shift[0], -0.12, 0.12)
        shift[2] = clamp(shift[2], -0.12, 0.12)

        mask = np.ones(17, dtype=bool)
        mask[[13, 16]] = False
        c[mask] += shift

    # Self-collision separation
    for i, j in COLLISION_PAIRS:
        if not (np.all(np.isfinite(c[i])) and np.all(np.isfinite(c[j]))):
            continue
        v = c[j] - c[i]
        d = float(np.linalg.norm(v))
        if d < cfg.collision_min_dist:
            if d > 1e-6:
                push = 0.5 * (cfg.collision_min_dist - d) * (v / d)
                c[i] -= push
                c[j] += push
            else:
                c[j, 0] += 0.5 * cfg.collision_min_dist

    # Ground contact
    c[13, 1] = max(float(c[13, 1]), 0.02)
    c[16, 1] = max(float(c[16, 1]), 0.02)

    return c


class HumanoidSim:
    def __init__(self, root_x: float = 0.0) -> None:
        self.pos = initial_humanoid_pose(root_x=root_x)
        self.vel = np.zeros_like(self.pos)

    def _integrate(self, target: np.ndarray, dt: float) -> None:
        kp = 90.0
        kd = 20.0
        acc = kp * (target - self.pos) - kd * self.vel
        acc[:, 1] -= 2.0

        self.vel += acc * dt
        self.vel *= 0.985
        self.pos += self.vel * dt

        for i in range(1, 17):
            p = int(PARENTS[i])
            v = self.pos[i] - self.pos[p]
            d = float(np.linalg.norm(v))
            if d > 1e-6:
                self.pos[i] = self.pos[p] + (v / d) * float(REST_LENGTHS[i])

        for i in (13, 16):
            if self.pos[i, 1] < 0.02:
                self.pos[i, 1] = 0.02
                self.vel[i, 1] = max(0.0, float(self.vel[i, 1]))

    def step(self, target: Optional[np.ndarray], dt: float) -> np.ndarray:
        if target is None:
            target = self.pos
        target = np.asarray(target, dtype=float)
        target = np.where(np.isfinite(target), target, self.pos)

        sub_dt = dt / 2.0
        for _ in range(2):
            self._integrate(target, sub_dt)
        return self.pos.copy()


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------

def project_points(points: np.ndarray, panel_w: int, panel_h: int) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    cam = np.array([0.0, 1.05, -2.75], dtype=float)
    rel = pts - cam
    rel[:, 2] = np.maximum(rel[:, 2], 0.18)

    fx = panel_w * 0.82
    fy = panel_h * 0.82
    cx = panel_w * 0.5
    cy = panel_h * 0.64

    sx = cx + fx * rel[:, 0] / rel[:, 2]
    sy = cy - fy * rel[:, 1] / rel[:, 2]
    out = np.column_stack((sx, sy))
    valid = np.isfinite(out).all(axis=1)
    return out, valid


def draw_skeleton_on(img: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], thickness: int = 2) -> None:
    proj, valid = project_points(points, img.shape[1], img.shape[0])
    for a, b in EDGES:
        if valid[a] and valid[b]:
            pt1 = (int(round(proj[a, 0])), int(round(proj[a, 1])))
            pt2 = (int(round(proj[b, 0])), int(round(proj[b, 1])))
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

    for i in range(17):
        if valid[i]:
            pt = (int(round(proj[i, 0])), int(round(proj[i, 1])))
            cv2.circle(img, pt, max(2, thickness), color, -1, cv2.LINE_AA)


def draw_ghost(panel: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], alpha: float) -> None:
    overlay = np.zeros_like(panel)
    draw_skeleton_on(overlay, points, color, 2)
    panel[:] = cv2.addWeighted(overlay, clamp(alpha, 0.0, 1.0), panel, 1.0, 0.0)


def draw_floor(panel: np.ndarray) -> None:
    for z in np.arange(0.0, 3.5, 0.5):
        pts = np.array([[-1.8, 0.0, z], [1.8, 0.0, z]], dtype=float)
        proj, valid = project_points(pts, panel.shape[1], panel.shape[0])
        if valid[0] and valid[1]:
            cv2.line(
                panel,
                (int(proj[0, 0]), int(proj[0, 1])),
                (int(proj[1, 0]), int(proj[1, 1])),
                (45, 45, 45),
                1,
                cv2.LINE_AA,
            )

    for x in np.arange(-1.8, 1.9, 0.45):
        pts = np.array([[x, 0.0, 0.0], [x, 0.0, 3.2]], dtype=float)
        proj, valid = project_points(pts, panel.shape[1], panel.shape[0])
        if valid[0] and valid[1]:
            cv2.line(
                panel,
                (int(proj[0, 0]), int(proj[0, 1])),
                (int(proj[1, 0]), int(proj[1, 1])),
                (38, 38, 38),
                1,
                cv2.LINE_AA,
            )


def draw_perception(
    dets: Dict[str, np.ndarray],
    confs: Dict[str, float],
    token_info: Dict[str, Any],
    t: float,
) -> np.ndarray:
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    for y in range(PANEL_H):
        frac = y / max(1, PANEL_H - 1)
        panel[y, :, :] = (
            int(18 + 10 * frac),
            int(26 + 14 * frac),
            int(42 + 22 * frac),
        )

    cv2.line(panel, (0, int(PANEL_H * 0.82)), (PANEL_W, int(PANEL_H * 0.82)), (70, 70, 70), 2, cv2.LINE_AA)

    colors = {
        "A": (255, 180, 60),
        "B": (60, 180, 255),
    }

    key_indices = [MP_IDX[k] for k in (
        "nose",
        "l_shoulder",
        "r_shoulder",
        "l_elbow",
        "r_elbow",
        "l_wrist",
        "r_wrist",
        "l_hip",
        "r_hip",
        "l_knee",
        "r_knee",
        "l_ankle",
        "r_ankle",
    )]

    for op, lm in dets.items():
        if lm is None:
            continue
        vis = lm[:, 2] > 0.25
        pts_px = []
        for i in range(33):
            x = int(clamp(lm[i, 0], 0.0, 1.0) * PANEL_W)
            y = int(clamp(lm[i, 1], 0.0, 1.0) * PANEL_H)
            pts_px.append((x, y))

        for a, b in MP_EDGES:
            if vis[a] and vis[b]:
                cv2.line(panel, pts_px[a], pts_px[b], colors.get(op, (200, 200, 200)), 2, cv2.LINE_AA)

        for i in key_indices:
            if vis[i]:
                cv2.circle(panel, pts_px[i], 3, (245, 245, 245), -1, cv2.LINE_AA)

        xs = [pts_px[i][0] for i in key_indices if vis[i]]
        ys = [pts_px[i][1] for i in key_indices if vis[i]]
        if xs and ys:
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            cv2.rectangle(panel, (x1, y1), (x2, y2), colors.get(op, (200, 200, 200)), 1, cv2.LINE_AA)
            label = f"Op {op} conf {confs.get(op, 0.0):.2f}"
            cv2.putText(
                panel,
                label,
                (x1, max(14, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            if token_info.get("holder") == op and float(token_info.get("expiry", 0.0)) > t:
                rem = max(0.0, float(token_info["expiry"]) - t)
                cv2.putText(
                    panel,
                    f"PRIORITY {rem:.1f}s",
                    (x1 + 2, max(30, y1 - 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    cv2.putText(
        panel,
        "PERCEPTION / Multi-person pose",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return panel


def draw_simulation(
    sim_pos: Optional[np.ndarray],
    trails: deque,
    candidate_cmds: Dict[str, np.ndarray],
    weights: Dict[str, float],
    final_cmd: Optional[np.ndarray],
    t: float,
    token_info: Dict[str, Any],
    mode: str,
    status: str,
    confs: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    for y in range(PANEL_H):
        frac = y / max(1, PANEL_H - 1)
        panel[y, :, :] = (
            int(12 + 8 * frac),
            int(16 + 10 * frac),
            int(24 + 14 * frac),
        )

    draw_floor(panel)

    # Fading FK trail
    if trails:
        n = max(1, len(trails))
        for idx, tr in enumerate(trails):
            a = idx / max(1, n - 1)
            color = (
                int(35 + 70 * a),
                int(80 + 90 * a),
                int(130 + 90 * a),
            )
            draw_skeleton_on(panel, tr, color, 1)

    # Candidate whole-body commands
    op_colors = {
        "A": (255, 180, 60),
        "B": (60, 180, 255),
    }
    for op, cmd in candidate_cmds.items():
        if cmd is None:
            continue
        w = float(weights.get(op, 0.0))
        alpha = 0.12 + 0.45 * w
        if confs is not None and confs.get(op, 0.0) > 0.25 and w < 0.02:
            alpha = max(alpha, 0.16)

        color = op_colors.get(op, (200, 200, 200))
        draw_ghost(panel, cmd, color, alpha)

        marker_pts = cmd[[0, 4, 7, 10]]
        proj, valid = project_points(marker_pts, PANEL_W, PANEL_H)
        for k in range(4):
            if valid[k]:
                cv2.circle(
                    panel,
                    (int(proj[k, 0]), int(proj[k, 1])),
                    5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    # Final arbitrated target
    if final_cmd is not None:
        draw_skeleton_on(panel, final_cmd, (80, 255, 120), 2)

    # Actual physics-twin pose
    if sim_pos is not None:
        draw_skeleton_on(panel, sim_pos, (245, 245, 245), 3)

    cv2.putText(
        panel,
        "WHOLE-BODY PHYSICS TWIN / FK + arbitration",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"Mode: {mode}",
        (PANEL_W - 165, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"Status: {status}",
        (PANEL_W - 165, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return panel


def draw_dashboard(
    confs: Dict[str, float],
    weights: Dict[str, float],
    mode: str,
    status: str,
    active_label: str,
    token_info: Dict[str, Any],
    t: float,
) -> np.ndarray:
    img = np.full((DASH_H, WIDTH, 3), (28, 32, 38), dtype=np.uint8)

    bar_specs = [
        ("A", 12, (255, 180, 60)),
        ("B", 210, (60, 180, 255)),
    ]
    for op, x0, color in bar_specs:
        cv2.putText(
            img,
            f"Op {op}",
            (x0, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        bx = x0 + 42
        by = 12
        bw = 110
        bh = 12
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (85, 85, 85), 1, cv2.LINE_AA)
        val = int(clamp(confs.get(op, 0.0), 0.0, 1.0) * bw)
        cv2.rectangle(img, (bx, by), (bx + val, by + bh), color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            f"{confs.get(op, 0.0):.2f}",
            (bx + bw + 6, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        img,
        f"Arbitration: {mode}",
        (420, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"Status: {status}",
        (420, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        active_label[:48],
        (420, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (180, 255, 180),
        1,
        cv2.LINE_AA,
    )

    holder = token_info.get("holder")
    expiry = float(token_info.get("expiry", 0.0))
    if holder and expiry > t:
        rem = max(0.0, expiry - t)
        cv2.putText(
            img,
            f"PRIORITY TOKEN Op {holder} {rem:.1f}s",
            (700, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        bx, by, bw, bh = 700, 28, 180, 10
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (85, 85, 85), 1, cv2.LINE_AA)
        frac = clamp(rem / 12.0, 0.0, 1.0)
        cv2.rectangle(img, (bx, by), (bx + int(frac * bw), by + bh), (0, 255, 255), -1, cv2.LINE_AA)
    else:
        cv2.putText(
            img,
            "Priority token: idle",
            (700, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        img,
        f"t={t:05.2f}s",
        (865, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return img


# ----------------------------------------------------------------------------
# 60-second recorded demo generator
# ----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def render_demo_video(
    duration_s: float,
    fps: int,
    mode_override: str,
    blend_exponent: float,
    handoff_margin: float,
    handoff_duration: float,
    token_duration: float,
    balance_gain: float,
    collision_min_dist: float,
    seed: int,
) -> Tuple[bytes, str, Dict[str, Any]]:
    cfg = Config(
        blend_exponent=float(blend_exponent),
        handoff_margin=float(handoff_margin),
        handoff_duration=float(handoff_duration),
        token_duration=float(token_duration),
        balance_gain=float(balance_gain),
        collision_min_dist=float(collision_min_dist),
    )

    rng = np.random.default_rng(int(seed))
    sim = HumanoidSim(root_x=0.0)
    arb = Arbitrator()

    hist = {op: deque(maxlen=12) for op in OPS}
    raised_since = {op: None for op in OPS}
    token_holder: Optional[str] = None
    token_expiry = -1.0

    trails = deque(maxlen=32)
    total_frames = max(1, int(round(float(duration_s) * float(fps))))
    dt = 1.0 / float(fps)

    mp4_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (WIDTH, HEIGHT))
    ext = "mp4"
    path = mp4_path

    if not writer.isOpened():
        try:
            os.unlink(mp4_path)
        except Exception:
            pass
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (WIDTH, HEIGHT))
        ext = "avi"

    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open a video writer. Install opencv-python and retry.")

    confs = {op: 0.0 for op in OPS}
    weights: Dict[str, float] = {}
    mode_used = "Init"
    status_label = "init"
    active_label = "None"
    t = 0.0

    try:
        for frame in range(total_frames):
            t = frame * dt

            dets: Dict[str, np.ndarray] = {}
            quals: Dict[str, float] = {}
            for op in OPS:
                res = generate_landmarks(op, t, rng)
                if res is not None:
                    dets[op], quals[op] = res

            cmds: Dict[str, np.ndarray] = {}
            for op, lm in dets.items():
                cmd = landmarks_to_command(lm, op, t)
                if cmd is not None and np.all(np.isfinite(cmd)):
                    cmds[op] = cmd

            for op in OPS:
                if op in cmds:
                    confs[op] = compute_confidence(
                        op,
                        t,
                        dets.get(op),
                        cmds.get(op),
                        hist[op],
                        quals.get(op, 0.0),
                    )
                    hist[op].append(cmds[op])
                else:
                    confs[op] = 0.0

            # Priority token detection
            for op, lm in dets.items():
                if hands_raised(lm):
                    if raised_since[op] is None:
                        raised_since[op] = t
                    if token_expiry <= t and (t - float(raised_since[op])) >= cfg.token_claim_time:
                        if token_holder is None or token_expiry <= t or confs[op] >= confs.get(token_holder, 0.0):
                            token_holder = op
                            token_expiry = t + cfg.token_duration
                            raised_since[op] = None
                else:
                    raised_since[op] = None

            if token_expiry <= t:
                token_holder = None

            token_info = {
                "holder": token_holder if token_expiry > t else None,
                "expiry": token_expiry,
            }

            resolved_mode = scripted_mode(t) if str(mode_override).startswith("Auto") else str(mode_override)

            result_cmd, weights, active_label, status_label, mode_used = arb.update(
                t,
                dt,
                cmds,
                confs,
                resolved_mode,
                cfg,
                token_holder,
                token_expiry,
            )

            if result_cmd is None:
                result_cmd = sim.pos.copy()

            final_cmd = apply_safety(result_cmd, cfg)
            sim_pos = sim.step(final_cmd, dt)

            if frame % 2 == 0:
                trails.append(sim_pos.copy())

            perc = draw_perception(dets, confs, token_info, t)
            simimg = draw_simulation(
                sim_pos,
                trails,
                cmds,
                weights,
                final_cmd,
                t,
                token_info,
                mode_used,
                status_label,
                confs,
            )
            dash = draw_dashboard(confs, weights, mode_used, status_label, active_label, token_info, t)
            frame_img = np.vstack((np.hstack((perc, simimg)), dash))
            writer.write(np.ascontiguousarray(frame_img, dtype=np.uint8))
    finally:
        writer.release()

    with open(path, "rb") as f:
        data = f.read()
    try:
        os.unlink(path)
    except Exception:
        pass

    meta = {
        "final_time": float(t),
        "confidences": {op: float(confs.get(op, 0.0)) for op in OPS},
        "weights": {op: float(v) for op, v in weights.items()},
        "mode": str(mode_used),
        "status": str(status_label),
        "active": str(active_label),
        "token_holder": token_holder if token_expiry > t else None,
        "token_remaining": float(max(0.0, token_expiry - t)) if token_holder and token_expiry > t else 0.0,
    }
    return data, ext, meta


# ----------------------------------------------------------------------------
# Instant interactive preview frame
# ----------------------------------------------------------------------------

def build_instant_frame(
    t: float,
    mode_override: str,
    cfg: Config,
    manual_conf: Optional[Dict[str, float]] = None,
    manual_presence: Optional[Dict[str, bool]] = None,
    force_token: bool = False,
    seed: int = 7,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float], str, str, str, Optional[str], float]:
    rng = np.random.default_rng(int(seed + round(t * 100.0)))

    dets: Dict[str, np.ndarray] = {}
    quals: Dict[str, float] = {}
    for op in OPS:
        if manual_presence is not None and not manual_presence.get(op, True):
            continue
        res = generate_landmarks(op, t, rng)
        if res is not None:
            dets[op], quals[op] = res

    cmds: Dict[str, np.ndarray] = {}
    for op, lm in dets.items():
        cmd = landmarks_to_command(lm, op, t)
        if cmd is not None and np.all(np.isfinite(cmd)):
            cmds[op] = cmd

    confs: Dict[str, float] = {op: 0.0 for op in OPS}
    if manual_conf is not None:
        for op in OPS:
            if op in cmds:
                confs[op] = float(np.clip(manual_conf.get(op, 0.0), 0.0, 1.0))
    else:
        hist = {op: deque() for op in OPS}
        for op in OPS:
            if op in cmds:
                confs[op] = compute_confidence(
                    op,
                    t,
                    dets.get(op),
                    cmds.get(op),
                    hist[op],
                    quals.get(op, 0.0),
                )

    token_holder: Optional[str] = None
    token_expiry = -1.0

    if force_token and "A" in cmds:
        token_holder = "A"
        token_expiry = t + cfg.token_duration
    else:
        for op, lm in dets.items():
            if hands_raised(lm):
                token_holder = op
                token_expiry = t + cfg.token_duration
                break

    mode_used = scripted_mode(t) if str(mode_override).startswith("Auto") else str(mode_override)

    arb = Arbitrator()
    cmd, weights, active_label, status_label, mode_used = arb.update(
        t,
        0.1,
        cmds,
        confs,
        mode_used,
        cfg,
        token_holder,
        token_expiry,
    )

    if cmd is None:
        cmd = initial_humanoid_pose()

    final_cmd = apply_safety(cmd, cfg)

    token_active = token_holder if token_expiry > t else None
    token_remaining = max(0.0, token_expiry - t) if token_active else 0.0
    token_info = {"holder": token_active, "expiry": token_expiry}

    perc = draw_perception(dets, confs, token_info, t)
    simimg = draw_simulation(
        final_cmd,
        deque([final_cmd], maxlen=1),
        cmds,
        weights,
        final_cmd,
        t,
        token_info,
        mode_used,
        status_label,
        confs,
    )
    dash = draw_dashboard(confs, weights, mode_used, status_label, active_label, token_info, t)
    frame = np.vstack((np.hstack((perc, simimg)), dash))

    return frame, confs, weights, mode_used, status_label, active_label, token_active, token_remaining


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------

st.title("Multi-Person Whole-Body Command Arbitration")
st.caption(
    "Two operators can gesture at the same humanoid. A confidence + priority-token arbitration layer "
    "blends or hands off whole-body commands. Safety balance/self-collision projections are applied only "
    "to the final arbitrated command."
)

st.sidebar.title("Control room")
st.sidebar.caption("No API key required. Demo data auto-loads.")

if "params" not in st.session_state:
    st.session_state.params = DEFAULT_PARAMS.copy()

p = st.session_state.params

duration_s = st.sidebar.slider("Record duration (s)", 5.0, 60.0, float(p["duration_s"]), 1.0)
fps = st.sidebar.select_slider("FPS", options=[10, 12, 15, 20, 24, 30], value=int(p["fps"]))
mode_override = st.sidebar.selectbox(
    "Recorder arbitration mode",
    MODE_OPTIONS,
    index=MODE_OPTIONS.index(p["mode_override"]) if p["mode_override"] in MODE_OPTIONS else 0,
)
blend_exponent = st.sidebar.slider("Blend exponent", 1.0, 5.0, float(p["blend_exponent"]), 0.1)
handoff_margin = st.sidebar.slider("Handoff confidence margin", 0.02, 0.40, float(p["handoff_margin"]), 0.01)
handoff_duration = st.sidebar.slider("Handoff duration (s)", 0.3, 3.0, float(p["handoff_duration"]), 0.1)
token_duration = st.sidebar.slider("Priority token duration (s)", 2.0, 20.0, float(p["token_duration"]), 1.0)
balance_gain = st.sidebar.slider("Balance projection gain", 0.0, 1.0, float(p["balance_gain"]), 0.01)
collision_min_dist = st.sidebar.slider("Self-collision min distance (m)", 0.02, 0.25, float(p["collision_min_dist"]), 0.005)
seed = st.sidebar.number_input("Random seed", value=int(p["seed"]), step=1, format="%d")

if st.sidebar.button("Apply && regenerate demo", use_container_width=True):
    new_params = {
        "duration_s": float(duration_s),
        "fps": int(fps),
        "mode_override": str(mode_override),
        "blend_exponent": float(blend_exponent),
        "handoff_margin": float(handoff_margin),
        "handoff_duration": float(handoff_duration),
        "token_duration": float(token_duration),
        "balance_gain": float(balance_gain),
        "collision_min_dist": float(collision_min_dist),
        "seed": int(seed),
    }
    st.session_state.params = new_params
    with st.spinner("Rendering multi-operator arbitration demo..."):
        try:
            st.session_state.video_result = render_demo_video(**new_params)
            st.session_state.pop("video_error", None)
        except Exception as exc:
            st.session_state.video_result = None
            st.session_state.video_error = str(exc)

if st.sidebar.button("Reset defaults", use_container_width=True):
    st.session_state.params = DEFAULT_PARAMS.copy()
    st.session_state.pop("video_result", None)
    st.session_state.pop("video_error", None)

if "video_result" not in st.session_state:
    with st.spinner("Auto-generating 60-second live demo..."):
        try:
            st.session_state.video_result = render_demo_video(**st.session_state.params)
            st.session_state.pop("video_error", None)
        except Exception as exc:
            st.session_state.video_result = None
            st.session_state.video_error = str(exc)

st.markdown("---")
st.subheader("60-second live demo recording")

if st.session_state.get("video_error"):
    st.warning(f"Video renderer failed: {st.session_state.video_error}")
    st.info("The interactive preview below still works and can be used to inspect arbitration behavior.")

if st.session_state.get("video_result") is not None:
    video_bytes, ext, meta = st.session_state.video_result
    mime = "video/mp4" if ext == "mp4" else "video/x-msvideo"
    st.video(video_bytes, format=mime)
    st.download_button(
        f"Download {int(meta['final_time'])}s demo ({ext.upper()})",
        data=video_bytes,
        file_name=f"multi_operator_whole_body_arbitration_demo.{ext}",
        mime=mime,
        use_container_width=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final mode", meta["mode"])
    m2.metric("Final status", meta["status"])
    m3.metric("Active", meta["active"][:32])
    m4.metric(
        "Priority token",
        f"{meta['token_holder']} {meta['token_remaining']:.1f}s" if meta["token_holder"] else "idle",
    )

    c1, c2 = st.columns(2)
    c1.write("Operator A confidence")
    c1.progress(float(meta["confidences"].get("A", 0.0)))
    c2.write("Operator B confidence")
    c2.progress(float(meta["confidences"].get("B", 0.0)))
else:
    st.info("Waiting for demo renderer...")

st.markdown("---")
st.subheader("Interactive arbitration preview")

p = st.session_state.params
cfg = Config(
    blend_exponent=float(p["blend_exponent"]),
    handoff_margin=float(p["handoff_margin"]),
    handoff_duration=float(p["handoff_duration"]),
    token_duration=float(p["token_duration"]),
    balance_gain=float(p["balance_gain"]),
    collision_min_dist=float(p["collision_min_dist"]),
)

col1, col2, col3 = st.columns(3)

with col1:
    manual_t = st.slider("Preview time (s)", 0.0, 60.0, 33.0, 0.1)
    manual_mode = st.selectbox("Preview mode", MODE_OPTIONS, index=0)

with col2:
    use_manual = st.checkbox("Manual confidence override", value=False)
    present_a = st.checkbox("Operator A present", value=True, disabled=not use_manual)
    present_b = st.checkbox("Operator B present", value=bool(manual_t >= 12.0), disabled=not use_manual)
    conf_a = st.slider("Operator A confidence", 0.0, 1.0, 0.90, 0.01, disabled=not use_manual)
    conf_b = st.slider("Operator B confidence", 0.0, 1.0, 0.55, 0.01, disabled=not use_manual)

with col3:
    force_token = st.checkbox("Force Operator A priority token", value=False)
    st.caption(
        "Priority token claims exclusive whole-body control for the configured duration and overrides "
        "pure confidence blending."
    )

manual_conf = None
manual_presence = None
if use_manual:
    manual_conf = {
        "A": float(conf_a) if present_a else 0.0,
        "B": float(conf_b) if present_b else 0.0,
    }
    manual_presence = {"A": bool(present_a), "B": bool(present_b)}

frame, confs_i, weights_i, mode_i, status_i, active_i, token_i, token_rem_i = build_instant_frame(
    manual_t,
    manual_mode,
    cfg,
    manual_conf,
    manual_presence,
    force_token,
    int(p["seed"]),
)

st.image(frame, channels="BGR", use_container_width=True)

im1, im2, im3, im4 = st.columns(4)
im1.metric("Mode", mode_i)
im2.metric("Status", status_i)
im3.metric("Active", active_i[:34])
im4.metric("Token", f"{token_i} {token_rem_i:.1f}s" if token_i else "idle")

pc1, pc2 = st.columns(2)
pc1.write("Operator A confidence")
pc1.progress(float(confs_i.get("A", 0.0)))
pc2.write("Operator B confidence")
pc2.progress(float(confs_i.get("B", 0.0)))

if weights_i:
    st.write("Command weights")
    wcols = st.columns(len(weights_i))
    for wc, (op, w) in zip(wcols, sorted(weights_i.items(), key=lambda kv: -kv[1])):
        wc.metric(f"Operator {op}", f"{float(w):.2f}")

st.markdown("---")
st.subheader("4-frame FK storyboard")

story_times = [6.0, 20.0, 34.0, 46.0]
story_cols = st.columns(4)
for sc, st_t in zip(story_cols, story_times):
    fr, *_ = build_instant_frame(
        st_t,
        manual_mode,
        cfg,
        manual_conf,
        manual_presence,
        force_token,
        int(p["seed"]),
    )
    sc.image(fr, channels="BGR", caption=f"t = {st_t:.0f}s", use_container_width=True)

st.markdown("---")
st.caption(
    "Implementation notes: MediaPipe-style multi-person pose streams are synthesized for reliable offline demo. "
    "The arbitration layer supports Exclusive, Blended, and Handoff modes. Priority tokens override blending. "
    "Balance and self-collision safety projections are applied after arbitration, never to raw operator commands."
)