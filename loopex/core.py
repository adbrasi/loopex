"""Seamless loop detection.

Every frame becomes a small grayscale thumbnail and every pair of frames gets a
distance. A loop is frames [start, repeat) played over and over; it is seamless
when the frames around repeat match the frames around start, so playback can
jump back without a visible discontinuity. Matching the neighbors as well makes
the seam respect the direction of motion. Two scale-free ratios judge each
candidate, so nothing depends on resolution, brightness or speed of motion:

- return ratio: seam error divided by how far the content travels away from
  the start frame during the loop. Real loops come back much closer than they
  travel; trivially short "loops" do not. This replaces the usual minimum
  loop length setting.
- seam: seam error measured in typical frame-to-frame steps around the seam.
  Near 0 the jump is invisible; around 1 it reads as one skipped frame.

Loops the video itself plays at least twice in a row come first; a loop that
closes only once (idle, action, back to idle) is a fallback. Among them the
best seam wins, and within its repetition the shortest loop with an equivalent
seam is the fundamental cycle; whole multiples of it add nothing. To find every
loop, the repetition of a chosen loop is set aside and the search continues on
the rest of the video.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction

import numpy as np

THUMB_SIDE = 128
# Frame offsets compared around the seam, relative to (start, repeat), and their
# weights. Matching neighbors too makes the seam respect motion direction.
SEAM_OFFSETS = (-2, -1, 0, 1)
SEAM_WEIGHTS = (1.0, 3.0, 3.0, 1.0)
# A loop's seam error must be at most this fraction of how far it travels.
MAX_RETURN_RATIO = 0.5
# A loop must travel this many times farther than the footage's noise floor
# (its smallest nonzero frame steps), or there is no motion to loop.
MIN_MOTION_OVER_NOISE = 4.0
# Missed frames bridged when following a repetition along its period.
MAX_REPEAT_GAP = 3
# Seams within this factor plus slack of the best one count as equivalent;
# the slack absorbs measurement noise between near-perfect seams.
SEAM_TOLERANCE = 2.0
SEAM_SLACK = 0.01
# Seams worse than this many frame steps are visible pops, not loops.
MAX_SEAM = 3.0


@dataclass(frozen=True)
class Loop:
    start: int  # first frame of the loop
    end: int  # last frame of the loop (inclusive)
    frames: int
    start_time: float
    duration: float
    seam: float  # seam jump in frame steps (lower is better)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Video:
    frames: np.ndarray  # (n, pixels) float32 grayscale thumbnails
    fps: Fraction


def probe(path: str) -> tuple[int, int, Fraction]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate", "-of", "json", path],
        check=True, capture_output=True, text=True,
    ).stdout
    streams = json.loads(out).get("streams")
    if not streams:
        raise ValueError(f"no video stream in {path}")
    s = streams[0]
    rate = s.get("avg_frame_rate", "0/0")
    if rate in ("0/0", "0/1"):
        rate = s["r_frame_rate"]
    return int(s["width"]), int(s["height"]), Fraction(rate)


def decode(path: str) -> Video:
    width, height, fps = probe(path)
    scale = THUMB_SIDE / max(width, height)
    w, h = max(2, round(width * scale)), max(2, round(height * scale))
    # Skipping the deblocking filter speeds up decoding; its effect vanishes in
    # the area-averaged thumbnail. Passthrough keeps one output per decoded frame
    # so indices match ffmpeg's trim filter.
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-skip_loop_filter", "all", "-i", path, "-map", "0:v:0",
         "-vf", f"scale={w}:{h}:flags=area", "-fps_mode", "passthrough",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        check=True, capture_output=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, w * h).astype(np.float32) / 255.0
    return Video(frames, fps)


def distances(frames: np.ndarray) -> np.ndarray:
    """Mean squared difference between every pair of frames."""
    x = frames - frames.mean(axis=0)
    sq = np.einsum("ij,ij->i", x, x)
    d = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    return np.maximum(d, 0.0) / x.shape[1]


def candidates(d: np.ndarray) -> tuple[np.ndarray, ...]:
    """Score every (start, repeat) pair.

    Returns (seam, valid, after, before, length) matrices indexed
    [start, repeat], where the loop is frames start..repeat-1 and repeat == n
    means it runs to the end. `after` and `before` measure how far the loop's
    period keeps repeating around its seam (see `repeated`).
    """
    n = len(d)
    pad = max(-min(SEAM_OFFSETS), max(SEAM_OFFSETS) + 1)
    padded = np.full((n + 2 * pad, n + 2 * pad), np.nan, dtype=np.float32)
    padded[pad:pad + n, pad:pad + n] = d
    error = np.zeros((n, n + 1), dtype=np.float32)
    weight = np.zeros((n, n + 1), dtype=np.float32)
    pairs = np.zeros((n, n + 1), dtype=np.int8)
    for k, w in zip(SEAM_OFFSETS, SEAM_WEIGHTS):
        block = padded[pad + k:pad + k + n, pad + k:pad + k + n + 1]
        ok = ~np.isnan(block)
        error += np.where(ok, block, 0.0) * w
        weight += ok * w
        pairs += ok
    del padded
    error /= np.maximum(weight, 1e-12)
    del weight

    length = np.arange(n + 1, dtype=np.int32)[None, :] - np.arange(n, dtype=np.int32)[:, None]

    # Mean distance from the start frame to the frames inside the loop.
    row_sum = np.zeros((n, n + 1), dtype=np.float32)
    np.cumsum(d, axis=1, out=row_sum[:, 1:])
    after_start = row_sum[np.arange(n), np.minimum(np.arange(n) + 1, n)][:, None]
    excursion = (row_sum - after_start) / np.maximum(length - 1, 1)
    del row_sum

    steps = np.diagonal(d, 1)
    if len(steps) == 0:
        steps = np.zeros(1, dtype=np.float32)
    idx = np.arange(n + 1)[:, None] + np.arange(-2, 2)[None, :]
    near = np.where((idx >= 0) & (idx < len(steps)), steps[np.clip(idx, 0, len(steps) - 1)], np.nan)
    motion = np.nanmean(near, axis=1).astype(np.float32)

    # Exactly repeated frames carry no noise to measure.
    moving = steps[steps > 0]
    noise = float(np.percentile(moving, 5)) if len(moving) else np.inf
    valid = (length >= 2) & (pairs >= 2) & (error <= MAX_RETURN_RATIO * excursion)
    valid &= excursion >= MIN_MOTION_OVER_NOISE * noise
    del excursion, pairs
    after, before = repeated(error, steps, noise, length)
    error /= 0.5 * (motion[:n, None] + motion[None, :]) + 1e-7
    return error, valid, after, before, length


def repeated(error: np.ndarray, steps: np.ndarray, noise: float, length: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """How far each loop's period keeps repeating around its seam.

    Frame i matches frame i + period when they differ less than the average
    frame step over half a period around i. Holding drawings or varying from
    cycle to cycle does not break that, a different motion does. Returns, for
    every (start, repeat), how many consecutive matches along the period run
    from the seam onwards (`after`) and back to it (`before`, counting the
    seam itself).
    """
    n = error.shape[0]
    period = np.maximum(length, 1)
    sums = np.concatenate([[0.0], np.cumsum(steps, dtype=np.float64)]).astype(np.float32)
    half = np.maximum(period // 2, 1)
    i = np.arange(n)[:, None]
    lo = np.clip(i - half, 0, len(steps))
    hi = np.clip(i + half, 0, len(steps))
    motion = (sums[hi] - sums[lo]) / np.maximum(hi - lo, 1)
    match = (error <= motion + noise) & (length >= 1)
    match[:, n] = False  # frame n does not exist
    # Generated cycles drift by a frame and drawings are held irregularly: a
    # match one frame off the period still continues the repetition, and a
    # few missed frames between matches do not break it.
    drift = match.copy()
    drift[:, 1:] |= match[:, :-1]
    drift[:, :-1] |= match[:, 1:]
    match = drift.copy()
    for gap in range(1, MAX_REPEAT_GAP + 1):
        for k in range(gap):
            # Frame k + 1 frames after a match and gap - k frames before one.
            a, b = k + 1, gap - k
            match[a:-b, a:-b] |= drift[:-a - b, :-a - b] & drift[a + b:, a + b:]
    forward = np.zeros((n + 1, n + 2), dtype=np.int32)
    for row in range(n - 1, -1, -1):
        forward[row, :n + 1] = match[row] * (1 + forward[row + 1, 1:n + 2])
    backward = np.zeros((n + 1, n + 2), dtype=np.int32)
    for row in range(n):
        backward[row + 1, 1:n + 2] = match[row] * (1 + backward[row, :n + 1])
    after = forward[:n, :n + 1]
    before = backward[1:, 1:]
    # A loop running to the last frame has its seam pair one frame earlier.
    after[1:, n] = after[:-1, n - 1]
    before[1:, n] = before[:-1, n - 1] + 1
    return after, before


def span(start: int, repeat: int, after: np.ndarray, before: np.ndarray) -> tuple[int, int]:
    """Frames the loop's repetition spans, at least the loop itself."""
    lo = start - max(int(before[start, repeat]) - 1, 0)
    hi = max(repeat, repeat + int(after[start, repeat]))
    return lo, hi


def find_loops(path: str, every_loop: bool = False) -> list[Loop]:
    """The best loop of the video, or the best loop of every repeating part."""
    video = decode(path)
    n = len(video.frames)
    if n < 3:
        return []
    d = distances(video.frames)
    seam, valid, after, before, length = candidates(d)
    # A loop is shown when the video plays its cycle at least twice in a row:
    # the repetition through its seam spans a whole period, give or take the
    # frames a repetition may miss.
    shown = after + before - 1 + np.minimum(MAX_REPEAT_GAP, length // 4) >= length
    start = np.arange(n)[:, None]
    repeat = np.arange(n + 1)[None, :]
    pool = valid & (seam <= MAX_SEAM)
    loops = []
    while pool.any():
        # Loops the video demonstrates come first; a loop that only closes
        # once (e.g. idle, action, back to idle) is a fallback.
        choice = pool & shown if (pool & shown).any() else pool
        s0, r0 = np.unravel_index(int(np.argmin(np.where(choice, seam, np.inf))), seam.shape)
        lo, hi = span(int(s0), int(r0), after, before)
        same = choice & (start >= lo) & (repeat <= hi) & (seam <= seam[s0, r0] * SEAM_TOLERANCE + SEAM_SLACK)
        # The shortest cycle, then the best seam among its variants a frame or
        # so longer: those are the same cycle, not a multiple of it.
        shortest = int(length[same].min())
        cycle = same & (length <= shortest + max(1, shortest // 8))
        s, r = np.unravel_index(int(np.argmin(np.where(cycle, seam, np.inf))), seam.shape)
        s, r = int(s), int(r)
        if not shown[s0, r0]:
            # A loop that closes only once spans just itself; the shorter loop
            # inside it is all that is taken.
            lo, hi = span(s, r, after, before)
        loops.append(Loop(
            start=s, end=r - 1, frames=r - s,
            start_time=round(float(s / video.fps), 6),
            duration=round(float((r - s) / video.fps), 6),
            seam=round(float(seam[s, r]), 4),
        ))
        if not every_loop:
            break
        # Later loops come from the rest of the video.
        pool &= (repeat <= lo) | (start >= hi)
    return sorted(loops, key=lambda loop: loop.start)


def cut(path: str, loop: Loop, out_path: str) -> None:
    """Write exactly the loop's frames to a new video at the source frame rate."""
    fps = probe(path)[2]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", path, "-map", "0:v:0",
         "-vf", f"trim=start_frame={loop.start}:end_frame={loop.end + 1},setpts=N/({fps})/TB",
         "-r", str(fps), "-fps_mode", "cfr", "-an", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
        check=True,
    )
