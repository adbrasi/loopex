"""End-to-end loop detection benchmark with ground truth.

Builds test videos from real footage whose loops are known by construction,
encodes them as H.264, runs the detector on the files and scores the answers.
Every built frame remembers which source frame it shows, so a loop
[start, start + frames) is exact when frame start + frames shows the same
source frame as frame start: playback continues exactly where it jumps to.

Usage: python bench/bench.py OUT_DIR SOURCE_VIDEO...
Writes OUT_DIR/report.json and OUT_DIR/report.md. The same sources and seed
give the same videos and report. Only one source is held in memory at a time.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopex import Loop, cut, find_loops  # noqa: E402

SIDE = 256
MAX_SOURCE_FRAMES = 360
FPS = 24
SEED = 7
CASES_PER_KIND = 24
KINDS = ("exact", "varied", "twos", "pingpong", "multi")


@dataclass
class Case:
    name: str
    kind: str
    ids: list  # source identity of every frame
    loops: list = field(default_factory=list)  # [(lo, hi, period)] repeating sections


def load(path: str) -> np.ndarray:
    probe = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", path]))
    w0, h0 = probe["streams"][0]["width"], probe["streams"][0]["height"]
    s = SIDE / max(w0, h0)
    w, h = round(w0 * s) // 2 * 2, round(h0 * s) // 2 * 2
    raw = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", path, "-map", "0:v:0", "-frames:v", str(MAX_SOURCE_FRAMES),
         "-vf", f"scale={w}:{h}:flags=area", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def thumb_d(frames: np.ndarray) -> np.ndarray:
    x = frames[:, ::8, ::8].astype(np.float32).mean(axis=3).reshape(len(frames), -1) / 255
    sq = (x * x).sum(1)
    return np.maximum(sq[:, None] + sq[None] - 2 * x @ x.T, 0) / x.shape[1]


def segment(rng, video: np.ndarray, period: int) -> int | None:
    """Start of a segment that moves and repeats at no shorter lag.

    Every shorter lag is checked at its best phase, the way a detector would
    see it, so the segment's period is its only fundamental cycle.
    """
    if len(video) <= period + 1:
        return None
    for _ in range(40):
        a = int(rng.integers(0, len(video) - period))
        d = thumb_d(video[a:a + period])
        steps = np.diagonal(d, 1)
        if steps.mean() < 2e-4:
            continue
        i = np.arange(period)
        # Error of every shorter lag at its best phase, matching each frame's
        # neighbors too, as the detector's seam does.
        lag_errors = [
            np.mean([d[(i + k) % period, (i + k + lag) % period] for k in (-1, 0, 1)], axis=0).min()
            for lag in range(2, period - 1)
        ]
        best_lag_error = min(lag_errors, default=np.inf)
        if best_lag_error > steps.mean():
            return a
    return None


def section(rng, video, src, kind):
    """Frames and ids of one repeating section, and its fundamental period."""
    period = int(rng.integers(8, 61))
    a = segment(rng, video, period)
    if a is None:
        return None
    order = list(range(a, a + period))
    if kind == "twos":
        order = [i for i in order for _ in range(int(rng.integers(2, 4)))]
    if kind == "pingpong":
        order = order + order[-2:0:-1]
    frames, ids = [], []
    for _ in range(int(rng.integers(2, 5))):
        cycle = video[order]
        if kind == "varied":
            # Generated motion never repeats exactly: per-cycle exposure,
            # sub-pixel drift and fresh noise.
            c = cycle.astype(np.float32) * np.float32(rng.uniform(0.98, 1.02)) + np.float32(rng.uniform(-3, 3))
            c = 0.5 * (c + np.roll(c, int(rng.choice([-1, 1])), axis=int(rng.choice([1, 2]))))
            c += rng.standard_normal(c.shape, dtype=np.float32) * 2
            cycle = np.clip(c, 0, 255).astype(np.uint8)
        frames.append(cycle)
        ids += [(src, i) for i in order]
    return np.concatenate(frames), ids, len(order), a


def build(rng, src: int, video: np.ndarray, kind: str, name: str):
    """One test case of the given kind from one source, or None if it has no usable segment."""
    if kind == "multi":
        # Two different repeating actions of one video back to back.
        s1, s2 = section(rng, video, src, "exact"), section(rng, video, src, "exact")
        if s1 is None or s2 is None:
            return None
        if abs(s1[3] - s2[3]) < max(s1[2], s2[2]) + 30:
            return None  # the two actions must be different footage
        n1 = len(s1[0])
        return np.concatenate([s1[0], s2[0]]), Case(name, kind, s1[1] + s2[1], [(0, n1, s1[2]), (n1, n1 + len(s2[0]), s2[2])])
    sec = section(rng, video, src, kind)
    if sec is None:
        return None
    frames, ids, period, a = sec
    # Unrelated footage of the same video before and after the loop, taken
    # away from the looped segment.
    pre, post = int(rng.integers(0, 25)), int(rng.integers(0, 25))
    clear = [f for f in range(0, len(video) - pre - post) if f + pre + post + 30 <= a or f >= a + period + 30]
    if not clear:
        return None
    far = int(rng.choice(clear))
    ctx = video[far:far + pre + post]
    all_ids = [(src, i, "ctx") for i in range(far, far + len(ctx))]
    return (np.concatenate([ctx[:pre], frames, ctx[pre:]]),
            Case(name, kind, all_ids[:pre] + ids + all_ids[pre:], [(pre, pre + len(frames), period)]))


def negatives(rng, video: np.ndarray, k: int):
    """A camera pan across a still and a frozen noisy frame: neither loops."""
    still = video[len(video) // 2]
    h, w = still.shape[:2]
    big = np.concatenate([still, still[:, ::-1]], axis=1)
    n = 150
    pan = np.stack([big[:, int(i * w / n):int(i * w / n) + w] for i in range(n)])
    yield pan, Case(f"pan{k:02d}", "none", [("pan", k, i) for i in range(n)])
    noise = rng.standard_normal((n, h, w, 3), dtype=np.float32) * 2
    frozen = np.clip(still[None].astype(np.float32) + noise, 0, 255).astype(np.uint8)
    yield frozen, Case(f"frozen{k:02d}", "none", [("frozen", k, i) for i in range(n)])


def encode(frames: np.ndarray, path: Path) -> None:
    n, h, w, _ = frames.shape
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
                    "-i", "-", "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)],
                   input=frames.tobytes(), check=True)


def exact(case: Case, loop: Loop) -> bool:
    """The loop's seam continues exactly and it is the fundamental cycle.

    With cycles that vary, no cycle is the fundamental one: any whole number
    of cycles whose seam continues exactly is right.
    """
    for lo, hi, period in case.loops:
        if lo <= loop.start and loop.start + loop.frames <= hi:
            s, r = loop.start, loop.start + loop.frames
            pairs = [(s + k, r + k) for k in (-1, 0) if lo <= s + k and r + k < hi]
            cycles_ok = loop.frames % period == 0 if case.kind == "varied" else loop.frames == period
            return bool(pairs) and all(case.ids[a] == case.ids[b] for a, b in pairs) and cycles_ok
    return False


def frame_count(path: str) -> int:
    return int(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                        "stream=nb_read_frames", "-of", "csv=p=0", path]).strip())


def main() -> None:
    out = Path(sys.argv[1])
    videos = out / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    paths = sorted(sys.argv[2:])
    rng = np.random.default_rng(SEED)
    cases = []
    for src, path in enumerate(paths):
        video = load(path)
        for kind in KINDS:
            for i in range(src, CASES_PER_KIND, len(paths)):
                for _ in range(5):
                    made = build(rng, src, video, kind, f"{kind}{i:02d}")
                    if made:
                        encode(made[0], videos / f"{made[1].name}.mp4")
                        cases.append(made[1])
                        break
        if src < CASES_PER_KIND // 8:
            for frames, case in negatives(rng, video, src):
                encode(frames, videos / f"{case.name}.mp4")
                cases.append(case)
        del video
        print(f"built cases from {Path(path).name}", flush=True)

    rows = []
    for case in cases:
        path = str(videos / f"{case.name}.mp4")
        t = time.perf_counter()
        best = find_loops(path)
        seconds = time.perf_counter() - t
        every = find_loops(path, every_loop=True)
        if case.kind == "none":
            ok = not best and not every
        elif case.kind == "multi":
            ok = len(every) == 2 and all(exact(case, l) for l in every) and bool(best) and exact(case, best[0])
        else:
            ok = bool(best) and exact(case, best[0])
        rows.append({
            "case": case.name, "kind": case.kind, "frames": len(case.ids), "ok": ok, "seconds": round(seconds, 3),
            "truth": [{"from": lo, "to": hi - 1, "period": p} for lo, hi, p in case.loops],
            "best": [l.to_dict() for l in best], "all": [l.to_dict() for l in every],
        })
        print(f"{case.name:10s} {'ok ' if ok else 'BAD'} truth={[(lo, hi - 1, p) for lo, hi, p in case.loops]} "
              f"found={[(l.start, l.frames, l.seam) for l in every]}", flush=True)

    # The written loop must hold exactly the reported frames.
    cuts = []
    (out / "cuts").mkdir(exist_ok=True)
    for row in rows[::5]:
        if row["best"]:
            target = str(out / "cuts" / f"{row['case']}.mp4")
            cut(str(videos / f"{row['case']}.mp4"), Loop(**row["best"][0]), target)
            cuts.append({"case": row["case"], "expected": row["best"][0]["frames"], "written": frame_count(target)})

    kinds = {}
    for row in rows:
        k = kinds.setdefault(row["kind"], [0, 0])
        k[0] += row["ok"]
        k[1] += 1
    summary = {
        "cases": len(rows), "correct": sum(r["ok"] for r in rows),
        "by_kind": {k: f"{a}/{b}" for k, (a, b) in kinds.items()},
        "mean_seconds": round(float(np.mean([r["seconds"] for r in rows])), 3),
        "cuts_exact": f"{sum(c['expected'] == c['written'] for c in cuts)}/{len(cuts)}",
    }
    (out / "report.json").write_text(json.dumps({"summary": summary, "cases": rows, "cuts": cuts}, indent=1))
    lines = ["# loopex benchmark", "", f"Sources: {len(paths)} videos, seed {SEED}.", "", "| kind | correct |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary["by_kind"].items()]
    lines += ["", f"Total: {summary['correct']}/{summary['cases']}. Mean detection time {summary['mean_seconds']}s per video.",
              f"Written loops with exactly the reported frames: {summary['cuts_exact']}.", "", "## Failures", ""]
    lines += [f"- {r['case']}: truth {r['truth']}, best {[(l['start'], l['frames'], l['seam']) for l in r['best']]}, "
              f"all {[(l['start'], l['frames'], l['seam']) for l in r['all']]}" for r in rows if not r["ok"]] or ["None."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
