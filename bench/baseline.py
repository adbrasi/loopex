"""Score the dodi_frame loop detector on a finished benchmark run.

Usage: python bench/baseline.py BENCH_DIR PATH_TO/dodi_frame/backend/app/loop_detection.py
Runs its "structural" and "motion" methods with the default minimum of 12 frames
on the single-loop cases and writes BENCH_DIR/baseline.json.
"""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

out = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("dodi", sys.argv[2])
dodi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dodi)


def frames(path: str) -> list:
    probe = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", path]))
    w, h = probe["streams"][0]["width"], probe["streams"][0]["height"]
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return [Image.fromarray(f) for f in np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)]


def descriptor(image, size):
    # Same as dodi's _frame_descriptor("structural"), fed from memory.
    path = out / "_tmp.png"
    image.save(path)
    return dodi._frame_descriptor(path, "structural", size)


report = json.loads((out / "report.json").read_text())
results = {}
for method in ("structural", "motion"):
    score, total, seconds = {}, {}, []
    for row in report["cases"]:
        if row["kind"] in ("none", "multi"):
            continue
        imgs = frames(str(out / "videos" / f"{row['case']}.mp4"))
        t = time.perf_counter()
        desc = np.stack([descriptor(im, 28) for im in imgs])
        s, r, _, _ = dodi._best_pair(desc, 12, method, None)
        seconds.append(time.perf_counter() - t)
        truth = row["truth"][0]
        frames_n = r - s
        ok = frames_n == truth["period"] and truth["from"] <= s and r - 1 <= truth["to"] and (r <= truth["to"] or s > truth["from"])
        score[row["kind"]] = score.get(row["kind"], 0) + ok
        total[row["kind"]] = total.get(row["kind"], 0) + 1
    results[method] = {k: f"{score[k]}/{total[k]}" for k in total} | {"mean_seconds_excluding_decode": round(float(np.mean(seconds)), 3)}
    print(method, results[method], flush=True)
(out / "_tmp.png").unlink(missing_ok=True)
(out / "baseline.json").write_text(json.dumps(results, indent=1))
