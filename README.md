# loopex

Finds seamless loops in videos. One command, no tuning: no minimum length, no
thresholds, no modes.

```bash
pip install -e .            # needs ffmpeg/ffprobe on PATH

loopex video.mp4            # the best loop
loopex video.mp4 --all      # the best loop of every repeating part (e.g. idle, walk, attack)
loopex video.mp4 -o out/    # also write each loop as a frame-exact H.264 clip
```

Output (exit code 1 when no loop is found):

```json
{
  "video": "video.mp4",
  "loops": [
    {"start": 57, "end": 72, "frames": 16, "start_time": 2.375, "duration": 0.666667, "seam": 0.1621,
     "file": "out/video_loop1.mp4"}
  ]
}
```

- `start`/`end`: first and last frame of the loop, inclusive, counted from 0.
- `seam`: the jump where the loop restarts, measured in the video's own typical
  frame-to-frame steps. Near 0 it is invisible; around 1 it reads as one
  skipped frame; above ~2 it is a visible pop.

From Python:

```python
from loopex import find_loops, cut

loops = find_loops("video.mp4")                  # or every_loop=True
cut("video.mp4", loops[0], "loop.mp4")
```

## How it works

Every frame becomes a 128 px grayscale thumbnail, and every pair of frames gets
a distance. A loop `[start, repeat)` is seamless when the frames around
`repeat` match the frames around `start`. The frames on both sides of the seam
are compared too, so the seam also matches the direction of motion: a
ping-pong turn point does not pass for a seam.

Every candidate is judged by two ratios, so nothing depends on resolution,
brightness or how fast things move:

- **return ratio**: seam error over how far the content travels away from the
  start during the loop. Real loops come back much closer than they travel.
  Two- or three-frame "loops" don't, which is what the usual minimum-length
  setting tries to prevent.
- **seam**: seam error in typical frame steps around the seam (the reported
  value).

Among the loops whose seam is equivalent to the best one, the shortest wins.
That is the fundamental cycle; whole multiples of it add nothing. With
`--all`, the frames of the motion a chosen loop repeats are set aside, meaning
every frame the loop already contains a match for, and the search continues
in the rest.

Held drawings (animation on twos and threes), period drift between generated
cycles and compression noise are handled by the same measures. There is no
special path for them.

## Benchmark

`bench/bench.py` builds test videos with known loops from real footage
(exact repeats, AI-like varied repeats, animation on twos/threes, ping-pong,
two actions back to back, and loop-free negatives), encodes them as H.264,
runs the detector on the files and checks every answer against the ground
truth. It also checks that written clips hold exactly the reported frames.

```bash
python bench/bench.py bench-out/ path/to/source/*.mp4   # writes bench-out/report.md and report.json
python bench/baseline.py bench-out/ path/to/dodi_frame/backend/app/loop_detection.py
```
