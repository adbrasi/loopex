# loopex

Finds seamless loops in videos. One command, no tuning: no minimum loop
length, no thresholds, no modes.

```bash
loopex video.mp4
```

```json
{
  "video": "video.mp4",
  "loops": [
    {"start": 57, "end": 72, "frames": 16, "start_time": 2.375, "duration": 0.666667, "seam": 0.1621}
  ]
}
```

Built for short generated clips (5–15 s): sprite animations with several
actions, character idles, live wallpapers. It also works on regular footage.
A typical 10 s clip takes 0.3–0.5 s, and almost all of that is decoding.

- [Install](#install)
- [Command line](#command-line)
- [Python](#python)
- [Output](#output)
- [How it decides](#how-it-decides)
- [Speed and memory](#speed-and-memory)
- [Limitations](#limitations)
- [Benchmark](#benchmark)
- [Development](#development)

## Install

Requirements: Python 3.10+ and FFmpeg, with both `ffmpeg` and `ffprobe` on
`PATH`. The only Python dependency is NumPy.

```bash
sudo apt install ffmpeg                         # Debian/Ubuntu; use your platform's FFmpeg elsewhere
pip install git+https://github.com/adbrasi/loopex.git
```

From a local checkout:

```bash
git clone https://github.com/adbrasi/loopex.git
pip install ./loopex
```

## Command line

```bash
loopex video.mp4                  # the best loop of the video
loopex video.mp4 --all            # the best loop of every repeating part (idle, walk, attack, ...)
loopex video.mp4 -o out/          # also write each loop as a clip into out/
loopex video.mp4 --all -o out/    # every loop, one clip per loop
```

| option | meaning |
|---|---|
| `--all` | Return one loop per repeating part of the video instead of only the best one. Use it for videos that hold several actions. |
| `-o DIR` | Also write each loop to `DIR/<video name>_loop<N>.mp4`. Clips are H.264 (CRF 16, yuv420p) at the source frame rate, without audio, and hold exactly the reported frames. |

Exit codes: `0` loops found, `1` no loop found, `2` error (missing file,
unreadable video, FFmpeg not installed), with the reason on stderr.

## Python

Calling the library from a long-running worker avoids starting Python for
every video.

```python
from loopex import find_loops, cut

loops = find_loops("video.mp4")                     # [Loop] with the best loop, or []
every = find_loops("video.mp4", every_loop=True)     # one Loop per repeating part, in video order

if loops:
    best = loops[0]
    print(best.start, best.end, best.frames, best.seam)
    cut("video.mp4", best, "loop.mp4")               # frame-exact H.264 clip
```

`Loop` is a frozen dataclass; `loop.to_dict()` gives the JSON form.

## Output

| field | meaning |
|---|---|
| `start`, `end` | First and last frame of the loop, inclusive, counted from 0 in decode order. |
| `frames` | `end - start + 1`. |
| `start_time`, `duration` | The same in seconds, from the video's average frame rate. |
| `seam` | How visible the jump back to the start is, in the video's own typical frame-to-frame steps. |
| `file` | Written clip, only with `-o`. |

Reading `seam`:

| seam | looks like |
|---|---|
| < 0.3 | invisible |
| ~ 1 | one skipped frame |
| > 2 | a visible pop |

Loops above 3 are never returned. Gate on `seam` if your product needs a
stricter bar.

## How it decides

Every frame becomes a 128 px grayscale thumbnail, and every pair of frames gets
a distance. A loop `[start, repeat)` is played over and over. It is seamless
when the frames around `repeat` match the frames around `start`. Frames on both
sides of the seam are compared, so the motion direction has to match too: the
turning point of a back-and-forth motion does not pass for a seam.

Every judgement is a ratio of the video against itself, so nothing depends on
resolution, brightness, compression or how fast things move.

1. **The loop must actually go somewhere.** The seam error is compared with how
   far the content travels away from the start frame during the loop. A real
   loop comes back much closer than it travels. Two- or three-frame "loops"
   don't, and neither do frozen frames or camera pans. This replaces the usual
   minimum-length setting.
2. **Loops the video demonstrates come first.** If the video plays a cycle at
   least twice in a row, that cycle beats a loop that only closes once. For
   example, "idle, walk, back to idle" closes once, while the walk cycle
   repeats. Following a repetition tolerates generated cycles drifting by a
   frame, drawings held for 2–3 frames (animation on twos/threes) and a few
   imperfect frames.
3. **Best seam, then the fundamental cycle.** The best seam wins. Within its
   repetition, the shortest loop with an equivalent seam is returned: whole
   multiples of a cycle add nothing.
4. **With `--all`**, the frames that a chosen loop's repetition spans are set
   aside and the search continues on the rest, so each action yields one loop.

When nothing repeats, as in a clip that only drifts back near its first frame,
the best loop that closes once is returned with its honest `seam`.

## Speed and memory

Measured on a 6-core WSL2 machine:

| video | frames | time |
|---|---|---|
| 864×480, 24 fps, 5 s | 124 | 0.30 s |
| 768×1376, 24 fps, 10 s | 243 | 0.47 s |
| 1056×608, 24 fps, 15 s | 362 | 0.53 s |
| 1280×720, 60 fps, 33 s | 2020 | 3.2 s, 277 MB peak |

Decoding is about 95% of the time; the analysis takes around 10 ms on these
clips. Everything runs on the CPU. GPU decoding was slower for clips
this short, because initialisation outweighs the decode. To raise throughput,
run several worker processes that call `find_loops` directly.

The analysis keeps a few frame-by-frame matrices, so memory grows with the
square of the frame count: 277 MB at 2020 frames, a few tens of MB for typical
clips.

## Limitations

- **Held drawings:** inside drawings held for many identical frames, a loop
  may come out one frame shorter. That frame is identical in pixels, so only
  the timing of that hold changes.
- **Integer periods:** loops are whole frames. A motion whose true period
  falls between frames keeps a small seam.
- **Channels:** alpha is ignored, and audio is not part of the analysis or
  the written clips.
- **Single loops:** a loop that only closes once, such as a character standing
  up and kneeling back down, is returned when the video has no repetition to
  prefer. That is the right call for wallpapers. For sprite sheets, check
  `seam` and the duration.

## Benchmark

`bench/bench.py` builds H.264 test videos with known loops from real footage,
runs the detector on the files and checks every answer against the ground
truth. Each built frame remembers which source frame it shows, so correctness
is exact rather than judged. The cases are:

- exact repeats, and AI-like repeats with exposure, drift and noise varying per cycle;
- animation on twos/threes and ping-pong motion;
- two different actions back to back (`--all`);
- loop-free negatives: a camera pan and a frozen noisy frame.

It also checks that written clips hold exactly the reported number of frames.

```bash
python bench/bench.py bench-out/ sources/*.mp4                      # writes bench-out/report.md and report.json
python bench/baseline.py bench-out/ path/to/other/loop_detection.py # scores a previous detector on the same videos
```

Latest results are in [bench/RESULTS.md](bench/RESULTS.md): 102 of 105 cases
correct, against 67–68 of 91 single-loop cases for the previous detector.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/loopex some.mp4 --all
```

All logic is in `loopex/core.py`; the CLI is `loopex/__main__.py`. Change the
detector against the benchmark: it builds its cases from whatever source videos
you pass, holds one source in memory at a time and is deterministic for the
same sources.
