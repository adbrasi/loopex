# Benchmark results

105 H.264 test videos built from 19 real videos (7 sprite-style loop videos,
12 general AI generations), seed 7. Run: `python bench/bench.py OUT sources...`.

| kind | loopex | dodi_frame structural (min 12) | dodi_frame motion (min 12) |
|---|---|---|---|
| exact repeats | 23/23 | 15/23 | 14/23 |
| AI-like varied repeats | 19/22 | 9/22 | 11/22 |
| animation on twos/threes | 22/22 | 20/22 | 20/22 |
| ping-pong | 24/24 | 23/24 | 23/24 |
| two actions (`--all`) | 8/8 | n/a | n/a |
| no loop (pan, frozen) | 6/6 | n/a | n/a |

Mean detection time: loopex 0.26 s per video including decoding;
dodi_frame 1.5 s excluding decoding. Written clips held exactly the reported
frame count in 21/21 checks.

The 3 varied failures are loops that reach 1-2 frames into the surrounding
footage or a genuine shorter loop found in it.
