import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .core import cut, find_loops


def fail(message: str) -> int:
    print(f"loopex: {message}", file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(prog="loopex", description="Find seamless loops in a video.")
    parser.add_argument("video")
    parser.add_argument("--all", action="store_true", help="the best loop of every repeating part, not just the best one")
    parser.add_argument("-o", "--output", metavar="DIR", help="also write each loop as a frame-exact video into DIR")
    args = parser.parse_args()

    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        return fail(f"{' and '.join(missing)} not found on PATH; install FFmpeg")
    if not Path(args.video).is_file():
        return fail(f"no such file: {args.video}")

    try:
        loops = find_loops(args.video, every_loop=args.all)
        result = []
        for i, loop in enumerate(loops, 1):
            item = loop.to_dict()
            if args.output:
                out_dir = Path(args.output)
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / f"{Path(args.video).stem}_loop{i}.mp4"
                cut(args.video, loop, str(out))
                item["file"] = str(out)
            result.append(item)
    except subprocess.CalledProcessError as error:
        stderr = error.stderr or ""
        detail = (stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr).strip().splitlines()
        return fail(f"ffmpeg could not read {args.video}" + (f": {detail[-1]}" if detail else ""))
    json.dump({"video": args.video, "loops": result}, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
