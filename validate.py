"""Cross-check the tripwire against each track's centroid path.

Runs the same pipeline as sightline.py with the same options, but also records
where every confirmed track's box center was on each frame, then derives
crossings from those paths independently of LineZone and prints where the two
disagree. A centroid crossing with no tripwire event is a candidate miss; a
tripwire event with no centroid crossing is a candidate false positive.

    python validate.py --source clip.mp4 --line 0,540,1920,540 --classes person
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Sequence

import supervision as sv

from sightline import Counter, build_counter, die, parse_args, parse_line

Point = tuple[float, float]
Path = list[tuple[int, Point]]


def side_of_line(start: Point, end: Point, point: Point) -> str | None:
    """'in' if the point is on the line's left looking from start to end, 'out' on
    the right, None if it is on the line. Matches LineZone's convention (image
    coordinates, y down): for a left-to-right line, above the line is 'in'."""
    (sx, sy), (ex, ey), (px, py) = start, end, point
    cross = (ex - sx) * (py - sy) - (ey - sy) * (px - sx)
    if cross == 0:
        return None
    return "in" if cross < 0 else "out"


def centroid_crossings(path: Path, start: Point, end: Point) -> list[tuple[int, str]]:
    """Direction changes of a track's center, ignoring frames exactly on the line."""
    crossings = []
    side = None
    for frame, point in path:
        now = side_of_line(start, end, point) or side
        if side is not None and now is not None and now != side:
            crossings.append((frame, now))
        side = now
    return crossings


def fmt(crossings: list[tuple[int, str]]) -> str:
    return " ".join(f"{f}:{d}" for f, d in crossings) or "-"


def collect(counter: Counter, source: str, max_frames: int | None) -> dict[int, Path]:
    paths: dict[int, Path] = defaultdict(list)
    for index, frame in enumerate(sv.get_video_frames_generator(source)):
        if max_frames is not None and index >= max_frames:
            break
        tracked = counter.step(frame, index)
        for (x1, y1, x2, y2), tid in zip(tracked.xyxy, tracked.tracker_id, strict=True):
            paths[int(tid)].append((index, ((x1 + x2) / 2, (y1 + y2) / 2)))
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not os.path.isfile(args.source):
        die(f"source file not found: {args.source}")
    info = sv.VideoInfo.from_video_path(args.source)
    counter = build_counter(args, info.width, info.height, info.fps)
    x1, y1, x2, y2 = parse_line(args.line, info.width, info.height)
    start, end = (float(x1), float(y1)), (float(x2), float(y2))

    paths = collect(counter, args.source, args.max_frames)
    zone: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for e in counter.events:
        zone[e.tracker_id].append((e.frame, e.direction))
    centroid = {
        tid: c for tid, path in paths.items() if (c := centroid_crossings(path, start, end))
    }

    total_in = sum(d == "in" for c in centroid.values() for _, d in c)
    total_out = sum(d == "out" for c in centroid.values() for _, d in c)
    print(f"clip: {info.total_frames} frames @ {info.fps:.0f} fps, line {start} -> {end}")
    print(f"confirmed tracks: {len(paths)}")
    print(
        f"tripwire ({args.anchor}): "
        f"in={counter.line_zone.in_count} out={counter.line_zone.out_count}"
    )
    print(f"centroid path:      in={total_in} out={total_out}")

    disagreements = 0
    print("\nper track (frame:direction):")
    for tid in sorted(set(centroid) | set(zone)):
        z, c = zone.get(tid, []), centroid.get(tid, [])
        flag = "" if len(z) == len(c) else "  <-- differs"
        disagreements += bool(flag)
        print(f"  #{tid:<4} tripwire[{fmt(z)}]  centroid[{fmt(c)}]{flag}")
    print(f"\n{disagreements} track(s) differ; review those frames before trusting the count.")


if __name__ == "__main__":
    main()
