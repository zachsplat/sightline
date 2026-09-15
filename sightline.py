"""Sightline: count people and vehicles crossing a tripwire in camera footage.

Detection (Roboflow Inference) -> tracking (Roboflow trackers, ByteTrack) ->
tripwire counts (supervision LineZone) -> a timestamped crossing log, per-class
totals, and an optional annotated video.

File mode (a recorded clip):
    python sightline.py --source clip.mp4 --line 0,540,1920,540 --classes person,car

Live mode (an RTSP camera, or a webcam index):
    python sightline.py --source rtsp://user:pass@192.168.1.20/stream --classes person

Direction: a crossing is "in" when the object ends up on the LEFT of the line
as you look from its start point toward its end point. For the default
left-to-right horizontal line, bottom-to-top is "in". Swap the endpoints to
flip it. Each direction change of a track counts once, so a track that turns
around on the line counts once in and once out. tests/test_sightline.py pins
this down.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

# inference imports every optional model family at import time and warns about
# each one this tool never loads. Turn them off before the import so a normal
# run prints counts, not a page of deprecation notices.
for _flag in (
    "CORE_MODEL_SAM_ENABLED",
    "CORE_MODEL_SAM2_ENABLED",
    "CORE_MODEL_SAM3_ENABLED",
    "CORE_MODEL_GAZE_ENABLED",
    "CORE_MODEL_YOLO_WORLD_ENABLED",
    "CORE_MODEL_CLIP_ENABLED",
    "CORE_MODEL_OWLV2_ENABLED",
    "CORE_MODEL_GROUNDINGDINO_ENABLED",
    "CORE_MODEL_TROCR_ENABLED",
    "CORE_MODEL_DOCTR_ENABLED",
    "CORE_MODEL_DEPTH_ESTIMATION_ENABLED",
    "CORE_MODEL_SMOLVLM2_ENABLED",
    "CORE_MODEL_MOONDREAM2_ENABLED",
    "CORE_MODEL_QWEN_2_5_ENABLED",
    "CORE_MODEL_PALIGEMMA_ENABLED",
    "CORE_MODEL_FLORENCE2_ENABLED",
):
    os.environ.setdefault(_flag, "False")
for _category in (FutureWarning, UserWarning, DeprecationWarning, SyntaxWarning):
    warnings.filterwarnings("ignore", category=_category)
# the same libraries also log at WARNING during import and model load
for _name in ("inference", "inference-models", "inference-models-verbose", "transformers", "timm"):
    logging.getLogger(_name).setLevel(logging.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np  # noqa: E402
import supervision as sv  # noqa: E402
from inference import InferencePipeline, get_model  # noqa: E402
from inference.core.interfaces.camera.entities import VideoFrame  # noqa: E402
from trackers import ByteTrackTracker  # noqa: E402

# inference-models resets its logger level while importing, so set it again here
for _name in ("inference-models", "inference-models-verbose"):
    logging.getLogger(_name).setLevel(logging.ERROR)

LIVE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")
# Streams do not always report a frame rate. 25 only affects the logged
# timestamps and ByteTrack's lost-track buffer, both of which degrade gracefully.
FALLBACK_FPS = 25.0
ANCHORS = {
    "center": (sv.Position.CENTER,),
    "corners": (
        sv.Position.TOP_LEFT,
        sv.Position.TOP_RIGHT,
        sv.Position.BOTTOM_LEFT,
        sv.Position.BOTTOM_RIGHT,
    ),
    "bottom": (sv.Position.BOTTOM_CENTER,),
}


class Detector(Protocol):
    """The slice of an inference model this tool relies on."""

    class_names: Sequence[str]

    def infer(self, image: np.ndarray, confidence: float) -> Sequence[object]: ...


def die(message: str) -> None:
    raise SystemExit(f"sightline: {message}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Count objects crossing a tripwire in a video file or live stream."
    )
    p.add_argument(
        "--source",
        required=True,
        help="Video file, rtsp:// or http(s):// stream URL, or a webcam index like 0.",
    )
    p.add_argument("--output", default="sightline_out.mp4", help="Annotated video (file mode).")
    p.add_argument("--no-video", action="store_true", help="Count and log only; skip the video.")
    p.add_argument("--events", default="events.csv", help="CSV log of every crossing.")
    p.add_argument("--model", default="yolov8n-640", help="Roboflow Inference model id.")
    p.add_argument(
        "--line",
        default=None,
        help="Tripwire as x1,y1,x2,y2 in pixels. Default: horizontal across the middle.",
    )
    p.add_argument("--classes", default="person", help="Class names to count, or 'all'.")
    p.add_argument("--confidence", type=float, default=0.3, help="Detection threshold, 0 to 1.")
    p.add_argument(
        "--anchor",
        choices=tuple(ANCHORS),
        default="center",
        help="Which point of the box must cross. center (default) counts a track that starts "
        "on the line; corners is supervision's default and more conservative; bottom is the feet.",
    )
    p.add_argument(
        "--min-crossing-frames",
        type=int,
        default=1,
        help="Frames a track must stay on the far side before it counts. Raise to ignore "
        "people loitering on the line.",
    )
    p.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames.")
    p.add_argument(
        "--live",
        action="store_true",
        help="Force the streaming path even for a file. Automatic for URLs and webcam indexes.",
    )
    return p.parse_args(argv)


def parse_line(line_arg: str | None, width: int, height: int) -> tuple[int, int, int, int]:
    if not line_arg:
        return 0, height // 2, width, height // 2
    parts = line_arg.split(",")
    if len(parts) != 4:
        die(f"--line needs four integers x1,y1,x2,y2, got {line_arg!r}")
    try:
        x1, y1, x2, y2 = (int(v) for v in parts)
    except ValueError:
        die(f"--line values must be integers, got {line_arg!r}")
    for x in (x1, x2):
        if not 0 <= x <= width:
            die(f"--line x={x} is outside the {width}px frame width")
    for y in (y1, y2):
        if not 0 <= y <= height:
            die(f"--line y={y} is outside the {height}px frame height")
    if (x1, y1) == (x2, y2):
        die("--line start and end are the same point")
    return x1, y1, x2, y2


def parse_classes(arg: str) -> set[str] | None:
    if arg.strip().lower() == "all":
        return None
    wanted = {c.strip().lower() for c in arg.split(",") if c.strip()}
    if not wanted:
        die("--classes is empty")
    return wanted


@dataclass
class Crossing:
    frame: int
    time_s: float
    tracker_id: int
    class_name: str
    direction: str


@dataclass
class Counter:
    """Everything that persists across frames: tracker, tripwire, annotators, log."""

    line_zone: sv.LineZone
    tracker: ByteTrackTracker
    fps: float
    wanted: set[str] | None
    confidence: float
    model: Detector
    events: list[Crossing] = field(default_factory=list)
    box: sv.BoxAnnotator = field(default_factory=sv.BoxAnnotator)
    label: sv.LabelAnnotator = field(default_factory=sv.LabelAnnotator)
    trace: sv.TraceAnnotator = field(default_factory=sv.TraceAnnotator)
    line: sv.LineZoneAnnotator = field(default_factory=lambda: sv.LineZoneAnnotator(text_scale=0.8))

    def filter_classes(self, detections: sv.Detections) -> sv.Detections:
        names = detections.data.get("class_name")
        if self.wanted is None or names is None or not len(detections):
            return detections
        return detections[np.array([str(n).lower() in self.wanted for n in names])]

    def track(self, detections: sv.Detections) -> sv.Detections:
        tracked = self.tracker.update(detections)
        if tracked.tracker_id is None:
            return sv.Detections.empty()
        # trackers' ByteTrack reports -1 for tracks it has not confirmed yet.
        # LineZone keys its state by tracker id, so one shared -1 would flip-flop
        # across the line and inflate the counts (it did: 97/95 on the sample).
        return tracked[tracked.tracker_id >= 0]

    def record(self, detections: sv.Detections, frame_index: int) -> list[Crossing]:
        """Run the tripwire on tracked detections and log any crossings."""
        crossed_in, crossed_out = self.line_zone.trigger(detections)
        time_s = round(frame_index / self.fps, 3) if self.fps else 0.0
        names = detections.data.get("class_name")
        new = [
            Crossing(
                frame=frame_index,
                time_s=time_s,
                tracker_id=int(detections.tracker_id[i]),
                class_name=str(names[i]) if names is not None else "object",
                direction=direction,
            )
            for direction, mask in (("in", crossed_in), ("out", crossed_out))
            for i in np.flatnonzero(mask)
        ]
        self.events.extend(new)
        return new

    def process(
        self, detections: sv.Detections, frame_index: int
    ) -> tuple[sv.Detections, list[Crossing]]:
        """The one pipeline both modes share: filter -> track -> tripwire."""
        tracked = self.track(self.filter_classes(detections))
        return tracked, self.record(tracked, frame_index)

    def step(self, frame: np.ndarray, frame_index: int) -> sv.Detections:
        """Detect on a raw frame, then process it. Returns the tracked detections."""
        result = self.model.infer(frame, confidence=self.confidence)[0]
        tracked, _ = self.process(sv.Detections.from_inference(result), frame_index)
        return tracked

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        names = detections.data.get("class_name")
        labels = [
            f"#{tid} {names[i] if names is not None else 'object'} {conf:.2f}"
            for i, (tid, conf) in enumerate(
                zip(detections.tracker_id, detections.confidence, strict=True)
            )
        ]
        out = self.trace.annotate(frame.copy(), detections)
        out = self.box.annotate(out, detections)
        out = self.label.annotate(out, detections, labels)
        return self.line.annotate(out, self.line_zone)

    def per_class(self) -> dict[str, tuple[int, int]]:
        totals: dict[str, list[int]] = {}
        for e in self.events:
            totals.setdefault(e.class_name, [0, 0])[0 if e.direction == "in" else 1] += 1
        return {k: (v[0], v[1]) for k, v in sorted(totals.items())}

    def by_interval(self, seconds: int) -> dict[int, tuple[int, int]]:
        """In/out totals per time bucket, keyed by the bucket's start second."""
        buckets: dict[int, list[int]] = {}
        for e in self.events:
            start = int(e.time_s // seconds) * seconds
            buckets.setdefault(start, [0, 0])[0 if e.direction == "in" else 1] += 1
        return {k: (v[0], v[1]) for k, v in sorted(buckets.items())}

    def write_events(self, path: str) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "time_s", "tracker_id", "class", "direction"])
            writer.writerows(
                (e.frame, e.time_s, e.tracker_id, e.class_name, e.direction) for e in self.events
            )

    def report(self) -> str:
        lines = [
            f"Crossed IN:  {self.line_zone.in_count}",
            f"Crossed OUT: {self.line_zone.out_count}",
        ]
        lines += [
            f"  {name:<12} in {i:>4}  out {o:>4}" for name, (i, o) in self.per_class().items()
        ]
        # an hourly rollup is what a site manager actually asks for; fall back to
        # minutes for short clips, and skip it when everything lands in one bucket
        span = max((e.time_s for e in self.events), default=0.0)
        bucket = 3600 if span >= 7200 else 60
        rollup = self.by_interval(bucket)
        if len(rollup) > 1:
            unit = "hour" if bucket == 3600 else "minute"
            lines.append(f"  per {unit}:")
            lines += [
                f"    {start // bucket:>4} {unit}  in {i:>4}  out {o:>4}"
                for start, (i, o) in rollup.items()
            ]
        return "\n".join(lines)


def build_counter(args: argparse.Namespace, width: int, height: int, fps: float) -> Counter:
    if not 0.0 <= args.confidence <= 1.0:
        die(f"--confidence must be between 0 and 1, got {args.confidence}")
    if args.min_crossing_frames < 1:
        die(f"--min-crossing-frames must be at least 1, got {args.min_crossing_frames}")
    wanted = parse_classes(args.classes)
    model: Detector = get_model(model_id=args.model)
    known = {str(n).lower() for n in getattr(model, "class_names", ())}
    if wanted is not None and known and (unknown := sorted(wanted - known)):
        die(
            f"model {args.model!r} has no class {', '.join(unknown)}; "
            f"it knows: {', '.join(sorted(known))}"
        )
    x1, y1, x2, y2 = parse_line(args.line, width, height)
    line_zone = sv.LineZone(
        start=sv.Point(x1, y1),
        end=sv.Point(x2, y2),
        triggering_anchors=ANCHORS[args.anchor],
        minimum_crossing_threshold=args.min_crossing_frames,
    )
    return Counter(
        line_zone=line_zone,
        tracker=ByteTrackTracker(frame_rate=fps or FALLBACK_FPS),
        fps=fps,
        wanted=wanted,
        confidence=args.confidence,
        model=model,
    )


def run_file(args: argparse.Namespace) -> Counter:
    if not os.path.isfile(args.source):
        die(f"source file not found: {args.source}")
    info = sv.VideoInfo.from_video_path(args.source)
    counter = build_counter(args, info.width, info.height, info.fps)
    started = time.time()

    if args.no_video:
        for index, frame in enumerate(sv.get_video_frames_generator(args.source)):
            if args.max_frames is not None and index >= args.max_frames:
                break
            counter.step(frame, index)
    else:

        def callback(frame: np.ndarray, index: int) -> np.ndarray:
            return counter.annotate(frame, counter.step(frame, index))

        sv.process_video(
            source_path=args.source,
            target_path=args.output,
            callback=callback,
            max_frames=args.max_frames,
            show_progress=True,
        )
        print(f"Wrote {args.output}")

    elapsed = time.time() - started
    frames = args.max_frames or info.total_frames
    if frames and elapsed:
        rate = frames / elapsed
        print(
            f"{frames} frames in {elapsed:.1f}s "
            f"({rate:.1f} fps processed, clip is {info.fps:.0f} fps)"
        )
    return counter


class LiveSession:
    """Feed an InferencePipeline's predictions through the same Counter, no video."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.counter: Counter | None = None
        self.frames = 0
        self.pipeline: InferencePipeline | None = None

    def on_prediction(self, predictions: dict | None, video_frame: VideoFrame | None) -> None:
        if predictions is None or video_frame is None:
            return
        if self.counter is None:
            height, width = video_frame.image.shape[:2]
            fps = float(getattr(video_frame, "fps", None) or FALLBACK_FPS)
            self.counter = build_counter(self.args, width, height, fps)
            print(f"stream open: {width}x{height}, counting {self.args.classes}")
        _, new = self.counter.process(sv.Detections.from_inference(predictions), self.frames)
        for e in new:
            zone = self.counter.line_zone
            print(
                f"[{e.time_s:8.1f}s] #{e.tracker_id} {e.class_name} {e.direction}"
                f"  (in {zone.in_count} / out {zone.out_count})"
            )
        self.frames += 1
        if self.args.max_frames is not None and self.frames >= self.args.max_frames:
            self.stop()

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.terminate()

    def run(self) -> Counter:
        source = self.args.source
        reference: str | int = int(source) if source.isdigit() else source
        self.pipeline = InferencePipeline.init(
            video_reference=reference,
            model_id=self.args.model,
            on_prediction=self.on_prediction,
            confidence=self.args.confidence,
        )
        self.pipeline.start()
        try:
            self.pipeline.join()
        except KeyboardInterrupt:
            self.stop()
            self.pipeline.join()
        if self.counter is None:
            die("no frames received from the stream")
        return self.counter


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    is_live = args.live or args.source.isdigit() or args.source.lower().startswith(LIVE_PREFIXES)
    counter = LiveSession(args).run() if is_live else run_file(args)
    counter.write_events(args.events)
    print(counter.report())
    print(f"{len(counter.events)} crossings logged to {args.events}")


if __name__ == "__main__":
    main()
