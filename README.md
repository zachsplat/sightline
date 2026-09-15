# Sightline

Hourly people and vehicle counts from the security cameras you already have.

I install and maintain camera-security systems. The question customers ask
most is one the NVR answers badly or behind a license tier: how many people or
vehicles came through this door or gate, and when? Sightline answers it from
any recorded clip or live RTSP stream with a virtual tripwire, using Roboflow's
open tooling: [Inference](https://inference.roboflow.com) for detection,
[trackers](https://github.com/roboflow/trackers) for ByteTrack, and
[supervision](https://supervision.roboflow.com) for the line counting and
annotation.

![Final frame of the sample clip: tracked people with IDs, motion traces, the tripwire, and the final in/out counts](docs/sample-output.png)

## What you get

- A **crossing log** (`events.csv`): one row per crossing with the frame,
  timestamp, track ID, class, and direction. This is the deliverable; the video
  is a debugging aid.
- **Per-class totals** (`person`, `car`, `truck`, ...) and a per-hour rollup
  (per-minute on short clips) when the crossings span more than one bucket.
- An optional **annotated video** with boxes, IDs, motion traces, and live counts.
- A **live mode** for `rtsp://` cameras and webcams, printing each crossing as
  it happens.

## Pipeline

```
frame -> Inference (YOLOv8n, COCO) -> trackers.ByteTrackTracker -> sv.LineZone -> log
```

`sv.Detections.from_inference` adapts the model output; the tracker gives each
person a stable ID; `LineZone` fires once per direction change per track. A
crossing is **in** when the object ends up on the left of the line as you look
from its start point to its end point. For the default left-to-right line across
the middle of the frame, bottom-to-top is in and top-to-bottom is out. Swap the
endpoints to flip it. `tests/test_sightline.py` pins this down.

## Run it

`inference` requires Python 3.10 to 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# a recorded clip
python sightline.py --source clip.mp4 --line 0,540,1920,540 --classes person,car,truck

# a live camera (no video is written; crossings print as they happen)
python sightline.py --source rtsp://user:pass@192.168.1.20/stream --classes person

# counts and log only, no annotated video (a little faster)
python sightline.py --source clip.mp4 --no-video
```

Options that matter:

- `--line x1,y1,x2,y2` places the tripwire in pixels (default: horizontal, mid-frame).
- `--anchor center|corners|bottom` picks which point of the box has to cross.
  See the accuracy section for why `center` is the default.
- `--min-crossing-frames N` requires a track to stay on the far side for N
  frames before it counts. Raise it to ignore people loitering on the line.
- `--classes all` counts every COCO class. Unknown class names are rejected up
  front against the model's class list.

The default `yolov8n-640` model downloads once and runs locally; no API key is
needed for it. Some Roboflow models need `ROBOFLOW_API_KEY` in the environment.

## Results on the sample clip

Roboflow's `people-walking.mp4` (1920x1080, 25 fps, 341 frames, an overhead
view of a concourse), tripwire across the middle, counting `person`:

```
Crossed IN:  7
Crossed OUT: 8
  person       in    7  out    8
15 crossings logged to events.csv
341 frames in 12.8s (26.6 fps processed, clip is 25 fps)
```

That is realtime on CPU only (no GPU); `--no-video` runs about 28 fps.
Versions: inference 1.6.0, supervision 0.29.1, trackers 2.4.0, Python 3.12.

### How I checked the number

A count nobody has verified is not worth much, so I checked it two ways.

1. `validate.py` takes the same arguments as `sightline.py`, re-runs the
   pipeline, independently derives crossings from each track's centroid path
   (any line angle), and prints the tracks where the two methods disagree:

   ```bash
   python validate.py --source clip.mp4 --line 0,540,1920,540 --classes person
   ```
2. I reviewed every logged crossing frame by frame (a contact sheet of the band
   around the line at each event), and the trajectories of every disputed track.

Every one of the 15 events is a real crossing, and I found no crossing that was
missed: **15 of 15 on this clip**. The independent evidence is the visual review;
the validator shares the centroid rule with the default anchor, so their
agreement alone would be circular.

The anchor choice is where the real accuracy story is. `LineZone` counts when
its triggering anchors move from one side of the line to the other:

| `--anchor` | anchors | in / out | notes |
|---|---|---|---|
| `corners` | all four box corners (supervision default) | 7 / 6 | missed 2 people who **started on the line** (box straddling it at their first frame, so there was no "from" side) |
| `center` | box center | **7 / 8** | catches both; matches the visual review |
| `bottom` | bottom-center, the feet | 9 / 8 | feet jitter: one track fired out, in, out within four frames; `--min-crossing-frames 3` suppresses it |

So `center` is the default. `corners` is the conservative choice if false
positives cost more than misses; `bottom` is the physically right one for a
floor-plane camera once you raise `--min-crossing-frames`.

## Upstream

The `tracker_id >= 0` filter in `Counter.track` works around supervision
treating every unconfirmed track (id -1) as one shared track, which
inflated the sample count to 97 / 95:
[roboflow/supervision#2578](https://github.com/roboflow/supervision/issues/2578).
The fix, a guard in `LineZone.trigger`, is in
[#2579](https://github.com/roboflow/supervision/pull/2579).

## Known limitations

- **One clip.** These numbers come from a single overhead concourse clip. A
  low-angle door camera or a night scene will behave differently; run
  `validate.py` and eyeball the events before trusting a new setup.
- **ID churn.** ByteTrack assigns a new ID when someone leaves the frame or is
  occluded and returns, so "distinct track IDs" is not "distinct people". Counts
  are still correct because each crossing is logged once per direction change
  per track.
- **Loiterers on the line** can register a crossing when their box jitters
  across it. `--min-crossing-frames` is the knob.
- **Fixed camera assumed.** The tripwire is in pixel space; a pan or zoom
  invalidates it.
- **Not for safety-critical occupancy compliance.** It is an operations tool.

## Privacy

Everything runs locally; no frames leave the machine. The design intent is
aggregate counts. The annotated video keeps faces and tracks, so treat it as a
debugging artifact and use `--no-video` in production.

## Tests

```bash
python -m unittest discover -s tests -v
```

Twenty tests, no model needed, milliseconds: argument validation; synthetic
tracks driven across a `LineZone` to pin down the direction semantics, that a
track which turns around counts once each way, and that a box hovering on the
line does not count; the class filter, the crossing log, per-class and
per-interval totals, and the CSV round trip; and the validator's geometry.

## Layout

```
sightline.py   the tool (file mode and live mode)
validate.py    centroid-vs-tripwire cross-check, same options as sightline.py
tests/         unit tests
docs/          sample output frame
```
