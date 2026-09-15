"""Tests that need no model and run in milliseconds.

python -m unittest discover -s tests -v
"""

import csv
import os
import tempfile
import unittest
import warnings

import numpy as np
import supervision as sv

from sightline import Counter, parse_classes, parse_line
from validate import centroid_crossings, side_of_line

# supervision's LineZone uses np.cross on 2-D vectors, which NumPy 2 deprecates;
# not ours to fix, and it would otherwise print in every test run
warnings.filterwarnings("ignore", category=DeprecationWarning)


def box_detections(tracker_id: int, x: int, y: int, size: int = 40, name: str = "person"):
    return sv.Detections(
        xyxy=np.array([[x, y, x + size, y + size]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        tracker_id=np.array([tracker_id]),
        data={"class_name": np.array([name])},
    )


class PassThroughTracker:
    """Stands in for ByteTrack: returns detections with their ids untouched."""

    def update(self, detections: sv.Detections) -> sv.Detections:
        return detections


def make_counter(wanted=None, fps=25.0) -> Counter:
    return Counter(
        line_zone=sv.LineZone(start=sv.Point(0, 100), end=sv.Point(200, 100)),
        tracker=PassThroughTracker(),
        fps=fps,
        wanted=wanted,
        confidence=0.3,
        model=None,
    )


class ParseLineTests(unittest.TestCase):
    def test_default_is_horizontal_midline(self):
        self.assertEqual(parse_line(None, 1920, 1080), (0, 540, 1920, 540))

    def test_parses_four_ints(self):
        self.assertEqual(parse_line("10,20,30,40", 100, 100), (10, 20, 30, 40))

    def test_rejects_wrong_count(self):
        with self.assertRaises(SystemExit):
            parse_line("0,540,1920", 1920, 1080)

    def test_rejects_non_integers(self):
        with self.assertRaises(SystemExit):
            parse_line("0,540,abc,540", 1920, 1080)

    def test_rejects_out_of_frame(self):
        with self.assertRaises(SystemExit):
            parse_line("0,540,5000,540", 1920, 1080)

    def test_rejects_degenerate_line(self):
        with self.assertRaises(SystemExit):
            parse_line("5,5,5,5", 100, 100)


class ParseClassesTests(unittest.TestCase):
    def test_all_means_no_filter(self):
        self.assertIsNone(parse_classes("all"))

    def test_lowercases_and_strips(self):
        self.assertEqual(parse_classes(" Person, CAR "), {"person", "car"})

    def test_rejects_empty(self):
        with self.assertRaises(SystemExit):
            parse_classes(" , ")


class LineZoneSemanticsTests(unittest.TestCase):
    """Pin down what 'in' and 'out' mean so the README stays honest."""

    def setUp(self):
        # left-to-right horizontal tripwire at y=100 in a 200x200 frame
        self.zone = sv.LineZone(start=sv.Point(0, 100), end=sv.Point(200, 100))

    def walk(self, ys):
        for y in ys:
            self.zone.trigger(box_detections(tracker_id=1, x=80, y=y))

    def test_bottom_to_top_counts_as_in(self):
        # "in" ends up on the line's left looking from start to end, which for a
        # left-to-right line in image coordinates is upward.
        self.walk([160, 140, 60, 40, 20])
        self.assertEqual((self.zone.in_count, self.zone.out_count), (1, 0))

    def test_top_to_bottom_counts_as_out(self):
        self.walk([20, 40, 60, 140, 160])
        self.assertEqual((self.zone.in_count, self.zone.out_count), (0, 1))

    def test_turning_around_counts_once_each_way(self):
        self.walk([160, 140, 60, 20, 60, 140, 160])
        self.assertEqual((self.zone.in_count, self.zone.out_count), (1, 1))

    def test_hovering_on_the_line_does_not_count(self):
        self.walk([85, 90, 85, 90])
        self.assertEqual((self.zone.in_count, self.zone.out_count), (0, 0))


class CounterTests(unittest.TestCase):
    def test_filter_classes_keeps_only_wanted(self):
        counter = make_counter(wanted={"car"})
        dets = sv.Detections.merge(
            [box_detections(1, 0, 0, name="person"), box_detections(2, 50, 50, name="car")]
        )
        kept = counter.filter_classes(dets)
        self.assertEqual(list(kept.data["class_name"]), ["car"])

    def test_filter_classes_none_means_all(self):
        counter = make_counter(wanted=None)
        dets = box_detections(1, 0, 0, name="person")
        self.assertEqual(len(counter.filter_classes(dets)), 1)

    def test_record_logs_crossing_with_time_and_class(self):
        counter = make_counter(fps=25.0)
        for frame, y in enumerate([160, 140, 60, 20]):
            counter.process(box_detections(tracker_id=7, x=80, y=y, name="car"), frame)
        self.assertEqual(len(counter.events), 1)
        e = counter.events[0]
        self.assertEqual((e.tracker_id, e.class_name, e.direction), (7, "car", "in"))
        # at frame 2 the box's bottom edge sits exactly on the line, so the
        # crossing lands on frame 3 when it is fully across
        self.assertEqual(e.frame, 3)
        self.assertAlmostEqual(e.time_s, 3 / 25.0)

    def test_per_class_and_interval_totals(self):
        counter = make_counter(fps=1.0)
        # car goes in at frame 2, person goes out at frame 70 (second minute)
        for frame, y in enumerate([160, 140, 60, 20]):
            counter.process(box_detections(1, 80, y, name="car"), frame)
        for offset, y in enumerate([20, 60, 140, 160]):
            counter.process(box_detections(2, 80, y, name="person"), 68 + offset)
        self.assertEqual(counter.per_class(), {"car": (1, 0), "person": (0, 1)})
        self.assertEqual(counter.by_interval(60), {0: (1, 0), 60: (0, 1)})

    def test_write_events_round_trips(self):
        counter = make_counter(fps=25.0)
        for frame, y in enumerate([160, 140, 60, 20]):
            counter.process(box_detections(3, 80, y), frame)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.csv")
            counter.write_events(path)
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tracker_id"], "3")
        self.assertEqual(rows[0]["direction"], "in")
        self.assertEqual(rows[0]["class"], "person")


class ValidateTests(unittest.TestCase):
    def test_side_matches_linezone_convention(self):
        start, end = (0.0, 100.0), (200.0, 100.0)
        self.assertEqual(side_of_line(start, end, (50.0, 20.0)), "in")  # above
        self.assertEqual(side_of_line(start, end, (50.0, 180.0)), "out")  # below
        self.assertIsNone(side_of_line(start, end, (50.0, 100.0)))  # on the line

    def test_centroid_crossings_ignore_frames_on_the_line(self):
        start, end = (0.0, 100.0), (200.0, 100.0)
        path = [(0, (50.0, 180.0)), (1, (50.0, 100.0)), (2, (50.0, 20.0)), (3, (50.0, 10.0))]
        self.assertEqual(centroid_crossings(path, start, end), [(2, "in")])


if __name__ == "__main__":
    unittest.main()
