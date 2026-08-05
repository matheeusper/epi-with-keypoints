import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import supervision as sv

from infer_ppe_keypoints import (
    HelmetTemporalFilter,
    PersonTracker,
    annotate_frame,
    parse_args,
)
from model_hub import resolve_checkpoint


def make_detections(boxes, scores, source_indices=None):
    boxes_array = np.asarray(boxes, dtype=float).reshape((-1, 4))
    if source_indices is None:
        source_indices = np.arange(len(boxes_array))
    return sv.Detections(
        xyxy=boxes_array,
        confidence=np.asarray(scores, dtype=float),
        class_id=np.zeros(len(boxes_array), dtype=int),
        data={"source_index": np.asarray(source_indices, dtype=int)},
    )


def make_person(track_id, status):
    return {
        "id": track_id,
        "ppe": {"capacete": {"status": status, "detections": []}},
        "conformes": int(status == "validado"),
    }


class PersonTrackerTests(unittest.TestCase):
    def test_ids_survive_detection_reordering(self):
        tracker = PersonTracker(fps=10.0, buffer_seconds=1.0, activation_threshold=0.5)
        first = tracker.update(
            make_detections([[0, 0, 10, 20], [30, 0, 40, 20]], [0.9, 0.9], [0, 1])
        )
        second = tracker.update(
            make_detections([[30, 0, 40, 20], [0, 0, 10, 20]], [0.9, 0.9], [1, 0])
        )

        first_ids = dict(zip(first.data["source_index"], first.tracker_id))
        second_ids = dict(zip(second.data["source_index"], second.tracker_id))
        self.assertEqual(first_ids, second_ids)
        self.assertTrue(all(track_id >= 0 for track_id in first.tracker_id))

    def test_short_occlusion_keeps_id_and_long_occlusion_creates_new_id(self):
        tracker = PersonTracker(fps=10.0, buffer_seconds=1.0, activation_threshold=0.5)
        detection = make_detections([[0, 0, 10, 20]], [0.9])
        original_id = int(tracker.update(detection).tracker_id[0])
        for _ in range(5):
            tracker.update(make_detections([], []))
        self.assertEqual(int(tracker.update(detection).tracker_id[0]), original_id)

        for _ in range(10):
            tracker.update(make_detections([], []))
        replacement_id = int(tracker.update(detection).tracker_id[0])
        self.assertNotEqual(replacement_id, original_id)


class HelmetTemporalFilterTests(unittest.TestCase):
    def test_isolated_absences_do_not_alert_but_majority_does(self):
        temporal = HelmetTemporalFilter(window_frames=5, buffer_frames=5)
        protected = make_person(1, "validado")
        temporal.apply([protected], 0)
        self.assertEqual(protected["ppe"]["capacete"]["status"], "validado")

        for frame_index in (1, 2):
            missed = make_person(1, "ausente")
            temporal.apply([missed], frame_index)
            helmet = missed["ppe"]["capacete"]
            self.assertEqual(helmet["status"], "validado")
            self.assertTrue(helmet["temporalmente_retido"])

        alert = make_person(1, "ausente")
        temporal.apply([alert], 3)
        self.assertEqual(alert["ppe"]["capacete"]["status"], "ausente")

    def test_unknown_and_occluded_frames_do_not_vote_for_absence(self):
        temporal = HelmetTemporalFilter(window_frames=3, buffer_frames=4)
        protected = make_person(7, "validado")
        temporal.apply([protected], 0)
        temporal.apply([], 1)
        unknown = make_person(7, "nao_verificavel")
        temporal.apply([unknown], 2)

        helmet = unknown["ppe"]["capacete"]
        self.assertEqual(helmet["status"], "validado")
        self.assertTrue(helmet["temporalmente_retido"])

    def test_valid_detection_recovers_immediately(self):
        temporal = HelmetTemporalFilter(window_frames=3, buffer_frames=3)
        for frame_index in range(2):
            absent = make_person(2, "ausente")
            temporal.apply([absent], frame_index)
        self.assertEqual(absent["ppe"]["capacete"]["status"], "ausente")

        protected = make_person(2, "validado")
        temporal.apply([protected], 2)
        self.assertEqual(protected["ppe"]["capacete"]["status"], "validado")
        self.assertFalse(protected["ppe"]["capacete"]["temporalmente_retido"])

    def test_state_expires_after_tracking_buffer(self):
        temporal = HelmetTemporalFilter(window_frames=3, buffer_frames=2)
        protected = make_person(3, "validado")
        temporal.apply([protected], 0)
        temporal.apply([], 1)
        temporal.apply([], 2)

        reused_id = make_person(3, "ausente")
        temporal.apply([reused_id], 3)
        helmet = reused_id["ppe"]["capacete"]
        self.assertEqual(helmet["status"], "nao_verificavel")
        self.assertTrue(helmet["temporalmente_retido"])


class FakePPEModel:
    def predict(self, image, threshold):
        return argparse.Namespace(
            xyxy=np.empty((0, 4)),
            confidence=np.array([]),
            class_id=np.array([], dtype=int),
            data={},
        )


class FakeKeypointModel:
    def predict(self, image, threshold):
        points = np.zeros((1, 17, 2), dtype=float)
        return argparse.Namespace(
            data={"xyxy": np.array([[1, 1, 9, 9]], dtype=float)},
            xy=points,
            keypoint_confidence=np.ones((1, 17), dtype=float),
            visible=np.ones((1, 17), dtype=bool),
            detection_confidence=np.array([0.9]),
        )


class CompatibilityTests(unittest.TestCase):
    def test_image_inference_remains_stateless_and_reports_new_audit_fields(self):
        args = argparse.Namespace(
            ppe_threshold=0.35,
            keypoint_threshold=0.55,
            person_nms_iou=0.75,
            keypoint_confidence=0.30,
            hide_person_boxes=False,
            draw_keypoints=False,
        )
        _, _, people_count, report = annotate_frame(
            np.zeros((10, 10, 3), dtype=np.uint8), FakePPEModel(), FakeKeypointModel(), args
        )

        self.assertEqual(people_count, 1)
        person = report["pessoas"][0]
        self.assertEqual(person["id"], 1)
        helmet = person["epis"]["capacete"]
        self.assertEqual(helmet["status"], "ausente")
        self.assertEqual(helmet["status_instantaneo"], "ausente")
        self.assertFalse(helmet["temporalmente_retido"])

    def test_temporal_cli_values_must_be_positive(self):
        for option in ("--track-buffer-seconds", "--helmet-window-seconds"):
            with self.subTest(option=option), patch(
                "sys.argv", ["infer_ppe_keypoints.py", "--image", "x.jpg", option, "0"]
            ), self.assertRaises(SystemExit):
                parse_args()


class ModelHubTests(unittest.TestCase):
    def test_existing_checkpoint_never_downloads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "model.pth"
            checkpoint.touch()
            with patch("model_hub.hf_hub_download") as download:
                self.assertEqual(resolve_checkpoint(checkpoint), checkpoint)
            download.assert_not_called()

    def test_default_checkpoint_is_downloaded_when_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "helmet_only_576_best.pth"
            with patch(
                "model_hub.hf_hub_download", return_value=str(checkpoint)
            ) as download:
                resolved = resolve_checkpoint(
                    checkpoint, repo_id="owner/model", default_path=checkpoint
                )
            self.assertEqual(resolved, checkpoint)
            download.assert_called_once_with(
                repo_id="owner/model",
                filename="helmet_only_576_best.pth",
                local_dir=checkpoint.parent,
            )

    def test_missing_custom_checkpoint_fails_without_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            custom = Path(temp_dir) / "custom.pth"
            default = Path(temp_dir) / "default.pth"
            with patch("model_hub.hf_hub_download") as download:
                with self.assertRaises(FileNotFoundError):
                    resolve_checkpoint(custom, default_path=default)
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
