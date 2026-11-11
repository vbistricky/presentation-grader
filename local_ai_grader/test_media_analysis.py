import tempfile
import unittest
from pathlib import Path

import numpy as np
import parselmouth

from media_analysis import (
    _match_hand_displacements,
    _summarize_tracking_windows,
    align_speech_and_motion,
    analyze_audio,
    speech_statistics,
)


class MediaAnalysisTests(unittest.TestCase):
    def test_pitch_features_are_extracted_for_timestamped_speech(self):
        sample_rate = 16000
        times = np.arange(sample_rate * 3) / sample_rate
        samples = 0.25 * np.sin(2 * np.pi * 180 * times)
        sound = parselmouth.Sound(samples, sampling_frequency=sample_rate)

        with tempfile.TemporaryDirectory() as temp:
            audio_path = Path(temp) / "synthetic.wav"
            sound.save(str(audio_path), "WAV")
            result = analyze_audio(
                audio_path,
                [{"start": 0.2, "end": 2.8, "text": "A short synthetic speech segment"}],
            )

        self.assertAlmostEqual(result["pitch_median_hz"], 180, delta=4)
        self.assertEqual(len(result["segment_features"]), 1)
        self.assertIsNotNone(result["segment_features"][0]["pitch_10_to_90_range_semitones"])

    def test_speech_statistics_reports_rates_and_neutral_gaps(self):
        segments = [
            {"start": 0.0, "end": 1.5, "text": "Hello there"},
            {"start": 2.0, "end": 3.0, "text": "welcome everyone"},
        ]
        result = speech_statistics(segments, 4.0)

        self.assertEqual(result["transcript_word_count"], 4)
        self.assertEqual(len(result["recognized_segment_gaps_over_0_35_seconds"]), 1)
        self.assertAlmostEqual(result["approx_wpm_during_recognized_segments"], 96.0)
        self.assertIn("not proof", result["limits"])

    def test_tracking_windows_are_bucketed_at_five_second_boundaries(self):
        observations = [
            {"time_seconds": 0.0, "pose": {"visible": True, "shoulder_line_tilt_degrees": 2.0, "torso_lateral_offset_over_torso_height": None, "nose_offset_from_shoulder_center_in_shoulder_widths": None}, "hand_count": 1, "hand_motion_detected": True, "shoulder_center_shift_since_previous_observation": 0.1},
            {"time_seconds": 4.9, "pose": {"visible": False, "shoulder_line_tilt_degrees": None, "torso_lateral_offset_over_torso_height": None, "nose_offset_from_shoulder_center_in_shoulder_widths": None}, "hand_count": 0, "hand_motion_detected": False, "shoulder_center_shift_since_previous_observation": None},
            {"time_seconds": 5.0, "pose": {"visible": True, "shoulder_line_tilt_degrees": 4.0, "torso_lateral_offset_over_torso_height": 0.2, "nose_offset_from_shoulder_center_in_shoulder_widths": 0.1}, "hand_count": 2, "hand_motion_detected": False, "shoulder_center_shift_since_previous_observation": 0.2},
        ]

        windows = _summarize_tracking_windows(observations, 6.0, 1.0)

        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["samples"], 2)
        self.assertEqual(windows[0]["pose_landmark_detection_percent"], 50.0)
        self.assertEqual(windows[1]["start_seconds"], 5.0)
        self.assertEqual(windows[1]["samples"], 1)
        self.assertEqual(windows[1]["hand_detection_percent"], 100.0)

    def test_hand_positions_are_matched_between_observations(self):
        distances = _match_hand_displacements(
            [(0.20, 0.50), (0.80, 0.50)],
            [(0.25, 0.50), (0.82, 0.50)],
        )
        self.assertEqual(len(distances), 2)
        self.assertAlmostEqual(max(distances), 0.05, delta=0.001)

    def test_speech_and_hand_movement_are_aligned_by_time(self):
        audio = {
            "segment_features": [{
                "start_seconds": 1.0,
                "end_seconds": 3.0,
                "transcript_excerpt": "spoken phrase",
                "pitch_10_to_90_range_semitones": 4.2,
                "pitch_median_change_first_to_second_half_semitones": 1.1,
                "relative_intensity_10_to_90_range_db": 3.0,
            }]
        }
        tracking = {
            "movement_events": [{"start_seconds": 2.0, "end_seconds": 2.5}],
            "five_second_windows": [{
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "hand_position_change_percent": 20.0,
            }],
        }
        aligned = align_speech_and_motion(audio, tracking)

        self.assertEqual(len(aligned), 1)
        self.assertEqual(len(aligned[0]["overlapping_hand_movement_intervals"]), 1)
        self.assertEqual(aligned[0]["hand_position_change_percent_in_overlapping_windows"], 20.0)


if __name__ == "__main__":
    unittest.main()
