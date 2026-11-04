from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import parselmouth


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _pitch_range_semitones(values):
    values = _finite(values)
    values = values[values > 0]
    if len(values) < 2:
        return None
    low, high = np.percentile(values, [10, 90])
    return round(float(12 * np.log2(high / low)), 1) if low > 0 else None


def _median(values):
    values = _finite(values)
    return round(float(np.median(values)), 1) if len(values) else None


def analyze_audio(audio_path: Path, segments: list[dict]) -> dict:
    """Extract timestamp-aligned pitch and relative-intensity features.

    These measurements describe the recorded signal. They are not calibrated
    sound-pressure levels and do not determine whether delivery is effective.
    """
    sound = parselmouth.Sound(str(audio_path))
    pitch = sound.to_pitch(time_step=0.02, pitch_floor=75, pitch_ceiling=500)
    pitch_times = np.asarray(pitch.xs(), dtype=float)
    pitch_values = np.asarray(pitch.selected_array["frequency"], dtype=float)
    intensity_obj = sound.to_intensity(time_step=0.02)
    intensity_times = np.asarray(intensity_obj.xs(), dtype=float)
    intensity_values = np.asarray(intensity_obj.values[0], dtype=float)

    voiced = pitch_values[pitch_values > 0]
    all_intensity = _finite(intensity_values)
    segment_features = []

    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        if end <= start:
            continue
        pmask = (pitch_times >= start) & (pitch_times <= end) & (pitch_values > 0)
        imask = (intensity_times >= start) & (intensity_times <= end)
        p = pitch_values[pmask]
        intensity = _finite(intensity_values[imask])
        if not len(p) and not len(intensity):
            continue

        first_half = p[pitch_times[pmask] <= start + (end - start) / 2] if len(p) else np.array([])
        second_half = p[pitch_times[pmask] > start + (end - start) / 2] if len(p) else np.array([])
        pitch_change = None
        if len(first_half) and len(second_half):
            first_median, second_median = np.median(first_half), np.median(second_half)
            if first_median > 0 and second_median > 0:
                pitch_change = round(float(12 * np.log2(second_median / first_median)), 1)

        segment_features.append({
            "start_seconds": round(start, 2),
            "end_seconds": round(end, 2),
            "transcript_excerpt": str(seg.get("text", ""))[:140],
            "recognized_word_count": len(re.findall(r"\b[\w’'-]+\b", str(seg.get("text", "")))),
            "voiced_pitch_median_hz": _median(p),
            "pitch_10_to_90_range_semitones": _pitch_range_semitones(p),
            "pitch_median_change_first_to_second_half_semitones": pitch_change,
            "relative_intensity_median_db": _median(intensity),
            "relative_intensity_10_to_90_range_db": (
                round(float(np.percentile(intensity, 90) - np.percentile(intensity, 10)), 1)
                if len(intensity) else None
            ),
        })

    return {
        "pitch_voiced_frame_percent": round(100 * len(voiced) / max(1, len(pitch_values)), 1),
        "pitch_median_hz": _median(voiced),
        "pitch_10_to_90_range_semitones": _pitch_range_semitones(voiced),
        "relative_intensity_median_db": _median(all_intensity),
        "relative_intensity_standard_deviation_db": (
            round(float(np.std(all_intensity)), 1) if len(all_intensity) else None
        ),
        "relative_intensity_10_to_90_range_db": (
            round(float(np.percentile(all_intensity, 90) - np.percentile(all_intensity, 10)), 1)
            if len(all_intensity) else None
        ),
        "segment_features": segment_features[:120],
        "measurement_limits": [
            "Pitch variation is a measurable feature, not proof of effective or audience-appropriate intonation.",
            "Intensity is affected by microphone distance, gain, compression, and recording conditions; it is not calibrated room loudness.",
            "Segment boundaries come from automatic speech recognition and are approximate.",
            "These measurements cannot determine whether a speaker sounds engaging, defensive, mannered, or off-putting."
        ]
    }


def speech_statistics(segments: list[dict], duration: float) -> dict:
    words = re.findall(r"\b[\w’'-]+\b", " ".join(str(s.get("text", "")) for s in segments))
    ordered = sorted(segments, key=lambda x: float(x["start"]))
    speech_seconds = sum(max(0.0, float(s["end"]) - float(s["start"])) for s in ordered)
    gaps = []
    for previous, current in zip(ordered, ordered[1:]):
        gap = float(current["start"]) - float(previous["end"])
        if gap >= 0.35:
            gaps.append({
                "start_seconds": round(float(previous["end"]), 2),
                "end_seconds": round(float(current["start"]), 2),
                "duration_seconds": round(gap, 2)
            })

    utterance_rates = []
    for seg in ordered:
        seconds = max(0.0, float(seg["end"]) - float(seg["start"]))
        count = len(re.findall(r"\b[\w’'-]+\b", str(seg.get("text", ""))))
        if seconds >= 0.5 and count:
            utterance_rates.append({
                "start_seconds": round(float(seg["start"]), 2),
                "end_seconds": round(float(seg["end"]), 2),
                "approx_wpm": round(60 * count / seconds, 1),
                "recognized_words": count
            })

    return {
        "transcript_word_count": len(words),
        "video_duration_seconds": round(duration, 2),
        "recognized_speech_segment_seconds_sum": round(speech_seconds, 2),
        "approx_wpm_during_recognized_segments": round(60 * len(words) / speech_seconds, 1) if speech_seconds > 0 else None,
        "approx_wpm_over_entire_video": round(60 * len(words) / duration, 1) if duration > 0 else None,
        "recognized_segment_gaps_over_0_35_seconds": gaps[:80],
        "utterance_rate_samples": utterance_rates[:80],
        "limits": "Whisper timestamps and transcript words are approximate. A detected gap is not proof of an unplanned pause; WPM values are estimates, not rigid grading cutoffs."
    }


def _landmark_visible(landmark, threshold=0.55):
    return getattr(landmark, "visibility", 1.0) >= threshold


def _hand_centers(hand_result):
    centers = []
    if not hand_result.multi_hand_landmarks:
        return centers
    for landmarks in hand_result.multi_hand_landmarks:
        wrist = landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
        index = landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
        pinky = landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_MCP]
        centers.append((
            float((wrist.x + index.x + pinky.x) / 3),
            float((wrist.y + index.y + pinky.y) / 3)
        ))
    return sorted(centers, key=lambda point: point[0])


def _match_hand_displacements(previous, current):
    """Greedy nearest-neighbour matching of normalized hand centers."""
    if not previous or not current:
        return []
    candidates = []
    for pi, p in enumerate(previous):
        for ci, c in enumerate(current):
            distance = math.hypot(c[0] - p[0], c[1] - p[1])
            candidates.append((distance, pi, ci))
    used_previous, used_current, distances = set(), set(), []
    for distance, pi, ci in sorted(candidates):
        if pi in used_previous or ci in used_current:
            continue
        if distance > 0.30:
            continue
        used_previous.add(pi)
        used_current.add(ci)
        distances.append(distance)
    return distances


def track_body_and_hands(video_path: str, target_fps: float = 5.0) -> dict:
    """Track pose and hand landmarks through the full video at a regular sampling rate.

    Outputs observable landmark positions and movement proxies only. It does not
    infer gaze, eye contact, intent, gesture meaning, or audience engagement.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video for body/hand tracking.")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / source_fps if source_fps > 0 else 0.0
    interval = 1.0 / max(1.0, target_fps)
    next_sample = 0.0
    frame_index = 0
    observations = []
    previous_hands = []
    previous_shoulder_center = None
    movement_flags = []
    hand_visible_flags = []
    pose_visible_flags = []

    pose_api = mp.solutions.pose
    hands_api = mp.solutions.hands
    try:
        with pose_api.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose, hands_api.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands_processor:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
                if timestamp <= 0 and source_fps > 0:
                    timestamp = frame_index / source_fps
                if timestamp + 1e-6 < next_sample:
                    continue
                next_sample = timestamp + interval

                height, width = frame.shape[:2]
                scale = min(1.0, 640.0 / max(width, height))
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                pose_result = pose.process(rgb)
                hand_result = hands_processor.process(rgb)

                pose_data = {"visible": False}
                shoulder_center = None
                if pose_result.pose_landmarks:
                    lm = pose_result.pose_landmarks.landmark
                    ids = pose_api.PoseLandmark
                    left_shoulder, right_shoulder = lm[ids.LEFT_SHOULDER], lm[ids.RIGHT_SHOULDER]
                    left_hip, right_hip = lm[ids.LEFT_HIP], lm[ids.RIGHT_HIP]
                    nose = lm[ids.NOSE]
                    if all(_landmark_visible(x) for x in (left_shoulder, right_shoulder)):
                        shoulder_center = (
                            float((left_shoulder.x + right_shoulder.x) / 2),
                            float((left_shoulder.y + right_shoulder.y) / 2)
                        )
                        shoulder_width = math.hypot(
                            right_shoulder.x - left_shoulder.x,
                            right_shoulder.y - left_shoulder.y
                        )
                        shoulder_tilt = math.degrees(math.atan2(
                            right_shoulder.y - left_shoulder.y,
                            right_shoulder.x - left_shoulder.x
                        ))
                        head_offset = (
                            float((nose.x - shoulder_center[0]) / shoulder_width)
                            if _landmark_visible(nose) and shoulder_width > 0.02 else None
                        )
                    else:
                        shoulder_width, shoulder_tilt, head_offset = None, None, None

                    torso_lean = None
                    if all(_landmark_visible(x) for x in (left_hip, right_hip)) and shoulder_center:
                        hip_center_x = (left_hip.x + right_hip.x) / 2
                        torso_height = abs(shoulder_center[1] - (left_hip.y + right_hip.y) / 2)
                        if torso_height > 0.03:
                            torso_lean = float((shoulder_center[0] - hip_center_x) / torso_height)

                    pose_data = {
                        "visible": True,
                        "shoulder_line_tilt_degrees": round(shoulder_tilt, 1) if shoulder_tilt is not None else None,
                        "nose_offset_from_shoulder_center_in_shoulder_widths": round(head_offset, 2) if head_offset is not None else None,
                        "torso_lateral_offset_over_torso_height": round(torso_lean, 2) if torso_lean is not None else None
                    }

                hand_centers = _hand_centers(hand_result)
                displacements = _match_hand_displacements(previous_hands, hand_centers)
                max_displacement = max(displacements) if displacements else 0.0
                hand_motion = bool(displacements and max_displacement >= 0.018)
                body_shift = (
                    math.hypot(
                        shoulder_center[0] - previous_shoulder_center[0],
                        shoulder_center[1] - previous_shoulder_center[1]
                    ) if shoulder_center and previous_shoulder_center else None
                )
                movement_flags.append(hand_motion)
                hand_visible_flags.append(bool(hand_centers))
                pose_visible_flags.append(bool(pose_data["visible"]))
                observations.append({
                    "time_seconds": round(timestamp, 2),
                    "pose": pose_data,
                    "hand_count": len(hand_centers),
                    "hand_motion_detected": hand_motion,
                    "max_normalized_hand_displacement_since_previous_observation": round(max_displacement, 3),
                    "shoulder_center_shift_since_previous_observation": round(body_shift, 3) if body_shift is not None else None
                })
                previous_hands = hand_centers
                previous_shoulder_center = shoulder_center

    finally:
        cap.release()

    if not observations:
        return {
            "tracking_fps": target_fps,
            "duration_seconds": round(duration, 2),
            "observations": 0,
            "summary": "No video frames could be analyzed.",
            "windows": [],
            "movement_events": [],
            "limitations": ["No continuous landmark observations were available."]
        }

    window_size = 5.0
    windows = []
    end_time = max(duration, observations[-1]["time_seconds"] + interval)
    cursor = 0.0
    while cursor < end_time:
        subset = [o for o in observations if cursor <= o["time_seconds"] < cursor + window_size]
        if subset:
            pose_rate = sum(bool(o["pose"]["visible"]) for o in subset) / len(subset)
            hand_rate = sum(o["hand_count"] > 0 for o in subset) / len(subset)
            motion_rate = sum(o["hand_motion_detected"] for o in subset) / len(subset)
            tilt = [o["pose"]["shoulder_line_tilt_degrees"] for o in subset if o["pose"]["shoulder_line_tilt_degrees"] is not None]
            torso = [o["pose"]["torso_lateral_offset_over_torso_height"] for o in subset if o["pose"]["torso_lateral_offset_over_torso_height"] is not None]
            windows.append({
                "start_seconds": round(cursor, 1),
                "end_seconds": round(min(cursor + window_size, end_time), 1),
                "samples": len(subset),
                "pose_landmark_detection_percent": round(100 * pose_rate, 1),
                "hand_detection_percent": round(100 * hand_rate, 1),
                "hand_position_change_percent": round(100 * motion_rate, 1),
                "median_shoulder_line_tilt_degrees": _median(tilt),
                "median_torso_lateral_offset_ratio": _median(torso)
            })
        cursor += window_size

    # Cluster adjacent movement observations into timestamped intervals.
    active_times = [o["time_seconds"] for o in observations if o["hand_motion_detected"]]
    events = []
    if active_times:
        start = previous = active_times[0]
        for timestamp in active_times[1:]:
            if timestamp - previous > 1.0:
                events.append({"start_seconds": round(start, 2), "end_seconds": round(previous, 2)})
                start = timestamp
            previous = timestamp
        events.append({"start_seconds": round(start, 2), "end_seconds": round(previous, 2)})

    return {
        "tracking_fps": target_fps,
        "duration_seconds": round(duration, 2),
        "observations": len(observations),
        "pose_landmark_detection_percent": round(100 * sum(pose_visible_flags) / len(pose_visible_flags), 1),
        "hand_detection_percent": round(100 * sum(hand_visible_flags) / len(hand_visible_flags), 1),
        "hand_motion_observations": sum(movement_flags),
        "hand_motion_percent_of_all_observations": round(100 * sum(movement_flags) / len(movement_flags), 1),
        "movement_event_count": len(events),
        "movement_events": events[:60],
        "five_second_windows": windows,
        "limitations": [
            "Tracking samples the entire presentation at the configured frame rate; it is not analysis of every video frame.",
            "Hand-position changes are movement detections, not proof of intentional or meaningful gestures.",
            "Pose landmarks describe visible body geometry; they do not establish audience focus, eye contact, confidence, or engagement.",
            "Camera framing, occlusion, clothing, lighting, and camera angle can reduce tracking accuracy.",
            "The system does not infer the meaning or rhetorical appropriateness of a movement."
        ]
    }

def align_speech_and_motion(audio_metrics: dict, visual_tracking: dict) -> list[dict]:
    """Place hand-movement observations alongside timestamped acoustic speech features.

    Temporal overlap is descriptive only; it does not establish that a movement
    was a gesture or that it intentionally emphasized the spoken content.
    """
    movement_events = visual_tracking.get("movement_events", [])
    windows = visual_tracking.get("five_second_windows", [])
    aligned = []
    for segment in audio_metrics.get("segment_features", []):
        start = float(segment["start_seconds"])
        end = float(segment["end_seconds"])
        overlapping_events = [
            {"start_seconds": event["start_seconds"], "end_seconds": event["end_seconds"]}
            for event in movement_events
            if float(event["start_seconds"]) <= end and float(event["end_seconds"]) >= start
        ]
        overlapping_windows = [
            window for window in windows
            if float(window["start_seconds"]) <= end and float(window["end_seconds"]) >= start
        ]
        movement_rates = [
            float(window["hand_position_change_percent"])
            for window in overlapping_windows
            if window.get("hand_position_change_percent") is not None
        ]
        aligned.append({
            "start_seconds": start,
            "end_seconds": end,
            "transcript_excerpt": segment.get("transcript_excerpt", ""),
            "pitch_10_to_90_range_semitones": segment.get("pitch_10_to_90_range_semitones"),
            "pitch_median_change_first_to_second_half_semitones": segment.get("pitch_median_change_first_to_second_half_semitones"),
            "relative_intensity_10_to_90_range_db": segment.get("relative_intensity_10_to_90_range_db"),
            "overlapping_hand_movement_intervals": overlapping_events,
            "hand_position_change_percent_in_overlapping_windows": (
                round(float(np.mean(movement_rates)), 1) if movement_rates else None
            )
        })
    return aligned[:100]
