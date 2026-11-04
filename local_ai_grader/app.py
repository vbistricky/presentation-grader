from __future__ import annotations

import base64
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import parselmouth

import cv2
import requests
import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="Presentation Grader — Local AI", page_icon="🎓", layout="wide")

RUBRIC = {
    "Vocal Delivery": {
        "sub": "Volume and intonation for emphasis",
        "levels": {
            4: "Volume is appropriate and clearly heard. Intonation engages the audience, emphasizes key points, and conveys the logic of the message. It is neither flat nor mannered.",
            3: "Volume is appropriate and clearly heard. Intonation emphasizes key points in support of the message and is neither flat nor mannered.",
            2: "Volume is appropriate and clearly heard. Intonation may be flat or mannered, limiting emphasis and audience engagement.",
            1: "Volume may be too loud or too quiet, and/or intonation may be flat, mannered, defensive, off-putting, or unengaging."
        }
    },
    "Fluency": {
        "sub": "Clear flow, smooth pace and cadence, no stammering or filler words",
        "levels": {
            4: "Pauses strategically for emphasis and establishes a regular cadence; speaks without unplanned pauses, generally at about 150 words per minute.",
            3: "Has no unplanned pauses, no stammering, and no filler words; speaks fluently at about 150 words per minute.",
            2: "May have an unplanned pause or two, or occasional stammering or filler words; pacing does not adequately emphasize key points.",
            1: "May have no clear cadence or rhythm due to frequent unplanned pauses, stammering, or filler words; pace obscures key points of emphasis."
        }
    },
    "Nonverbal Communication": {
        "sub": "Body language, eye contact, and gestures",
        "levels": {
            4: "Focuses on the audience engagingly; uses gestures judiciously, unaffectedly, and strategically to emphasize key points; is neither stiff nor casual.",
            3: "Focuses on the audience; uses gestures strategically to emphasize key points; is neither stiff nor casual.",
            2: "Focuses on the audience; may stiffly use limited or no gestures, occasionally focus on text, or use occasional gestures/mannerisms unconnected with key points.",
            1: "May not clearly focus on the audience, may consistently focus on text, or may use consistently distracting gestures/mannerisms unconnected with the message."
        }
    },
    "Word Choice": {
        "sub": "Relevant vocabulary showing familiarity and confidence",
        "levels": {
            4: "Uses technical language or terms of art appropriately for precision, not as jargon; contextualizes terms for a broad audience without oversimplifying.",
            3: "Uses clear, plain language free of jargon and understandable to a broad audience without oversimplifying; may use limited correct technical vocabulary needing context.",
            2: "Uses technical language as jargon that obscures meaning, and/or oversimplifies key concepts.",
            1: "May use technical terms imprecisely or incorrectly, and/or oversimplify or distort concepts through erroneous word choice."
        }
    }
}

def ollama_chat(model: str, messages: list[dict], timeout: int = 900) -> str:
    response = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={"model": model, "messages": messages, "stream": False, "options": {"temperature": 0.1}},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]

def extract_media(video_path: str, work: Path, sample_count: int = 24):
    audio_path = work / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", str(audio_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps else 0
    images = []
    if duration:
        for sec in [duration * (i + 0.5) / sample_count for i in range(sample_count)]:
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ok, frame = cap.read()
            if ok:
                frame = cv2.resize(frame, (min(768, frame.shape[1]), int(frame.shape[0] * min(768, frame.shape[1]) / frame.shape[1])))
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
                if ok:
                    images.append({"time": round(sec, 1), "base64": base64.b64encode(encoded).decode("ascii")})
    cap.release()
    return audio_path, images, duration

def analyze_audio(audio_path: Path):
    """Descriptive acoustic measures, not calibrated loudness or a substitute for listening."""
    sound = parselmouth.Sound(str(audio_path))
    pitch = sound.to_pitch(time_step=0.02, pitch_floor=75, pitch_ceiling=500)
    frequencies = pitch.selected_array["frequency"]
    voiced = frequencies[frequencies > 0]
    intensity = sound.to_intensity(time_step=0.02).values[0]
    intensity = intensity[np.isfinite(intensity)]
    p10 = float(np.percentile(voiced, 10)) if len(voiced) else None
    p90 = float(np.percentile(voiced, 90)) if len(voiced) else None
    return {
        "pitch_voiced_frame_percent": round(100 * len(voiced) / max(1, len(frequencies)), 1),
        "pitch_median_hz": round(float(np.median(voiced)), 1) if len(voiced) else None,
        "pitch_10th_percentile_hz": round(p10, 1) if p10 else None,
        "pitch_90th_percentile_hz": round(p90, 1) if p90 else None,
        "pitch_10_to_90_range_semitones": round(12 * np.log2(p90 / p10), 1) if p10 and p90 else None,
        "intensity_median_db_relative": round(float(np.median(intensity)), 1) if len(intensity) else None,
        "intensity_standard_deviation_db": round(float(np.std(intensity)), 1) if len(intensity) else None,
        "intensity_10_to_90_range_db": round(float(np.percentile(intensity, 90) - np.percentile(intensity, 10)), 1) if len(intensity) else None,
        "measurement_note": "Pitch and intensity are signal-derived descriptive measures. They do not establish perceived engagement, appropriate room volume, or effective emphasis. Intensity is not calibrated to the room or listener."
    }

def speech_statistics(segments, duration: float):
    words = re.findall(r"\b[\w’'-]+\b", " ".join(s["text"] for s in segments))
    spoken_seconds = sum(max(0.0, s["end"] - s["start"]) for s in segments)
    gaps = []
    ordered = sorted(segments, key=lambda x: x["start"])
    for previous, current in zip(ordered, ordered[1:]):
        gap = current["start"] - previous["end"]
        if gap > 0:
            gaps.append(round(gap, 2))
    return {
        "transcript_word_count": len(words),
        "video_duration_seconds": round(duration, 2),
        "whisper_segment_duration_seconds_sum": round(spoken_seconds, 2),
        "approx_words_per_minute_during_recognized_segments": round(60 * len(words) / spoken_seconds, 1) if spoken_seconds > 0 else None,
        "approx_words_per_minute_over_entire_video": round(60 * len(words) / duration, 1) if duration > 0 else None,
        "gaps_between_recognized_segments_seconds": gaps,
        "note": "Whisper boundaries are approximate. Gaps may reflect pauses, silence, editing, or recognition behavior; they are not automatically unplanned."
    }

def transcribe(audio_path: Path, model_size: str):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(audio_path), beam_size=5, word_timestamps=False, vad_filter=True)
    result = []
    for seg in segments:
        result.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()})
    return result, info.language, info.language_probability

def make_transcript(segments):
    return "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}s] {s['text']}" for s in segments)

def json_from_text(text: str):
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("The local model did not return valid JSON. Try again or use a different model.")
    return json.loads(match.group(0))

def rubric_prompt(transcript: str, duration: float, speech_metrics: dict, audio_metrics: dict, vision_notes: str):
    levels = "\n".join(
        f"{name}:\n" + "\n".join(f"Level {n}: {desc}" for n, desc in item["levels"].items())
        for name, item in RUBRIC.items()
    )
    return f"""You are an evidence-led university presentation assessor. Apply this rubric exactly; do not change its meaning.
OFFICIAL RUBRIC:
{levels}

PRESENTATION DATA:
Duration: {duration:.1f} seconds.
Speech timing metrics:
{json.dumps(speech_metrics, indent=2)}
Acoustic measurements (not calibrated to room volume):
{json.dumps(audio_metrics, indent=2)}
Timestamped speech recognition transcript (may contain recognition errors):
{transcript[:24000]}

VISUAL OBSERVATIONS FROM A SPARSE SET OF VIDEO FRAMES:
{vision_notes}

STRICT RULES:
- Evaluate only evidence explicitly present in the supplied data. Do not invent words, pauses, gestures, gaze, audience reaction, or speaker intent.
- Transcript-based claims must be described as transcript evidence; note likely recognition uncertainty.
- For Fluency, use both WPM measures and explain that segment-based WPM is approximate. Do not treat 150 WPM as a rigid cutoff. Mention filler words only when clearly present in the transcript; transcription can omit or hallucinate them. Segment gaps are not proof that a pause was unplanned. Never call a pause strategic unless rhetorical context clearly supports emphasis; otherwise describe it neutrally.
- For Word Choice, cite exact transcript phrases. Distinguish specialized vocabulary used precisely from unexplained jargon. Do not claim a technical term is incorrect unless the transcript and context support that conclusion. Flag uncertainty when topic context is missing.
- For Vocal Delivery, use pitch and intensity measurements only as descriptive evidence. Pitch range may suggest variation but does not prove effective intonation; intensity does not establish whether volume is appropriate to the room. Do not infer emotion, confidence, engagement, or perceived loudness from these metrics. Return score null and provide an audio evidence summary for instructor review.
- For Nonverbal Communication, sampled frames support only frame-specific observations. Do not claim sustained eye contact, gesture frequency, strategic gesture use, or behavior between sampled frames. Return score null unless supplied observations directly establish a rubric behavior; qualify the limited sample.
- Do not treat sampled frames as continuous observation.
- Do not force a score. Use null when evidence is insufficient. Scores are provisional recommendations, not final grades.
- For each numeric score, justify why that level fits and why the nearest adjacent level is not better supported.
- Output ONLY valid JSON in this exact shape:
{{
 "categories": {{
  "Vocal Delivery": {{"score": null, "confidence": "low|medium|high", "rationale": "...", "evidence": ["..."], "adjacent_level_reasoning": "...", "limitations": "..."}},
  "Fluency": {{"score": 1, "confidence": "low|medium|high", "rationale": "...", "evidence": ["..."], "adjacent_level_reasoning": "...", "limitations": "..."}},
  "Nonverbal Communication": {{"score": null, "confidence": "low|medium|high", "rationale": "...", "evidence": ["..."], "adjacent_level_reasoning": "...", "limitations": "..."}},
  "Word Choice": {{"score": 1, "confidence": "low|medium|high", "rationale": "...", "evidence": ["..."], "adjacent_level_reasoning": "...", "limitations": "..."}}
 }},
 "strengths": ["..."],
 "improvements": ["..."],
 "overall_comments": "...",
 "transcript_language": "..."
}}
"""

def run_grading(text_model: str, vision_model: str, transcript: str, duration: float, segments: list[dict], images: list[dict], audio_metrics: dict):
    visual_notes = "No usable video frames were extracted."
    if images:
        visual_messages = [{
            "role": "user",
            "content": (
                "Review these presentation frames in chronological order. Image timestamps, in image order: "
                + ", ".join(f"{x['time']:.1f}s" for x in images)
                + ". For each frame describe only clearly visible position, body orientation, posture, and hand/arm position. "
                "Do not infer eye contact, intent, engagement, confidence, strategic emphasis, gesture frequency, or behavior between frames. "
                "Do not interpret a still pose as a gesture. State when the speaker is too small, obscured, off-camera, or inconclusive. "
                "Separate observations by timestamp."
            ),
            "images": [x["base64"] for x in images]
        }]
        visual_notes = ollama_chat(vision_model, visual_messages)
    speech_metrics = speech_statistics(segments, duration)
    prompt = rubric_prompt(transcript, duration, speech_metrics, audio_metrics, visual_notes)
    raw = ollama_chat(text_model, [{"role": "user", "content": prompt}])
    return json_from_text(raw), visual_notes

st.title("Presentation Grader — Local AI")
st.caption("Local processing prototype · student video and transcript stay on this computer")
st.warning(
    "This is an instructor-assistance tool, not an autonomous or validated grading authority. "
    "It reports acoustic measurements but does not infer perceived loudness or effective intonation from them. "
    "Sampled frames cannot establish sustained eye contact or gesture frequency."
)

with st.sidebar:
    st.header("Local model settings")
    text_model = st.text_input("Text model in Ollama", "qwen2.5:7b")
    vision_model = st.text_input("Vision model in Ollama", "qwen2.5vl:7b")
    whisper_size = st.selectbox("Speech recognition model", ["base", "small", "medium"], index=1)
    sample_count = st.slider("Video frames to inspect", 8, 36, 24)
    st.markdown("More frames improve temporal coverage but still do not provide continuous observation.")

video = st.file_uploader("Choose a presentation video", type=["mp4", "mov", "mkv", "webm", "avi"])
if video and st.button("Analyze presentation", type="primary"):
    try:
        with tempfile.TemporaryDirectory(prefix="presentation_grader_") as tmp:
            work = Path(tmp)
            video_path = work / video.name
            video_path.write_bytes(video.getbuffer())
            with st.status("Extracting audio and video evidence…", expanded=True) as status:
                audio_path, images, duration = extract_media(str(video_path), work, sample_count)
                st.write(f"Video duration: {duration:.1f} seconds · frames sampled: {len(images)}")
                status.update(label="Transcribing speech locally…")
                segments, language, language_probability = transcribe(audio_path, whisper_size)
                transcript = make_transcript(segments)
                if not transcript.strip():
                    raise RuntimeError("No speech transcript was detected. Check the audio track or try a different Whisper model.")
                st.write(f"Detected language: {language} (confidence {language_probability:.2f})")
                status.update(label="Measuring audio features…")
                audio_metrics = analyze_audio(audio_path)
                status.update(label="Running local rubric analysis…")
                result, visual_notes = run_grading(text_model, vision_model, transcript, duration, segments, images, audio_metrics)
                status.update(label="Analysis complete", state="complete")
            result["transcript"] = transcript
            result["visual_observations"] = visual_notes
            result["audio_measurements"] = audio_metrics
            result["speech_timing_metrics"] = speech_statistics(segments, duration)
            result["video_duration_seconds"] = duration
            result["analysis_method"] = "Whisper + Parselmouth acoustic measurements + sampled OpenCV frames + local Ollama text/vision models"
            result["human_review_required"] = True
            st.session_state["result"] = result
    except FileNotFoundError:
        st.error("FFmpeg was not found. Install FFmpeg and make sure it is available on your PATH.")
    except requests.ConnectionError:
        st.error("Ollama is not running. Start Ollama, download the selected models, and try again.")
    except requests.HTTPError as exc:
        st.error(f"Ollama returned an error: {exc}. Check that both selected models are installed and support vision where required.")
    except Exception as exc:
        st.exception(exc)

if "result" in st.session_state:
    result = st.session_state["result"]
    st.subheader("Provisional rubric recommendations")
    categories = result.get("categories", {})
    for name in RUBRIC:
        item = categories.get(name, {})
        score = item.get("score")
        st.markdown(f"### {name} — {'Not scored' if score is None else f'{score} / 4'}")
        st.caption(f"Confidence: {item.get('confidence', 'not stated')}")
        st.write(item.get("rationale", "No rationale returned."))
        if item.get("evidence"):
            st.markdown("**Evidence**")
            for ev in item["evidence"]:
                st.markdown(f"- {ev}")
        if item.get("adjacent_level_reasoning"):
            st.markdown(f"**Adjacent-level reasoning:** {item['adjacent_level_reasoning']}")
        if item.get("limitations"):
            st.markdown(f"**Limitations:** {item['limitations']}")
    st.markdown("### Strengths")
    for x in result.get("strengths", []):
        st.markdown(f"- {x}")
    st.markdown("### Areas for improvement")
    for x in result.get("improvements", []):
        st.markdown(f"- {x}")
    st.markdown("### Overall comments")
    st.write(result.get("overall_comments", ""))
    st.download_button(
        "Download AI evaluation (.json)",
        data=json.dumps(result, indent=2, ensure_ascii=False),
        file_name="presentation-ai-evaluation.json",
        mime="application/json",
        type="primary"
    )
    with st.expander("Review audio measurements, transcript, and visual evidence"):
        st.markdown("**Audio measurements**")
        st.json(result.get("audio_measurements", {}))
        st.caption("Pitch and intensity are descriptive measures, not calibrated perceived volume or proof of effective intonation.")
        st.markdown("**Speech timing**")
        st.json(result.get("speech_timing_metrics", {}))
        st.text_area("Transcript", result.get("transcript", ""), height=260)
        st.markdown("**Sampled visual observations**")
        st.write(result.get("visual_observations", ""))
