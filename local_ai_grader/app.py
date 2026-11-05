from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import requests
import streamlit as st
from faster_whisper import WhisperModel
from media_analysis import analyze_audio, align_speech_and_motion, speech_statistics, track_body_and_hands

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

def ollama_chat(model: str, messages: list[dict], timeout: int = 900, num_ctx: int | None = None) -> str:
    """Call Ollama and preserve its detailed error response for troubleshooting."""
    options = {"temperature": 0.1}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    try:
        response = requests.post(
            "http://127.0.0.1:11434/api/chat",
            json={"model": model, "messages": messages, "stream": False, "options": options},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not contact Ollama for model '{model}': {exc}") from exc

    if not response.ok:
        detail = (response.text or response.reason or "No error details returned by Ollama").strip()
        if len(detail) > 3000:
            detail = detail[:3000] + "…"
        raise RuntimeError(
            f"Ollama rejected the request for model '{model}' "
            f"(HTTP {response.status_code}): {detail}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON for model '{model}': {response.text[:1000]}") from exc

    if "message" not in payload or "content" not in payload["message"]:
        raise RuntimeError(f"Ollama response for model '{model}' did not contain message content: {str(payload)[:1000]}")
    return payload["message"]["content"]

def ensure_ollama_model(model: str) -> None:
    """Download a selected local model on demand, with visible pull progress."""
    tags = requests.get("http://127.0.0.1:11434/api/tags", timeout=15)
    tags.raise_for_status()
    installed = {item.get("name", "") for item in tags.json().get("models", [])}
    if model in installed:
        return

    st.info(f"Downloading {model} from Ollama. This is a one-time download for this computer.")
    progress = st.progress(0.0)
    detail = st.empty()
    response = requests.post(
        "http://127.0.0.1:11434/api/pull",
        json={"name": model, "stream": True},
        stream=True,
        timeout=(15, None),
    )
    response.raise_for_status()
    last_status = "Preparing download…"
    for line in response.iter_lines():
        if not line:
            continue
        item = json.loads(line)
        if item.get("error"):
            raise RuntimeError(f"Ollama could not download {model}: {item['error']}")
        last_status = item.get("status", last_status)
        total = item.get("total")
        completed = item.get("completed")
        if total:
            fraction = min(1.0, completed / total) if completed is not None else 0.0
            progress.progress(fraction)
            detail.write(f"{model}: {last_status} — {fraction:.0%}")
        else:
            detail.write(f"{model}: {last_status}")
    progress.progress(1.0)
    detail.success(f"{model} is ready.")


def ffmpeg_executable() -> str:
    installed = shutil.which("ffmpeg")
    if installed:
        return installed
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise FileNotFoundError("Bundled FFmpeg could not be located.") from exc


def extract_media(video_path: str, work: Path, sample_count: int = 24):
    audio_path = work / "audio.wav"
    subprocess.run(
        [ffmpeg_executable(), "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000", str(audio_path)],
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
                scale = min(1.0, 384.0 / max(frame.shape[:2]))
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
                if ok:
                    images.append({"time": round(sec, 1), "base64": base64.b64encode(encoded).decode("ascii")})
    cap.release()
    return audio_path, images, duration

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

def rubric_prompt(transcript: str, duration: float, speech_metrics: dict, audio_metrics: dict, visual_tracking: dict, speech_motion_alignment: list[dict], vision_notes: str):
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

CONTINUOUS BODY/HAND LANDMARK TRACKING ACROSS THE PRESENTATION:
{json.dumps(visual_tracking, indent=2)[:14000]}

TIME-ALIGNED SPEECH, ACOUSTIC, AND HAND-MOVEMENT OBSERVATIONS:
{json.dumps(speech_motion_alignment, indent=2)[:10000]}
Temporal overlap is descriptive only; do not infer that a movement was intentional or emphasized the words.

SUPPLEMENTAL VISION-MODEL NOTES FROM SPARSE IMAGE SAMPLES:
{vision_notes}

STRICT RULES:
- Evaluate only evidence explicitly present in the supplied data. Do not invent words, pauses, gestures, gaze, audience reaction, or speaker intent.
- Transcript-based claims must be described as transcript evidence; note likely recognition uncertainty.
- For Fluency, use both WPM measures and explain that segment-based WPM is approximate. Do not treat 150 WPM as a rigid cutoff. Mention filler words only when clearly present in the transcript; transcription can omit or hallucinate them. Segment gaps are not proof that a pause was unplanned. Never call a pause strategic unless rhetorical context clearly supports emphasis; otherwise describe it neutrally.
- For Word Choice, cite exact transcript phrases. Distinguish specialized vocabulary used precisely from unexplained jargon. Do not claim a technical term is incorrect unless the transcript and context support that conclusion. Flag uncertainty when topic context is missing.
- For Vocal Delivery, provide a BEST-FIT TENTATIVE score from 1–4 whenever the recording contains usable speech and acoustic evidence. Use pitch variation, relative-intensity variation, segment-level contours, and transcript context to make a cautious estimate of the observable delivery pattern. Explain exactly which evidence supports the estimate and which rubric elements remain uncertain. Acoustic measurements do not establish effective intonation, appropriate perceived volume, or whether the speaker sounds defensive/off-putting; never claim that they do. Do not infer emotion, confidence, or engagement. Use low confidence when the score relies mainly on acoustic proxies. Return null only if the audio is missing, unusable, or too incomplete to support even a cautious estimate.
- For Nonverbal Communication, provide a BEST-FIT TENTATIVE score from 1–4 whenever there is sufficient visible speaker evidence. Use full-video sampled landmark tracking, detection coverage, timestamped movement intervals, posture proxies, and timestamped frame observations. Distinguish visible orientation and movement from interpretations of intent. Movement detections are not proof of intentional gestures. The system cannot reliably establish eye contact, audience focus, gesture meaning, or strategic emphasis; explicitly identify these uncertainties and do not pretend they were observed. Use low confidence when camera framing, occlusion, sparse frame samples, or weak detection coverage limit the estimate. Return null only if the speaker is not sufficiently visible or the visual evidence is otherwise too incomplete/unreliable to support even a cautious estimate.
- IMPORTANT EVIDENCE CHECK: If the timestamped transcript below contains words, then usable speech evidence exists. You MUST NOT say that the transcript contains no usable speech evidence, that speech segments are missing, or that audio is unavailable. Use the actual transcript excerpts and supplied speech metrics. If a particular metric is unavailable, identify only that metric as unavailable.
- For Fluency, if transcript_word_count is greater than zero and recognized speech segments exist, assess the available transcript, approximate rates, and gaps. Do not substitute a generic no-evidence statement.
- For Word Choice, if transcript_word_count is greater than zero, quote at least two short exact phrases from the transcript when available and evaluate their clarity/relevance. If the transcript is too short to support a judgment, explain that specific limitation rather than claiming no transcript exists.
- For Vocal Delivery, if audio_metrics contains pitch or intensity measurements, discuss the actual supplied values. Do not treat missing pitch as missing speech when intensity or transcript evidence exists.
- For Nonverbal Communication, use the visual tracking and vision notes independently of transcript availability. If those sources contain observations, discuss them; do not use transcript limitations as the reason for a nonverbal score.
- For both categories, do not automatically abstain merely because some rubric dimensions cannot be measured. Select the closest-supported rubric level as a tentative recommendation, explain why it is the best fit, and explain why the nearest adjacent level is less supported. Clearly state which criteria require instructor verification from the original presentation.
- Tracking is sampled at a regular frame rate, not every frame. Do not claim continuous observation between tracking samples.
- Scores are tentative recommendations, not final grades. For Vocal Delivery and Nonverbal Communication, make a best-fit 1–4 estimate whenever usable evidence exists, even though some rubric dimensions cannot be verified automatically. Use null only when the relevant media evidence is genuinely insufficient or unusable. State uncertainty and identify the exact criteria needing instructor verification.
- Never default all categories to Level 1. Score each category independently from its own evidence; Level 1 is appropriate only when that category's evidence supports Level 1 descriptors. Do not use Level 1 as a placeholder for uncertainty or missing measurements.
- For each numeric score, justify why that level fits and why the nearest adjacent level is not better supported.
- Output ONLY valid JSON in this exact shape:
{{
 "categories": {{
  "Vocal Delivery": {{"score": null, "confidence": "low|medium|high", "rationale": "...", "evidence": ["..."], "adjacent_level_reasoning": "...", "limitations": "..."}},
  "Fluency": {{"score": null, "confidence": "low|medium|high", "rationale": "...", "evidence": ["..."], "adjacent_level_reasoning": "...", "limitations": "..."}},
  "Nonverbal Communication": {{"score": null, "confidence": "low|medium|high", "rationale": "...", "evidence": ["..."], "adjacent_level_reasoning": "...", "limitations": "..."}},
  "Word Choice": {{"score": null, "confidence": "low|medium|high", "rationale": "...", "evidence": ["..."], "adjacent_level_reasoning": "...", "limitations": "..."}}
 }},
 "strengths": ["..."],
 "improvements": ["..."],
 "overall_comments": "...",
 "transcript_language": "..."
}}
"""

def run_grading(text_model: str, vision_model: str, transcript: str, duration: float, segments: list[dict], images: list[dict], audio_metrics: dict, visual_tracking: dict):
    visual_notes = "No usable video frames were extracted."
    if images:
        observations = []
        for image in images:
            msg = {"role": "user", "content": f"At {image['time']:.1f}s, describe only visible speaker posture, body orientation and hand/arm position. Do not infer eye contact, intent, engagement, gesture meaning or behavior between frames. State if inconclusive.", "images": [image["base64"]]}
            observations.append(ollama_chat(vision_model, [msg], num_ctx=8192))
        visual_notes = "\n".join(observations)
    speech_metrics = speech_statistics(segments, duration)
    alignment = align_speech_and_motion(audio_metrics, visual_tracking)
    categories = {}
    for name, rubric in RUBRIC.items():
        if name in ("Fluency", "Word Choice"):
            evidence = {"transcript": transcript[:18000], "speech_metrics": speech_metrics}
            has_evidence = bool(transcript.strip()) and speech_metrics.get("transcript_word_count", 0) > 0
        elif name == "Vocal Delivery":
            evidence = {"acoustic_measurements": audio_metrics, "transcript_context": transcript[:5000]}
            has_evidence = bool(audio_metrics.get("segment_features")) or any(audio_metrics.get(k) is not None for k in ("pitch_median_hz", "pitch_10_to_90_range_semitones", "relative_intensity_median_db"))
        else:
            evidence = {"body_hand_tracking": visual_tracking, "sampled_frame_observations": visual_notes, "speech_motion_alignment": alignment}
            has_evidence = visual_tracking.get("observations", 0) > 0 or (bool(images) and bool(visual_notes.strip()))
        levels = "\n".join(f"Level {n}: {desc}" for n, desc in rubric["levels"].items())
        prompt = f"""Assess ONLY {name} for a university presentation.
Official rubric ({rubric['sub']}):
{levels}
Evidence:
{json.dumps(evidence, ensure_ascii=False)[:20000]}
Rules: use only evidence above; score this category independently. The transcript is {"NONEMPTY and contains usable spoken words" if transcript.strip() else "EMPTY"}. Never claim there is no transcript or no speech when it is nonempty. Fluency: use approximate rate/timing cautiously; do not invent stammering or fillers. Word Choice: quote exact transcript phrases. Vocal Delivery: use actual acoustic measurements, but they do not prove perceived loudness or effective intonation. Nonverbal: use only visible tracking/frame observations; transcript availability is irrelevant. Do not infer eye contact, intent, engagement or gesture meaning. Assign 1–4 only when supported by this category's evidence; use null only when genuinely insufficient. Never use 1 or null as a placeholder. Explain why the score fits and why the nearest adjacent level is less supported. Return ONLY JSON: {{"score": 1-4 or null, "confidence":"low|medium|high", "rationale":"...", "evidence":["specific evidence"], "adjacent_level_reasoning":"...", "limitations":"..."}}"""
        item, error = None, ""
        for attempt in range(2):
            retry = "" if attempt == 0 else f" Your previous answer failed validation: {error}. Correct it using the supplied evidence; do not invent a score."
            raw = ollama_chat(text_model, [{"role": "user", "content": prompt + retry}], num_ctx=8192)
            try:
                candidate = json_from_text(raw)
                score = candidate.get("score")
                bad_score = score is not None and (isinstance(score, bool) or not isinstance(score, int) or score not in (1, 2, 3, 4))
                fields = " ".join(str(candidate.get(k, "")) for k in ("rationale", "evidence", "adjacent_level_reasoning", "limitations")).lower()
                contradiction = name in ("Fluency", "Word Choice") and has_evidence and any(x in fields for x in ("no usable speech evidence", "no speech segments", "transcript does not contain", "transcript lacks"))
                bad_abstention = has_evidence and score is None
                unsupported_score = not has_evidence and score is not None
                missing = any(not candidate.get(k) for k in ("rationale", "evidence", "adjacent_level_reasoning", "limitations"))
                if bad_score or contradiction or bad_abstention or unsupported_score or missing:
                    error = "response contradicted available evidence, omitted required reasoning, or returned an unsupported score"
                    continue
                item = candidate
                break
            except (ValueError, json.JSONDecodeError) as exc:
                error = f"invalid JSON: {exc}"
        if item is None:
            item = {"score": None, "confidence": "low", "rationale": "No grade is recommended because the model failed evidence validation.", "evidence": [], "adjacent_level_reasoning": "A defensible adjacent-level comparison was not returned.", "limitations": error or "The model did not return a valid assessment."}
        categories[name] = item
    strengths = [f"{n}: {v['rationale']}" for n, v in categories.items() if v.get("score") is not None]
    improvements = [f"{n}: {v['limitations']}" for n, v in categories.items() if v.get("limitations")]
    return {"categories": categories, "strengths": strengths[:6], "improvements": improvements[:6], "overall_comments": "Each rubric category was assessed independently. Instructor review is required.", "transcript_language": "See detected language reported during transcription."}, visual_notes, alignment
st.title("Presentation Grader — Local AI")
st.caption("Local processing · student video and transcript stay on this computer · application code checks for verified updates at startup")
st.warning(
    "This is an instructor-assistance tool, not an autonomous or validated grading authority. "
    "It extracts timestamped acoustic features and tracks pose/hand landmarks throughout the video at a configurable sampling rate. "
    "These measurements do not establish effective vocal delivery, eye contact, or strategic gesture use."
)

with st.sidebar:
    st.header("Local model settings")
    st.caption("Compact models are selected by default to reduce first-run downloads. Advanced models can be downloaded automatically when selected.")
    text_choices = {
        "Recommended — Qwen 2.5 7B (about 4.7 GB)": "qwen2.5:7b",
        "Compact — Qwen 2.5 3B (about 1.9 GB)": "qwen2.5:3b",
        "Advanced — Qwen 2.5 7B (about 4.7 GB)": "qwen2.5:7b",
    }
    vision_choices = {
        "Compact — Qwen 2.5 VL 3B (about 3.2 GB)": "qwen2.5vl:3b",
        "Advanced — Qwen 2.5 VL 7B (about 6.0 GB)": "qwen2.5vl:7b",
    }
    text_label = st.selectbox("Rubric reasoning model", list(text_choices), index=0)
    vision_label = st.selectbox("Visual analysis model", list(vision_choices), index=0)
    text_model = text_choices[text_label]
    vision_model = vision_choices[vision_label]
    whisper_size = st.selectbox("Speech recognition model", ["tiny", "base", "small", "medium"], index=1)
    sample_count = st.slider("Vision-model image samples", 8, 36, 24)
    tracking_fps = st.slider("Continuous pose/hand tracking rate (frames per second)", 2, 10, 5)
    st.markdown("Landmark tracking samples the full video at the selected rate; it does not analyze every source frame.")

video = st.file_uploader("Choose a presentation video", type=["mp4", "mov", "mkv", "webm", "avi"])
if video:
    st.video(video.getvalue())
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
                status.update(label="Measuring pitch, intensity, and speech timing…")
                audio_metrics = analyze_audio(audio_path, segments)
                status.update(label=f"Tracking body and hand landmarks at {tracking_fps} fps…")
                visual_tracking = track_body_and_hands(str(video_path), target_fps=float(tracking_fps))
                st.write(f"Tracked {visual_tracking.get('observations', 0)} time points; pose landmarks detected in {visual_tracking.get('pose_landmark_detection_percent', 0)}% of samples; hands detected in {visual_tracking.get('hand_detection_percent', 0)}%.")
                status.update(label="Checking selected local AI models…")
                ensure_ollama_model(text_model)
                ensure_ollama_model(vision_model)
                status.update(label="Running local rubric analysis…")
                result, visual_notes, speech_motion_alignment = run_grading(text_model, vision_model, transcript, duration, segments, images, audio_metrics, visual_tracking)
                status.update(label="Analysis complete", state="complete")
            result["transcript"] = transcript
            result["visual_observations"] = visual_notes
            result["body_hand_tracking"] = visual_tracking
            result["speech_motion_alignment"] = speech_motion_alignment
            result["audio_measurements"] = audio_metrics
            result["speech_timing_metrics"] = speech_statistics(segments, duration)
            result["video_duration_seconds"] = duration
            result["analysis_method"] = "Whisper + Praat/Parselmouth timestamped acoustic features + MediaPipe pose/hand tracking + OpenCV frames + local Ollama text/vision models"
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
    st.subheader("Tentative rubric recommendations")
    categories = result.get("categories", {})
    for name in RUBRIC:
        item = categories.get(name, {})
        score = item.get("score")
        st.markdown(f"### {name} — {'Unable to estimate' if score is None else f'{score} / 4 · tentative'}")
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
        st.caption("Pitch and relative intensity are descriptive measures, not calibrated perceived volume or proof of effective intonation.")
        st.markdown("**Timestamped acoustic features by recognized speech segment**")
        st.json(result.get("audio_measurements", {}).get("segment_features", []))
        st.markdown("**Speech timing**")
        st.json(result.get("speech_timing_metrics", {}))
        st.text_area("Transcript", result.get("transcript", ""), height=260)
        st.markdown("**Speech and hand-movement timing alignment**")
        st.json(result.get("speech_motion_alignment", []))
        st.markdown("**Full-presentation pose and hand tracking**")
        st.json(result.get("body_hand_tracking", {}))
        st.markdown("**Supplemental sampled-frame observations**")
        st.write(result.get("visual_observations", ""))
