from __future__ import annotations

import base64
import json
import re
import subprocess
import tempfile
from pathlib import Path

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

def extract_media(video_path: str, work: Path, sample_count: int = 10):
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

def rubric_prompt(transcript: str, duration: float, segments: list[dict], vision_notes: str):
    levels = "\n".join(
        f"{name}:\n" + "\n".join(f"Level {n}: {desc}" for n, desc in item["levels"].items())
        for name, item in RUBRIC.items()
    )
    return f"""You are an evidence-led university presentation assessor. Apply this rubric exactly; do not change its meaning.
OFFICIAL RUBRIC:
{levels}

PRESENTATION DATA:
Duration: {duration:.1f} seconds.
Timestamped speech recognition transcript (may contain recognition errors):
{transcript[:24000]}

VISUAL OBSERVATIONS FROM A SPARSE SET OF VIDEO FRAMES:
{vision_notes}

STRICT RULES:
- Evaluate only evidence explicitly present in the supplied data. Do not invent words, pauses, gestures, gaze, audience reaction, or speaker intent.
- Transcript-based claims must be described as transcript evidence; note likely recognition uncertainty.
- For Fluency, calculate approximate words per minute from transcript word count and presentation duration. Mention filler words only if they appear clearly in the transcript. Segment gaps are not proof that a pause was unplanned. Never call a pause strategic unless the surrounding context clearly supports rhetorical emphasis; otherwise describe it neutrally.
- For Word Choice, cite actual words or phrases from the transcript and explain why they are clear, jargon-heavy, precise, or oversimplified.
- Sparse frames cannot establish sustained eye contact, frequency of gestures, or behavior between frames. Do not infer those. Nonverbal score may be null if evidence is insufficient.
- CRITICAL: You have not heard the audio. Do not score Vocal Delivery based on transcript, loudness assumptions, or visual impressions. Return score null and explain that an instructor must listen for actual volume and intonation.
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

def run_grading(text_model: str, vision_model: str, transcript: str, duration: float, images: list[dict]):
    visual_notes = "No usable video frames were extracted."
    if images:
        visual_messages = [{
            "role": "user",
            "content": (
                "Review these presentation video frames in chronological order. Describe only visible, concrete facts "
                "about speaker position, orientation, posture, and gestures that are clearly visible. Do not infer "
                "eye contact, intent, engagement, frequency, or what happened between frames. Include frame timestamps. "
                "If a person is too small/obscured or frames are inconclusive, say so."
            ),
            "images": [x["base64"] for x in images]
        }]
        visual_notes = ollama_chat(vision_model, visual_messages)
    segments = []
    # The transcript is already timestamped; retain segment times as the evidence source.
    for line in transcript.splitlines():
        segments.append(line)
    prompt = rubric_prompt(transcript, duration, segments, visual_notes)
    raw = ollama_chat(text_model, [{"role": "user", "content": prompt}])
    return json_from_text(raw), visual_notes

st.title("Presentation Grader — Local AI")
st.caption("Local processing prototype · student video and transcript stay on this computer")
st.warning(
    "This is an instructor-assistance tool, not an autonomous or validated grading authority. "
    "It deliberately leaves Vocal Delivery unscored because this version does not listen to vocal tone. "
    "Sparse video frames cannot establish sustained eye contact or gesture frequency."
)

with st.sidebar:
    st.header("Local model settings")
    text_model = st.text_input("Text model in Ollama", "qwen2.5:7b")
    vision_model = st.text_input("Vision model in Ollama", "qwen2.5vl:7b")
    whisper_size = st.selectbox("Speech recognition model", ["base", "small", "medium"], index=1)
    sample_count = st.slider("Video frames to inspect", 4, 20, 10)
    st.markdown("Models must already be downloaded in Ollama. First run can take several minutes.")

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
                status.update(label="Running local rubric analysis…")
                result, visual_notes = run_grading(text_model, vision_model, transcript, duration, images)
                status.update(label="Analysis complete", state="complete")
            result["transcript"] = transcript
            result["visual_observations"] = visual_notes
            result["video_duration_seconds"] = duration
            result["analysis_method"] = "Local Whisper transcription + local Ollama text/vision models"
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
    with st.expander("View transcript and visual observations"):
        st.text_area("Transcript", result.get("transcript", ""), height=260)
        st.write(result.get("visual_observations", ""))
