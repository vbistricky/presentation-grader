from __future__ import annotations

import base64
import json
import os
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

GUIDE_ROOT = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local" / "share")))
GUIDE_PATH = GUIDE_ROOT / "PresentationGrader" / "grading_guide.json"
GUIDE_BASELINE = {
    "schema_version": 1,
    "guide_title": "Presentation Grader — Instructor Interpretation Guide",
    "approved_guidelines": [
        {"id": "evidence-first", "status": "approved", "text": "Every score and factual statement must be supported by observable evidence in this presentation. Do not infer filler words, pauses, eye contact, intent, confidence, audience engagement, or strategic emphasis when the supplied evidence cannot verify them."},
        {"id": "adjacent-levels", "status": "approved", "text": "For every score, explain why the selected rubric level fits and why the nearest adjacent level is less supported. Explicitly explain borderline decisions rather than treating level descriptions as automatic thresholds."},
        {"id": "independent-categories", "status": "approved", "text": "Assess each rubric category independently from its own evidence. Missing or weak evidence in one category must not determine another category's score. Do not use Level 1 as a placeholder for uncertainty."},
        {"id": "acoustic-proxies", "status": "approved", "text": "Pitch and relative-intensity measurements are descriptive proxies. They do not establish perceived room volume, effective intonation, emotional tone, confidence, or audience engagement."},
        {"id": "visual-proxies", "status": "approved", "text": "Pose and hand tracking and sparse frame descriptions do not establish eye contact, audience focus, intent, gesture meaning, or strategic emphasis. State sampling and detection limitations."},
        {"id": "transcript-limits", "status": "approved", "text": "Treat speech recognition as fallible. Quote exact transcript wording, qualify transcript-based claims, and do not infer that undetected filler words or pauses were absent. Segment gaps are not proof of unplanned pauses."},
        {"id": "human-review", "status": "approved", "text": "AI scores are tentative recommendations. Identify criteria that require instructor verification from the original recording."},
        {"id": "category-boundaries", "status": "approved", "text": "Keep rubric categories strictly separate. Vocal Delivery covers audible volume and intonation used for emphasis; Fluency covers speech flow, cadence, pace, pauses, stammering, and filler words; Nonverbal Communication covers observable body language, eye contact only when genuinely visible, and gestures; Word Choice covers vocabulary relevance and precision, technical terminology, jargon, definitions/contextualization, and oversimplification. Do not use intonation, tone, volume, pauses, pace, stammering, or fillers to justify Word Choice. Do not use vocabulary quality to justify Fluency or Vocal Delivery. Do not use transcript fluency to justify Nonverbal Communication. Apply category boundaries to scores, rationale, evidence, strengths, feedback, and adjacent-level reasoning. If a statement is factually supported but belongs to another category, flag it as category-irrelevant. Category contamination in score reasoning requires instructor review; do not silently change the score."}
        ,{"id": "filler-word-context", "status": "approved", "text": "The official Fluency rubric includes filler words at Levels 3, 2, and 1 but does not name specific words. Words such as “like” and “yeah” are not automatically fillers: count an instance only when its audible use is a verbal placeholder or unnecessary habitual discourse marker, not when it has a meaningful grammatical, descriptive, or responsive function. Use the audio and surrounding context to distinguish functions; a transcript alone may be ambiguous or inaccurate. Consider frequency and effect in the full delivery: occasional genuine fillers may align with Level 2 when the other descriptor evidence supports it, while frequent fillers that disrupt cadence or obscure emphasis may align with Level 1. Do not lower a score merely because the words “like” or “yeah” appear in a transcript, and do not treat this note as changing the official rubric."}
    ],
    "borderline_decisions": [
        {"id": "historical-vocal-fluency-example", "status": "instructor-provided-example", "category": ["Vocal Delivery", "Fluency"], "decision": "Instructor preferred Level 2 rather than Level 3 for both categories.", "reported_evidence": "The instructor described a consistently similar tone throughout, excessive volume, pauses the instructor judged strategic, and several filler words.", "interpretation": "This records one instructor judgment based on the combined observations in this particular presentation. Do not generalize it into a rule that any flat-sounding passage, loudness concern, pause, or filler automatically determines a score. For each new presentation, compare the overall observed performance with the full descriptors at the neighboring levels and explain why the identified issue is consequential enough to support the selected level. Strategic pauses alone do not establish Level 4; cadence and emphasis must also be supported.", "scope_warning": "This is one instructor-described example, not a universal rule or an automatic scoring template. Assess each new presentation from its own evidence."}
    ],
    "proposed_refinements": [],
    "notes": "Supplemental interpretation guidance only; the official Oral Communication Badge Rubric remains unchanged. Prior examples calibrate interpretation but must never predetermine a current score."
}

def load_grading_guide():
    GUIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not GUIDE_PATH.exists():
        GUIDE_PATH.write_text(json.dumps(GUIDE_BASELINE, indent=2), encoding="utf-8")
    try:
        data = json.loads(GUIDE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("Unsupported guide format")
        for key in ("approved_guidelines", "borderline_decisions", "proposed_refinements"):
            if not isinstance(data.get(key, []), list): raise ValueError(key + " must be a list")
            data.setdefault(key, [])
        # Add newly introduced baseline guidance to existing local guides without
        # overwriting instructor edits or previously approved decisions.
        existing_ids = {item.get("id") for item in data["approved_guidelines"] if isinstance(item, dict)}
        for item in GUIDE_BASELINE["approved_guidelines"]:
            if item.get("id") not in existing_ids:
                data["approved_guidelines"].append(item.copy())
        return data
    except Exception as exc:
        st.error(f"Could not read saved guide: {exc}")
        return json.loads(json.dumps(GUIDE_BASELINE))

def save_grading_guide(data):
    if not isinstance(data, dict) or data.get("schema_version") != 1: raise ValueError("Invalid guide schema")
    for key in ("approved_guidelines", "borderline_decisions", "proposed_refinements"):
        if not isinstance(data.get(key, []), list): raise ValueError(key + " must be a list")
    GUIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = GUIDE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(GUIDE_PATH)

def guide_prompt_context(guide):
    return json.dumps({"approved_guidelines": guide.get("approved_guidelines", []), "borderline_decisions": guide.get("borderline_decisions", [])}, ensure_ascii=False)[:12000]

def ollama_chat(model: str, messages: list[dict], timeout: int = 900, num_ctx: int | None = None, num_predict: int | None = None) -> str:
    """Call Ollama with bounded output to avoid wasting time on overlong generations."""
    options = {"temperature": 0.1}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if num_predict is not None:
        options["num_predict"] = num_predict
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


def extract_media(video_path: str, work: Path, sample_count: int = 24, progress_callback=None):
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
    sample_times = [duration * (i + 0.5) / sample_count for i in range(sample_count)] if duration else []
    if progress_callback:
        progress_callback(2, "Extracting audio and video", "Audio extracted; sampling representative video frames…")
    if duration:
        for sample_index, sec in enumerate(sample_times):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ok, frame = cap.read()
            if ok:
                scale = min(1.0, 384.0 / max(frame.shape[:2]))
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
                if ok:
                    images.append({"time": round(sec, 1), "base64": base64.b64encode(encoded).decode("ascii")})
            if progress_callback:
                progress_callback(2 + 8 * (sample_index + 1) / max(1, len(sample_times)), "Extracting audio and video", f"Sampled frame {sample_index + 1} of {len(sample_times)}")
    cap.release()
    return audio_path, images, duration

@st.cache_resource(max_entries=1)
def get_whisper_model(model_size: str):
    """Load one Whisper model and reuse it across presentation evaluations."""
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe(audio_path: Path, model_size: str, progress_callback=None, duration: float = 0):
    model = get_whisper_model(model_size)
    if progress_callback:
        progress_callback(10, "Transcribing speech locally", "Speech model loaded; processing recognized segments…")
    segments, info = model.transcribe(str(audio_path), beam_size=5, word_timestamps=False, vad_filter=True)
    result = []
    for seg in segments:
        result.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()})
        if progress_callback and duration > 0:
            fraction = min(1.0, max(0.0, seg.end / duration))
            progress_callback(10 + 20 * fraction, "Transcribing speech locally", f"Processed approximately {fraction:.0%} of the recording")
    return result, info.language, info.language_probability

def make_transcript(segments):
    return "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}s] {s['text']}" for s in segments)

def json_from_text(text: str):
    """Parse model JSON, tolerating a bare abstention marker in a score field."""
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        excerpt = text[:300].replace("\\n", " ")
        raise ValueError(f"The local model did not return a JSON object. Response began: {excerpt!r}")
    payload = match.group(0)
    try:
        return json.loads(payload)
    except json.JSONDecodeError as original_error:
        # Some local models emit score: abstained_insufficient_visual_evidence
        # instead of the required JSON null. Normalize only that bare token form.
        repaired = re.sub(
            r'(:\\s*)abstained_[A-Za-z0-9_-]+(?=\\s*[,}])',
            r'\\1null',
            payload,
            flags=re.I,
        )
        if repaired != payload:
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    parsed.setdefault("_parser_repair", "A bare abstained_* value was normalized to null; review the raw response diagnostics.")
                return parsed
            except json.JSONDecodeError:
                pass
        raise ValueError(
            f"Model response contained malformed JSON ({original_error.msg} at character {original_error.pos}). "
            f"Response excerpt: {payload[:500]!r}"
        ) from original_error

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
- For Fluency, use both WPM measures and explain that segment-based WPM is approximate. Do not treat 150 WPM as a rigid cutoff. Assess possible filler words by their function in context, not by the word alone: “like,” “yeah,” “um,” “uh,” and “you know” are examples that may be fillers but may also have legitimate grammatical, descriptive, or responsive meanings. Count an instance only when the audible use functions as an unnecessary verbal placeholder or habitual speech marker. Use the audio and surrounding context to distinguish these uses; a transcript alone may be ambiguous or misrecognized. Do not count every occurrence or lower the score merely because one of these words appears in the transcript. Consider frequency and effect on the full delivery: occasional genuine fillers may support Level 2 when the other descriptor evidence fits, while frequent fillers that disrupt cadence or obscure key points may support Level 1. Level 3's no-filler descriptor requires evidence that fillers are absent; do not infer absence from an imperfect transcript. Transcription can omit or hallucinate words. Segment gaps are not proof that a pause was unplanned. Never call a pause strategic unless rhetorical context clearly supports emphasis; otherwise describe it neutrally.
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

CATEGORY_SCOPE = {
    "Vocal Delivery": "ONLY audible volume and intonation used for emphasis, including whether intonation is flat or mannered. Exclude speech rate, pauses, fillers, vocabulary, gestures, and gaze.",
    "Fluency": "ONLY speech flow, cadence, pace, pauses, stammering, and filler words. Exclude intonation quality, perceived loudness, vocabulary, gestures, and gaze.",
    "Nonverbal Communication": "ONLY observable body language, posture, and gestures; eye contact only when genuinely verifiable. Exclude vocal delivery, fluency, and vocabulary.",
    "Word Choice": "ONLY vocabulary relevance and precision, correct technical terminology, jargon, plain-language explanation/contextualization, and oversimplification or conceptual distortion caused by word choice. NEVER assess intonation, tone, volume, pauses, speech pace, stammering, or fillers."
}
CATEGORY_SCOPE_FORBIDDEN = {
    "Word Choice": r"\b(?:intonation|pitch|prosody|volume|loudness|pauses?|pacing|speech rate|speaking rate|words per minute|wpm|stammer(?:ing)?|fluency|filler words?|verbal fillers?)\b",
    "Vocal Delivery": r"\b(?:filler(?:s|\s+words?)?|verbal fillers?|disfluenc\w*|stammer(?:ing)?|unplanned pauses?|speech rate|speaking rate|words per minute|wpm|jargon|vocabular(?:y|ies)|word choice|terminolog(?:y|ies)|technical (?:terms?|words?|vocabulary)|speciali[sz]ed (?:terms?|words?|vocabulary)|gestures?|eye contact)\b",
    "Fluency": r"\b(?:intonation|pitch|prosody|volume|loudness|tone of voice|jargon|vocabulary|word choice|gestures?|eye contact)\b",
    "Nonverbal Communication": r"\b(?:intonation|pitch|prosody|volume|loudness|tone of voice|filler words?|verbal fillers?|stammer(?:ing)?|unplanned pauses?|speech rate|speaking rate|words per minute|wpm|jargon|vocabulary|word choice)\b"
}

def category_scope_violations(category, assessment):
    pattern = re.compile(CATEGORY_SCOPE_FORBIDDEN[category], re.I)
    fields = ("rationale", "evidence", "positive_observations", "actionable_feedback",
              "adjacent_level_reasoning", "limitations")
    violations = []
    for field in fields:
        value = assessment.get(field, [])
        values = value if isinstance(value, list) else [value]
        if any(isinstance(entry, str) and pattern.search(entry) for entry in values):
            violations.append(field)
    return sorted(set(violations))

def remove_out_of_scope_content(category, assessment):
    pattern = re.compile(CATEGORY_SCOPE_FORBIDDEN[category], re.I)
    fields = ("rationale", "evidence", "positive_observations", "actionable_feedback",
              "adjacent_level_reasoning", "limitations")
    cleaned = dict(assessment)
    removed = False
    for field in fields:
        value = cleaned.get(field)
        if isinstance(value, list):
            kept = [entry for entry in value if not (isinstance(entry, str) and pattern.search(entry))]
            removed = removed or len(kept) != len(value)
            cleaned[field] = kept
        elif isinstance(value, str) and pattern.search(value):
            sentences = re.split(r"(?<=[.!?])\s+", value)
            replacement = " ".join(sentence for sentence in sentences if not pattern.search(sentence)).strip()
            removed = removed or replacement != value
            cleaned[field] = replacement
    if not cleaned.get("rationale"):
        cleaned["rationale"] = "The generated rationale included material outside this rubric category and was withheld; instructor review is required."
    if not cleaned.get("adjacent_level_reasoning"):
        cleaned["adjacent_level_reasoning"] = "A category-specific adjacent-level comparison could not be retained; instructor review is required."
    if not cleaned.get("evidence"):
        cleaned["evidence"] = []
    if not cleaned.get("limitations"):
        cleaned["limitations"] = "Some generated content was withheld because it crossed rubric-category boundaries."
    return cleaned, removed

def validate_category_claims(text_model, category, assessment, evidence_json):
    prompt = f'''Audit BOTH factual support and rubric-category relevance in this {category} assessment. CATEGORY DEFINITION: {CATEGORY_SCOPE[category]} A claim can be factually true yet irrelevant to this category. Repeated fillers may be supported by a transcript but are NOT Word Choice evidence; intonation/flat tone is NOT Word Choice evidence; vocabulary quality is NOT Fluency evidence. Review rationale, evidence, positive_observations, actionable_feedback, adjacent_level_reasoning, and limitations. Flag category-irrelevant claims separately from unsupported claims. Category contamination in rationale, evidence, or adjacent_level_reasoning is material and requires instructor review of the score. Do not change the score.
DRAFT: {json.dumps(assessment, ensure_ascii=False)}
EVIDENCE: {evidence_json[:18000]}
Flag unsupported inferences, invented quotations, incorrect values, and claims stronger than evidence. Absence of a detected event is not proof it never happened. Acoustic measures do not prove perceived loudness or effective intonation. Tracking does not prove eye contact, intent, audience focus, or strategic gesture meaning. Transcripts may contain recognition errors. Do not change the score.
Return ONLY JSON with keys status (pass or revise), supported_claims (list), unsupported_claims (list), irrelevant_claims (list), corrections (list), score_review_required (boolean).'''
    audit = json_from_text(ollama_chat(text_model, [{"role": "user", "content": prompt}], num_ctx=8192, num_predict=400))
    if audit.get("status") not in ("pass", "revise"): raise ValueError("Invalid audit status")
    for key in ("supported_claims", "unsupported_claims", "irrelevant_claims", "corrections"):
        if not isinstance(audit.get(key), list): raise ValueError("Missing audit list: " + key)
    if not isinstance(audit.get("score_review_required"), bool): raise ValueError("Missing score_review_required boolean")
    if audit["unsupported_claims"] or audit["irrelevant_claims"]: audit["status"] = "revise"
    return audit
def compact_visual_tracking(tracking: dict) -> dict:
    """Keep grading-relevant summaries while omitting the bulky raw frame-by-frame trace."""
    keys = (
        "tracking_fps", "duration_seconds", "observations",
        "pose_landmark_detection_percent", "hand_detection_percent",
        "hand_motion_observations", "hand_motion_percent_of_all_observations",
        "movement_event_count", "summary", "limitations"
    )
    compact = {key: tracking[key] for key in keys if key in tracking}
    compact["movement_events"] = tracking.get("movement_events", [])[:60]
    compact["five_second_windows"] = tracking.get("five_second_windows", [])[:90]
    return compact


def run_grading(text_model: str, vision_model: str, transcript: str, duration: float, segments: list[dict], images: list[dict], audio_metrics: dict, visual_tracking: dict, grading_guide: dict, progress_callback=None):
    visual_notes = "No usable video frames were extracted."
    if images:
        observations = []
        # Multi-image requests preserve each timestamp while cutting sequential
        # vision-model round trips by roughly two thirds.
        batch_size = 3
        if progress_callback:
            progress_callback(60, "Analyzing sampled video frames", f"Sending {len(images)} sampled frames to the vision model in batches…")
        for offset in range(0, len(images), batch_size):
            batch = images[offset:offset + batch_size]
            manifest = "\n".join(
                f"Image {index + 1}: timestamp {image['time']:.1f}s"
                for index, image in enumerate(batch)
            )
            msg = {
                "role": "user",
                "content": (
                    "Describe EACH supplied image separately, in the same order, using its timestamp below. "
                    "For each image, report only visible speaker posture, body orientation, and hand/arm position. "
                    "Do not infer eye contact, intent, engagement, gesture meaning, or behavior between frames. "
                    "Say when a feature is inconclusive. Keep each entry concise.\n" + manifest
                ),
                "images": [image["base64"] for image in batch],
            }
            observations.append(ollama_chat(vision_model, [msg], num_ctx=8192, num_predict=500))
            if progress_callback:
                done = min(len(images), offset + len(batch))
                progress_callback(60 + 10 * done / max(1, len(images)), "Analyzing sampled video frames", f"Completed vision batch {offset // batch_size + 1} of {(len(images) + batch_size - 1) // batch_size}")
        visual_notes = "\n".join(observations)
    if progress_callback:
        progress_callback(70, "Preparing rubric-specific evidence", "Audio, transcript, tracking, and visual evidence are ready.")
    speech_metrics = speech_statistics(segments, duration)
    alignment = align_speech_and_motion(audio_metrics, visual_tracking)
    categories = {}
    guide_context = guide_prompt_context(grading_guide)
    for category_index, (name, rubric) in enumerate(RUBRIC.items()):
        category_start = 70 + category_index * 7.5
        if progress_callback:
            progress_callback(category_start + 0.2, f"Scoring {name}", "Generating this category's separate rubric assessment…")
        if name == "Fluency":
            evidence = {"transcript": transcript[:18000], "speech_metrics": speech_metrics}
            has_evidence = bool(transcript.strip()) and speech_metrics.get("transcript_word_count", 0) > 0
        elif name == "Word Choice":
            # Speech-rate and gap metrics are irrelevant to vocabulary assessment.
            evidence = {"transcript": transcript[:18000]}
            has_evidence = bool(transcript.strip()) and speech_metrics.get("transcript_word_count", 0) > 0
        elif name == "Vocal Delivery":
            # Vocal Delivery is limited to audible volume and intonation.
            # Do not provide transcript text: fillers belong to Fluency and
            # terminology/vocabulary belong to Word Choice.
            evidence = {"acoustic_measurements": audio_metrics}
            has_evidence = bool(audio_metrics.get("segment_features")) or any(audio_metrics.get(k) is not None for k in ("pitch_median_hz", "pitch_10_to_90_range_semitones", "relative_intensity_median_db"))
        else:
            evidence = {"body_hand_tracking": compact_visual_tracking(visual_tracking), "sampled_frame_observations": visual_notes[:12000], "speech_motion_alignment": alignment[:50]}
            has_evidence = visual_tracking.get("observations", 0) > 0 or (bool(images) and bool(visual_notes.strip()))
        evidence_json = json.dumps(evidence, ensure_ascii=False)
        levels = "\n".join(f"Level {n}: {desc}" for n, desc in rubric["levels"].items())
        prompt = f"""Assess ONLY {name} for a university presentation.
Instructor-approved interpretation guidance (supplemental, NOT the official rubric; examples must not predetermine scores):
{guide_context}
Official rubric ({rubric['sub']}):
{levels}
Evidence:
{evidence_json[:20000]}
CATEGORY SCOPE — STRICT SEPARATION:
{CATEGORY_SCOPE[name]}
- Base this category's score ONLY on the criteria named above. Do not import evidence from another category, even if factually correct.
- VOCAL DELIVERY EXCLUSION: Do not discuss, quote, count, characterize, or evaluate filler words, disfluencies, vocabulary, terminology, technical terms, jargon, or word choice. These are not Vocal Delivery criteria. The Vocal Delivery evidence payload contains acoustic measurements only; assess only volume and intonation for emphasis. If an excluded topic seems relevant, omit it entirely rather than mentioning it as a limitation or comparison.
- Apply this boundary to score rationale, evidence, positive observations, actionable feedback, limitations, and adjacent-level reasoning.
- Word Choice must never be justified or penalized by intonation, tone, volume, pauses, pace, stammering, or fillers.
SCORING CALIBRATION:
- Judge the overall performance in this category against the complete official level descriptions. Do not treat the rubric as a mechanical checklist or count isolated imperfections.
- A descriptor saying a performance "may" include a behavior describes what can be consistent with that level; it is not an automatic score trigger. Explain why the observed behavior is representative or consequential enough to make that level the best fit.
- Do not demand perfection for Level 3 or Level 4. Give credit when the evidence demonstrates the relevant level, while acknowledging specific limitations or imperfections proportionately.
- Do not inflate scores either: choose the level whose full description best matches the evidence. A lower score needs positive evidence of the lower-level characteristics, not merely missing proof of higher-level characteristics.
- Distinguish occasional from recurring behavior, and minor from materially disruptive behavior, wherever the evidence permits. Do not exaggerate frequency, severity, or audience impact.
- For borderline decisions, compare the selected level with the nearest higher and lower levels. State which specific descriptor(s) are supported and which are not. Do not claim an adjacent level is ruled out merely because one feature is uncertain.
- Missing or unmeasurable evidence is a limitation, not proof of poor performance. Do not lower a score solely because an automated method cannot verify a rubric dimension.
- Vocal Delivery: the rubric's distinction between Levels 1 and 2 includes appropriateness of volume and the severity/effect of flat or mannered intonation. Acoustic proxies cannot establish perceived room volume or audience impact; do not use them alone to assert that volume is too loud/quiet or that delivery is unengaging. Clearly qualify tentative estimates.
- Fluency: the rubric distinguishes Level 2's occasional issues from Level 1's frequent issues that disrupt cadence or obscure emphasis. Use actual transcript/timing evidence and do not overstate frequency. A WPM estimate alone does not determine a score.
- Nonverbal Communication: distinguish occasional stiffness, limited gestures, or occasional text focus (Level 2 descriptors) from consistent text focus or consistently distracting mannerisms (Level 1 descriptors). Sparse samples/tracking cannot establish these frequencies unless the evidence actually supports them.
- Word Choice: distinguish jargon that obscures meaning or oversimplification (Level 2) from imprecise/incorrect terminology or distorted concepts (Level 1). Do not penalize accurate technical vocabulary merely for being specialized.
GENERAL EVIDENCE RULES: use only evidence above; score this category independently. The transcript is {"NONEMPTY and contains usable spoken words" if transcript.strip() else "EMPTY"}. Never claim there is no transcript or no speech when it is nonempty. Fluency: use approximate rate/timing cautiously; do not invent stammering or fillers. Word Choice: quote exact transcript phrases. Vocal Delivery: use actual acoustic measurements, but they do not prove perceived loudness or effective intonation. Nonverbal: use only visible tracking/frame observations; transcript availability is irrelevant. Do not infer eye contact, intent, engagement or gesture meaning. Assign 1–4 only when supported by this category's evidence; use null only when genuinely insufficient. Never use 1 or null as a placeholder. Explain why the score fits and why the nearest adjacent level is less supported.
OUTPUT FORMAT IS MANDATORY: Return one syntactically valid JSON object and nothing else. The score value must be the JSON number 1, 2, 3, or 4, or the JSON literal null. Never output tokens such as abstained_, abstain, N/A, or explanatory text in place of JSON null. Do not wrap the object in Markdown fences.
FEEDBACK QUALITY:
- Identify at least one genuine, specific strength supported by the supplied evidence when one is present. Do not invent praise or use generic compliments.
- "actionable_feedback" and the overall "Areas for the presenter to improve" section MUST address the presenter and the presentation itself. Phrase each item as advice to the presenter (for example, "Vary your intonation to emphasize key points" or "Contextualize specialized terms for a broad audience").
- Include only a specific, observable, presenter-controllable behavior or presentation choice that the evidence supports and that meaningfully aligns with an improvement opportunity in the rubric. Do not criticize personality, ability, identity, or presumed intent.
- Never put model/system shortcomings, missing measurements, tracking failures, uncertain transcription, unavailable camera views, evidence gaps, or instructor-verification tasks in "actionable_feedback". Those belong exclusively in "limitations" and the evidence-validation notes.
- If no presenter-focused improvement is supported, return an empty "actionable_feedback" list. Do not manufacture criticism to fill the section.
- Separate demonstrated strengths from presenter improvement opportunities. A presentation can have a strong score and still have a useful, limited suggestion; it does not need a manufactured weakness in every category.
- Write feedback in a balanced, respectful instructor voice, addressing the presenter directly where natural ("you could..." / "consider..."). Explain the impact of a meaningful issue without overstating it. Avoid repetitive, exhaustive nitpicking.
- "limitations" is reserved for uncertainty in the evidence or method; it is NOT a list of things the presenter did poorly.
Return ONLY JSON: {{"score": 1-4 or null, "confidence":"low|medium|high", "rationale":"balanced explanation", "evidence":["specific evidence"], "positive_observations":["specific demonstrated strengths"], "actionable_feedback":["specific improvement only if warranted; otherwise empty"], "adjacent_level_reasoning":"...", "limitations":"evidence/method uncertainty only"}}
For Nonverbal Communication specifically: if visual evidence is insufficient, set "score": null (the literal JSON null), explain the evidence limitation in "limitations", and still return every required key. Do not return a standalone abstention token."""
        item, error = None, ""
        attempt_diagnostics = []
        for attempt in range(2):
            retry = "" if attempt == 0 else f" Your previous answer failed validation: {error}. Correct it using the supplied evidence; return complete JSON and do not invent a score."
            raw = ""
            try:
                # Fluency needs room for transcript-specific evidence and the
                # required comparison of adjacent rubric levels.
                output_limit = 1400 if name == "Fluency" else 1000
                raw = ollama_chat(text_model, [{"role": "user", "content": prompt + retry}], num_ctx=8192, num_predict=output_limit)
                if progress_callback:
                    progress_callback(category_start + 3.2, f"Scoring {name}", "Assessment response received; checking its evidence and rubric scope…")
                candidate = json_from_text(raw)
                score = candidate.get("score")
                errors = []
                if score is not None and (isinstance(score, bool) or not isinstance(score, int) or score not in (1, 2, 3, 4)):
                    errors.append("invalid_score_value")
                fields = " ".join(str(candidate.get(k, "")) for k in ("rationale", "evidence", "adjacent_level_reasoning", "limitations")).lower()
                if name in ("Fluency", "Word Choice") and has_evidence and any(x in fields for x in ("no usable speech evidence", "no speech segments", "transcript does not contain", "transcript lacks")):
                    errors.append("response_claimed_speech_evidence_was_missing")
                if has_evidence and score is None:
                    errors.append("abstained_despite_available_category_evidence")
                if not has_evidence and score is not None:
                    errors.append("assigned_score_without_category_evidence")
                for field in ("rationale", "evidence", "adjacent_level_reasoning", "limitations"):
                    if not candidate.get(field):
                        errors.append("missing_or_empty_" + field)
                for field in ("evidence", "positive_observations", "actionable_feedback"):
                    if field in candidate and not isinstance(candidate[field], list):
                        errors.append("invalid_type_" + field)
                for field in ("positive_observations", "actionable_feedback"):
                    if field not in candidate:
                        errors.append("missing_" + field)
                if errors:
                    error = "; ".join(errors)
                    attempt_diagnostics.append({
                        "attempt": attempt + 1, "stage": "assessment_validation",
                        "errors": errors, "raw_response_excerpt": raw[:1800]
                    })
                    continue
                # Deterministic category-boundary check: do not rely on the same
                # language model to police its own cross-category leakage.
                scope_violations = category_scope_violations(name, candidate)
                if scope_violations and attempt == 0:
                    error = "category_scope_violation:" + ",".join(scope_violations)
                    attempt_diagnostics.append({
                        "attempt": attempt + 1, "stage": "category_scope_validation",
                        "errors": [error], "raw_response_excerpt": raw[:1800]
                    })
                    continue
                candidate, scope_content_removed = remove_out_of_scope_content(name, candidate)
                try:
                    if progress_callback:
                        progress_callback(category_start + 4.8, f"Validating {name} evidence", "Checking factual support and category boundaries…")
                    audit = validate_category_claims(text_model, name, candidate, evidence_json)
                    audit_text = json.dumps(audit, ensure_ascii=False)
                    if re.search(CATEGORY_SCOPE_FORBIDDEN[name], audit_text, re.I):
                        scope_content_removed = True
                        audit = {
                            "status": "revise", "supported_claims": [], "unsupported_claims": [],
                            "irrelevant_claims": ["The evidence audit returned material outside this rubric category; instructor review is required."],
                            "corrections": ["Review this category independently against its official descriptor."],
                            "score_review_required": True
                        }
                    candidate["evidence_validation"] = audit
                    if progress_callback:
                        progress_callback(category_start + 7.0, f"Validating {name} evidence", "Evidence validation finished.")
                    if audit.get("score_review_required"):
                        candidate["score_review_required"] = True
                        candidate["confidence"] = "low"
                except Exception as audit_exc:
                    candidate["evidence_validation"] = {"status": "review_required", "unsupported_claims": [], "irrelevant_claims": [], "corrections": [], "score_review_required": True, "error": str(audit_exc)}
                    candidate["score_review_required"] = True
                    candidate["confidence"] = "low"
                if scope_content_removed:
                    candidate["score_review_required"] = True
                    candidate["confidence"] = "low"
                    candidate["evidence_validation"] = {
                        "status": "revise", "supported_claims": [], "unsupported_claims": [],
                        "irrelevant_claims": ["Generated content outside this rubric category was removed before display."],
                        "corrections": ["Review the revised assessment against this category's official descriptor."],
                        "score_review_required": True
                    }
                item = candidate
                break
            except (ValueError, json.JSONDecodeError) as exc:
                error = "invalid_json:" + str(exc)
                attempt_diagnostics.append({
                    "attempt": attempt + 1, "stage": "json_parsing",
                    "errors": [error], "raw_response_excerpt": raw[:1800]
                })
            except Exception as exc:
                error = "model_request_error:" + str(exc)
                attempt_diagnostics.append({
                    "attempt": attempt + 1, "stage": "model_request",
                    "errors": [error], "raw_response_excerpt": raw[:1800]
                })
        if item is None:
            item = {
                "score": None, "confidence": "low",
                "rationale": "No grade is recommended because the assessment did not pass validation.",
                "evidence": [],
                "adjacent_level_reasoning": "A defensible adjacent-level comparison was not returned.",
                "limitations": "The model response could not be validated. See assessment diagnostics for the exact failure.",
                "assessment_diagnostics": {
                    "failure_stage": attempt_diagnostics[-1]["stage"] if attempt_diagnostics else "unknown",
                    "attempts": attempt_diagnostics,
                    "final_validation_errors": error or "No diagnostic details were captured."
                }
            }
        categories[name] = item
        if progress_callback:
            progress_callback(category_start + 7.5, f"Completed {name}", "Moving to the next independent rubric category…")
    strengths = [f"{n}: {point}" for n, v in categories.items() for point in v.get("positive_observations", [])]
    improvements = [f"{n}: {point}" for n, v in categories.items() for point in v.get("actionable_feedback", [])]
    return {"categories": categories, "strengths": strengths[:6], "improvements": improvements[:6], "overall_comments": "Each rubric category was assessed independently. Instructor review is required.", "transcript_language": "See detected language reported during transcription.", "grading_guide": {"path": str(GUIDE_PATH), "approved_guidelines_used": len(grading_guide.get("approved_guidelines", [])), "borderline_decisions_used": len(grading_guide.get("borderline_decisions", []))}}, visual_notes, alignment, speech_metrics
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

grading_guide = load_grading_guide()
with st.sidebar.expander("AI Grading Guide", expanded=False):
    st.caption("Supplemental instructor guidance only. The official rubric remains unchanged.")
    imported_guide = st.file_uploader("Import guide (.json)", type=["json"], key="guide_import")
    if imported_guide is not None and st.button("Import and replace saved guide", key="guide_import_btn"):
        try:
            imported_data = json.loads(imported_guide.getvalue().decode("utf-8"))
            save_grading_guide(imported_data)
            st.session_state["guide_json_editor"] = json.dumps(imported_data, indent=2, ensure_ascii=False)
            st.success("Guide imported; it will be used in the next evaluation.")
            st.rerun()
        except Exception as exc: st.error(f"Import failed: {exc}")
    guide_json = st.text_area("Edit guide JSON", json.dumps(grading_guide, indent=2, ensure_ascii=False), height=260, key="guide_json_editor")
    if st.button("Save grading guide", key="guide_save_btn"):
        try:
            save_grading_guide(json.loads(guide_json))
            st.success("Guide saved locally.")
            st.rerun()
        except Exception as exc: st.error(f"Guide was not saved: {exc}")
    st.download_button("Export grading guide (.json)", json.dumps(grading_guide, indent=2, ensure_ascii=False), "presentation-grading-guide.json", "application/json", key="guide_export_btn")
    st.caption(f"Local file: {GUIDE_PATH}")
    st.caption("Only approved_guidelines and borderline_decisions are used. proposed_refinements are excluded until instructor approval.")

video = st.file_uploader("Choose a presentation video", type=["mp4", "mov", "mkv", "webm", "avi"])
if video:
    st.video(video.getvalue())
if video and st.button("Analyze presentation", type="primary"):
    try:
        with tempfile.TemporaryDirectory(prefix="presentation_grader_") as tmp:
            work = Path(tmp)
            video_path = work / video.name
            video_path.write_bytes(video.getbuffer())
            progress_bar = st.progress(0.0)
            progress_label = st.empty()
            progress_detail = st.empty()
            def report_progress(percent, label, detail=""):
                progress_bar.progress(min(1.0, max(0.0, percent / 100.0)))
                progress_label.markdown(f"**{label}**")
                if detail:
                    progress_detail.caption(detail)
            report_progress(0, "Preparing analysis", "The progress estimate is approximate; individual model requests can take time.")
            with st.status("Analyzing presentation evidence…", expanded=True) as status:
                report_progress(1, "Extracting audio and video", "Extracting the audio track and sampling representative frames…")
                audio_path, images, duration = extract_media(str(video_path), work, sample_count, report_progress)
                st.write(f"Video duration: {duration:.1f} seconds · frames sampled: {len(images)}")
                status.update(label="Transcribing speech locally…")
                report_progress(10, "Transcribing speech locally", "Loading the selected speech-recognition model…")
                segments, language, language_probability = transcribe(audio_path, whisper_size, report_progress, duration)
                transcript = make_transcript(segments)
                if not transcript.strip():
                    raise RuntimeError("No speech transcript was detected. Check the audio track or try a different Whisper model.")
                st.write(f"Detected language: {language} (confidence {language_probability:.2f})")
                status.update(label="Measuring pitch, intensity, and speech timing…")
                report_progress(30, "Measuring vocal and speech features", "Calculating acoustic measurements and speech timing…")
                audio_metrics = analyze_audio(audio_path, segments)
                report_progress(38, "Tracking body and hand movement", f"Sampling the full video at {tracking_fps} fps; this stage may take a while…")
                status.update(label=f"Tracking body and hand landmarks at {tracking_fps} fps…")
                visual_tracking = track_body_and_hands(str(video_path), target_fps=float(tracking_fps))
                report_progress(55, "Body and hand tracking complete", "Preparing local AI models…")
                st.write(f"Tracked {visual_tracking.get('observations', 0)} time points; pose landmarks detected in {visual_tracking.get('pose_landmark_detection_percent', 0)}% of samples; hands detected in {visual_tracking.get('hand_detection_percent', 0)}%.")
                status.update(label="Checking selected local AI models…")
                report_progress(55, "Checking the rubric reasoning model", "Confirming that the selected local model is available…")
                ensure_ollama_model(text_model)
                report_progress(57, "Checking the visual analysis model", "Confirming that the selected vision model is available…")
                ensure_ollama_model(vision_model)
                status.update(label="Running local rubric analysis…")
                result, visual_notes, speech_motion_alignment, speech_metrics = run_grading(text_model, vision_model, transcript, duration, segments, images, audio_metrics, visual_tracking, grading_guide, report_progress)
                report_progress(100, "Analysis complete", "All four separate rubric assessments and their evidence checks have finished.")
                status.update(label="Analysis complete", state="complete")
            result["transcript"] = transcript
            result["visual_observations"] = visual_notes
            result["body_hand_tracking"] = visual_tracking
            result["speech_motion_alignment"] = speech_motion_alignment
            result["audio_measurements"] = audio_metrics
            result["speech_timing_metrics"] = speech_metrics
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
        if item.get("positive_observations"):
            st.markdown("**Demonstrated strengths**")
            for point in item["positive_observations"]:
                st.markdown(f"- {point}")
        if item.get("actionable_feedback"):
            st.markdown("**Constructive next steps**")
            for point in item["actionable_feedback"]:
                st.markdown(f"- {point}")
        if item.get("evidence"):
            st.markdown("**Evidence**")
            for ev in item["evidence"]:
                st.markdown(f"- {ev}")
        if item.get("adjacent_level_reasoning"):
            st.markdown(f"**Adjacent-level reasoning:** {item['adjacent_level_reasoning']}")
        if item.get("limitations"):
            st.markdown(f"**Limitations:** {item['limitations']}")
        diagnostics = item.get("assessment_diagnostics", {})
        if diagnostics:
            with st.expander("Assessment failure diagnostics", expanded=True):
                st.caption(f"Failure stage: {diagnostics.get('failure_stage', 'unknown')}")
                st.json(diagnostics.get("final_validation_errors", "No details captured."))
                st.json(diagnostics.get("attempts", []))
        audit = item.get("evidence_validation", {})
        audit_status = audit.get("status", "not run")
        st.caption(f"Evidence-to-claim validation: {audit_status}")
        if audit.get("score_review_required") or item.get("score_review_required"):
            st.warning("Instructor review required: the assessment may rely on evidence outside this rubric category. The tentative score has not been automatically changed.")
        if audit.get("unsupported_claims"):
            st.warning("Claims the evidence auditor could not verify:")
            for claim in audit["unsupported_claims"]:
                st.markdown(f"- {claim}")
        if audit.get("irrelevant_claims"):
            st.warning("Claims that may be supported but do not belong to this rubric category:")
            for claim in audit["irrelevant_claims"]:
                st.markdown(f"- {claim}")
        if audit.get("corrections"):
            st.markdown("**Suggested evidence-aligned corrections**")
            for correction in audit["corrections"]:
                st.markdown(f"- {correction}")
        if audit.get("error"):
            st.caption(f"Evidence audit error: {audit['error']}")
    st.markdown("### Strengths")
    for x in result.get("strengths", []):
        st.markdown(f"- {x}")
    st.markdown("### Areas for the presenter to improve")
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
