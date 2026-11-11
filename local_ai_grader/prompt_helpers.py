"""Pure prompt helpers shared by the grading app and regression tests."""


def category_prompt_instructions(category: str) -> str:
    """Return instructions that apply only to the requested rubric category."""
    instructions = {
        "Vocal Delivery": """CATEGORY-SPECIFIC BOUNDARY:
- Assess ONLY audible volume and intonation used for emphasis.
- Do not discuss, quote, count, characterize, or evaluate filler words, disfluencies, speech rate, pauses, vocabulary, terminology, technical terms, jargon, word choice, gestures, or gaze. Omit these topics entirely; do not mention them as limitations or comparisons.
- Acoustic measurements are proxies and do not establish perceived room volume or effective intonation. Qualify conclusions accordingly.""",
        "Fluency": """CATEGORY-SPECIFIC BOUNDARY:
- Assess ONLY speech flow, cadence, pace, pauses, stammering, and filler words.
- Do not evaluate intonation quality, perceived loudness, vocabulary, terminology, jargon, gestures, or gaze.
- Use transcript and timing evidence cautiously; segment gaps do not prove unplanned pauses, and transcript recognition may be imperfect.""",
        "Word Choice": """CATEGORY-SPECIFIC BOUNDARY:
- Assess ONLY vocabulary relevance and precision, technical terminology, jargon, contextualization, and oversimplification.
- Do not evaluate intonation, tone, volume, pauses, speech pace, stammering, fillers, gestures, or gaze.
- Quote exact transcript wording when available; do not call a term incorrect without contextual evidence.""",
        "Nonverbal Communication": """NONVERBAL EVIDENCE INTERPRETATION — IMPORTANT:
- Use sampled-frame descriptions and pose tracking as TWO INDEPENDENT visual evidence sources. A weak result from one does not erase usable evidence from the other.
- If sampled-frame notes describe the presenter’s visible posture, body orientation, or arm/hand positions, discuss those observations even when MediaPipe hand detection is low.
- Low or zero hand-detection percentage means the hand tracker did not reliably detect hands; it does NOT establish that the presenter made no gestures and does NOT invalidate visible body/posture descriptions.
- Pose-detection percentage is coverage, not a performance score. Report it as a limitation and do not treat undetected samples as evidence of poor body language.
- Use visual tracking and sampled-frame observations only. Do not use transcript content, speech timing, pitch, volume, intonation, or other audio information.
- Do not claim eye contact, audience focus, intent, strategic emphasis, or gesture meaning from camera-facing posture or movement alone.
- If sampled frames show the presenter and/or pose detections are available, make a cautious best-fit 1–4 recommendation rather than abstaining solely because hand tracking is weak. Use null only if the presenter is not sufficiently visible or visual observations genuinely cannot support a tentative level.
- If visual evidence is genuinely insufficient, return the JSON literal null for score and explain the specific limitation in the required limitations string. Never return a standalone abstention token.""",
    }
    if category not in instructions:
        raise ValueError(f"Unknown rubric category: {category}")
    return instructions[category]


def ensure_nonempty_limitations(assessment: dict) -> dict:
    """Ensure the required limitations field is a nonempty string."""
    value = assessment.get("limitations")
    if not isinstance(value, str) or not value.strip():
        assessment["limitations"] = (
            "No material evidence limitations were identified in the supplied "
            "category-specific evidence."
        )
    return assessment
