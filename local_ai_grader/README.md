# Local AI Grader (free, local processing)

This is an experimental companion application for Presentation Grader. It runs on the instructor's computer and processes presentation media locally. It uses:

- **faster-whisper** to create timestamped speech transcripts.
- **Praat/Parselmouth** to measure pitch and relative-intensity contours across the recording and within recognized speech segments.
- **MediaPipe Pose and Hands** to track visible body and hand landmarks throughout the full video at a configurable sampling rate (default 5 observations per second).
- **OpenCV and FFmpeg** for video and audio processing.
- **Ollama** for local rubric reasoning and supplementary visual descriptions from selected video frames.

## What the upgraded analysis measures

### Audio

The application reports:
- Overall median fundamental frequency (F0), voiced-frame proportion, and 10th–90th percentile pitch range in semitones.
- Relative intensity distribution and variation in the recorded audio.
- Timestamp-aligned pitch range, first-half-to-second-half pitch change, and relative-intensity variation for recognized speech segments.
- Estimated speaking rates and timestamped gaps between Whisper speech segments.

These are descriptive signal measurements, not direct judgments of presentation quality. Pitch variation does not prove effective emphasis. Intensity is not calibrated sound-pressure level and is affected by microphone distance, gain, compression, and recording conditions. A human listener must still judge whether volume is appropriate and intonation effectively supports the message.

### Body and hand tracking

The application processes frames across the entire video at a selected rate (2–10 frames per second; default 5). It reports:
- Pose and hand landmark detection coverage.
- Timestamped hand-position changes and clustered movement intervals.
- Five-second summaries of hand detection, movement, shoulder-line tilt, and torso lateral-offset proxies.
- Shoulder-center movement between consecutive observations.

A hand movement detection is not proof of an intentional gesture. Pose geometry does not establish eye contact, audience focus, confidence, engagement, or whether a gesture strategically emphasizes a point. Camera framing, occlusion, lighting, clothing, and angle can reduce tracking reliability. The application therefore does not automatically assign a Nonverbal Communication score from these measurements; the instructor must inspect the original video.

## Instructor AI Grading Guide and evidence audit

The app maintains an instructor-owned guide at `%LOCALAPPDATA%\\PresentationGrader\\grading_guide.json` on Windows (or the user's local data directory on other platforms). It persists independently of application source updates and includes three separate collections:

- `approved_guidelines`: instructor-approved interpretation guidance that is supplied to the grader.
- `borderline_decisions`: documented close calls and the reasoning used to distinguish adjacent rubric levels; supplied as calibration examples, not scoring templates.
- `proposed_refinements`: suggestions that are not sent to the scoring model until an instructor moves them into approved guidance.

Use **AI Grading Guide** in the sidebar to edit the JSON, import a guide, or export it for backup/sharing with other instructors. Only approved guidelines and borderline decisions are included in the grading prompt. The official rubric text remains separate and unchanged. The guide should use anonymized examples and must not contain student-identifying information.

Each category assessment also receives a separate evidence-to-claim audit from the local model. The audit checks factual statements against that category's supplied transcript, acoustic measurements, or visual tracking evidence; it reports unsupported claims, suggested safer corrections, and audit errors in the evaluation. A passing audit does not prove the grade is correct, and a flagged claim still requires instructor review against the original recording. This audit is an additional model review, not a deterministic fact-checker. The audit now checks **category relevance as well as factual support**: a statement can be true but still be invalid evidence for the category being scored. It separately flags category-irrelevant claims, including delivery observations (intonation, tone, volume, pauses, pace, or fillers) used to justify Word Choice. Contamination in the score rationale, evidence, or adjacent-level reasoning triggers an explicit instructor-review warning and lowers confidence; the app does not silently alter the tentative score. The grading prompt applies the same boundaries to strengths and presenter-facing feedback.

## Rubric category boundaries

The four official categories are evaluated independently:

- **Vocal Delivery:** audible volume and intonation used for emphasis.
- **Fluency:** speech flow, cadence, pace, pauses, stammering, and filler words.
- **Nonverbal Communication:** observable body language, posture, and gestures; eye contact only when genuinely verifiable.
- **Word Choice:** vocabulary relevance and precision, correct technical terminology, jargon, plain-language explanation/contextualization, and oversimplification or conceptual distortion caused by word choice.

In particular, intonation, tone, volume, pauses, pace, stammering, and fillers must never be used to justify or penalize Word Choice. The relevance rule applies to scores, rationales, evidence, strengths, presenter feedback, limitations, and adjacent-level reasoning. The official rubric itself is unchanged; these are supplemental interpretation safeguards.

## Important grading safeguards

- **Vocal Delivery receives a tentative best-fit score when usable audio evidence exists.** Pitch and relative-intensity patterns can support a cautious estimate of vocal variation, but cannot establish perceived room volume or whether intonation is engaging, mannered, defensive, or off-putting. The instructor should listen to the original presentation and confirm the score.
- **Nonverbal Communication receives a tentative best-fit score when the speaker is sufficiently visible.** Tracking and sampled frames can support a cautious estimate based on visible movement, orientation, and posture proxies, but cannot establish audience focus, eye contact, or the rhetorical meaning/strategic use of gestures. The instructor should inspect the original video and confirm the score.
- These categories are no longer automatically left blank just because some criteria cannot be measured. The AI should choose the closest-supported level, mark confidence, explain uncertainty, and identify what needs human verification. It should return no score only when the relevant media evidence is too incomplete or unreliable to support even a tentative estimate.
- Speech recognition may misrecognize or omit words and filler words. Verify quotations, filler-word claims, and technical vocabulary against the original video.
- **Filler-word interpretation:** The official Fluency rubric names filler words but does not list specific words. “Like,” “yeah,” “um,” “uh,” and “you know” count only when they function as unnecessary verbal placeholders or habitual speech markers—not when used meaningfully. Use audio context; do not classify by transcript word matching alone. Consider frequency and impact on cadence when distinguishing occasional from frequent fillers. This is supplemental interpretation guidance and does not change the official rubric.
- Whisper segment boundaries are approximate. Gaps are not automatically unplanned pauses, and WPM estimates are not rigid cutoffs.
- The system does not infer that a pause is strategic based on duration alone.
- All scores are provisional recommendations. The instructor remains responsible for reviewing the evidence and assigning final scores.
- This is an experimental, unvalidated system. Check all observations and scoring rationales against the original presentation.

## Windows desktop installer and portable package

The **Build Windows desktop installer** GitHub Actions workflow creates two Windows x64 downloads:

- **Installer:** `PresentationGraderSetup-Windows-x64.exe` — installs Presentation Grader under the current user's `%LOCALAPPDATA%\\Programs` folder. The Inno Setup package is configured for per-user installation (`PrivilegesRequired=lowest`), not a machine-wide install.
- **Portable package:** `PresentationGrader-Portable-Windows-x64.zip` — extract the ZIP to a folder the user can write to (for example, Documents or Downloads), then run `Launch Presentation Grader.bat`. No Presentation Grader installer is needed.

Both packages include an isolated Python runtime, application dependencies, and FFmpeg. Neither requires the instructor to separately install Python, pip, FFmpeg, or configure a virtual environment. On first launch, the launcher downloads and installs Ollama into the current user's `%LOCALAPPDATA%\\Programs\\Ollama` folder, starts its local service, and downloads the compact local models (`qwen2.5:3b` and `qwen2.5vl:3b`, approximately 5.1 GB combined). The app offers optional 7B models (`qwen2.5:7b` and `qwen2.5vl:7b`) that are downloaded only if an instructor selects them.. Ollama's official Windows documentation states that its installer installs in the user's home directory and does not require Administrator rights. The faster-whisper speech model is downloaded automatically the first time speech recognition is run.

The first run requires an internet connection and approximately 5.1 GB for the two compact default AI models, plus the Whisper `base` model and application files. Selecting both optional 7B models increases the two Ollama model downloads to approximately 10.7 GB combined. Model runtime memory requirements are additional to on-disk sizes. The portable ZIP avoids installing Presentation Grader itself, but it does not eliminate the separate Ollama and model downloads. University device-management policies may still block running downloaded applications, installing Ollama, or downloading large model files; this package does not bypass those restrictions.

### Built-in application updates

The Windows launcher includes an updater inside the same Presentation Grader application. Each time the instructor starts the app, it checks the public GitHub repository for newer copies of the application code, downloads only files whose verified Git blob SHA differs, verifies each downloaded file against GitHub's repository tree, and atomically replaces the changed files. It does not download a separate updater, reinstall the application, or replace instructor settings and locally downloaded AI models. If the network is unavailable or a downloaded file fails verification, the installed version is kept and startup continues.

These automatic updates cover the Python application code (`launcher.py`, `app.py`, and `media_analysis.py`). Changes to Python dependencies, the bundled Python runtime, FFmpeg packaging, or installer configuration still require a new full installer. Updates are checked against the repository's `main` branch, so published code changes become available on the next launch. The launcher may need to restart for its own changes to take effect; the update is applied before the grading interface starts.

The app runs in the instructor's browser but the processing and model inference run locally. A console window remains available during first-run setup so model-download progress and errors are visible. Logs are stored under `%LOCALAPPDATA%\\PresentationGrader\\logs`.

## Development requirements

For running from source instead of using the Windows installer:

- Windows 10 22H2 or newer, macOS, or Linux.
- Python 3.10 or 3.11 (64-bit).
- FFmpeg installed and available on PATH.
- Ollama installed and running: https://ollama.com/
- A computer with sufficient RAM and CPU/GPU resources for local speech and vision models. MediaPipe tracking adds CPU work; increasing tracking FPS increases processing time.

## Install

1. Install Python and FFmpeg.
2. Install Ollama from https://ollama.com/ and start it.
3. Open a terminal in this folder and install dependencies:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   macOS/Linux:
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. Download local models (downloads are large and model requirements vary by computer):

   ```bash
   ollama pull qwen2.5:7b
   ollama pull qwen2.5vl:7b
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

6. Upload a presentation. The original video preview is available in the app. Select the MediaPipe tracking rate and run the analysis.

## Privacy and cost

The video, extracted audio, sampled frames, landmark tracking, and transcript are processed on the instructor's computer. Ollama runs the models locally. No API key or paid AI service is used. Initial software and model downloads require an internet connection and disk space. The app does not send presentation content to a hosted AI service.

## Output

The downloadable JSON includes tentative category recommendations where usable evidence exists, rationale, evidence, confidence, limitations, transcript, timestamped acoustic measurements, speech timing, full-video pose/hand tracking summaries, and supplemental visual observations. Vocal Delivery and Nonverbal Communication scores are explicitly tentative and require instructor verification. The instructor can review the original video alongside the evidence.
