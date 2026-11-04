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

## Important grading safeguards

- **Vocal Delivery receives a tentative best-fit score when usable audio evidence exists.** Pitch and relative-intensity patterns can support a cautious estimate of vocal variation, but cannot establish perceived room volume or whether intonation is engaging, mannered, defensive, or off-putting. The instructor should listen to the original presentation and confirm the score.
- **Nonverbal Communication receives a tentative best-fit score when the speaker is sufficiently visible.** Tracking and sampled frames can support a cautious estimate based on visible movement, orientation, and posture proxies, but cannot establish audience focus, eye contact, or the rhetorical meaning/strategic use of gestures. The instructor should inspect the original video and confirm the score.
- These categories are no longer automatically left blank just because some criteria cannot be measured. The AI should choose the closest-supported level, mark confidence, explain uncertainty, and identify what needs human verification. It should return no score only when the relevant media evidence is too incomplete or unreliable to support even a tentative estimate.
- Speech recognition may misrecognize or omit words and filler words. Verify quotations, filler-word claims, and technical vocabulary against the original video.
- Whisper segment boundaries are approximate. Gaps are not automatically unplanned pauses, and WPM estimates are not rigid cutoffs.
- The system does not infer that a pause is strategic based on duration alone.
- All scores are provisional recommendations. The instructor remains responsible for reviewing the evidence and assigning final scores.
- This is an experimental, unvalidated system. Check all observations and scoring rationales against the original presentation.

## Windows desktop installer and portable package

The **Build Windows desktop installer** GitHub Actions workflow creates two Windows x64 downloads:

- **Installer:** `PresentationGraderSetup-Windows-x64.exe` — installs Presentation Grader under the current user's `%LOCALAPPDATA%\\Programs` folder. The Inno Setup package is configured for per-user installation (`PrivilegesRequired=lowest`), not a machine-wide install.
- **Portable package:** `PresentationGrader-Portable-Windows-x64.zip` — extract the ZIP to a folder the user can write to (for example, Documents or Downloads), then run `Launch Presentation Grader.bat`. No Presentation Grader installer is needed.

Both packages include an isolated Python runtime, application dependencies, and FFmpeg. Neither requires the instructor to separately install Python, pip, FFmpeg, or configure a virtual environment. On first launch, the launcher downloads and installs Ollama into the current user's `%LOCALAPPDATA%\\Programs\\Ollama` folder, starts its local service, and downloads the two configured local models (`qwen2.5:7b` and `qwen2.5vl:7b`). Ollama's official Windows documentation states that its installer installs in the user's home directory and does not require Administrator rights. The faster-whisper speech model is downloaded automatically the first time speech recognition is run.

The first run requires an internet connection and several GB of additional disk space; the AI models can require many more GB depending on model versions. The portable ZIP avoids installing Presentation Grader itself, but it does not eliminate the separate Ollama and model downloads. University device-management policies may still block running downloaded applications, installing Ollama, or downloading large model files; this package does not bypass those restrictions.

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
