# Local AI Grader (free, local processing)

This is an experimental companion application for Presentation Grader. It runs on the instructor's computer and does not upload the presentation to a hosted AI service. It uses:
- **faster-whisper** locally to create a timestamped transcript.
- **Ollama** locally for text-based rubric analysis and image-based observations.
- **FFmpeg** and OpenCV locally to extract audio and a small sample of video frames.

## Important limitations

This is a first prototype, not a validated grading system.

- It does **not** listen to or interpret vocal tone. Vocal Delivery is intentionally returned as **Not scored**; the instructor must listen and score volume and intonation.
- A sparse set of still frames cannot establish sustained eye contact, gesture frequency, or behavior between frames. The model must leave Nonverbal Communication unscored if the sampled evidence is insufficient.
- Speech recognition can make errors. Verify quotations, filler-word claims, and technical vocabulary against the original video.
- A transcript cannot establish whether a pause was strategic. The model is instructed not to label pauses as strategic based on duration alone.
- Scores are provisional recommendations. The instructor remains responsible for reviewing evidence and assigning the final score.
- Local model output can vary. Do not use scores without checking the cited evidence.

## Requirements

- Windows 10/11, macOS, or Linux.
- Python 3.10 or 3.11 (64-bit).
- FFmpeg installed and available on PATH.
- Ollama installed and running: https://ollama.com/
- A computer with enough RAM for local models. A vision model can require substantial memory; performance depends on hardware.

## Install

1. Install Python and FFmpeg.
2. Install Ollama from https://ollama.com/ and start it.
3. Open a terminal in this folder and install Python dependencies:

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

4. Download local models (the downloads are large and model requirements vary by computer):

   ```bash
   ollama pull qwen2.5:7b
   ollama pull qwen2.5vl:7b
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

6. Open the local address Streamlit prints (usually http://localhost:8501), upload a video, and run the analysis.

## Privacy and cost

The video, extracted audio, sampled frames, and transcript are processed on the instructor's computer. Ollama runs the models locally. No API key or paid AI service is used by this application. Installing/downloading models requires an internet connection initially; model downloads use disk space. The app does not send presentation content to an external service.

## Output

The app creates a downloadable JSON evaluation. It includes category recommendations, rationale, evidence, confidence, limitations, transcript, and visual observations. The JSON can be retained as a record. Importing these recommendations directly into the hosted evaluation form is a planned integration step.
