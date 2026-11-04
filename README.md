# Presentation Grader

A free, local-first browser workspace for instructor-led evaluation using the Oral Communication Badge Rubric (Ash Faulkner, 6/30/2026).

## Use online
Visit the GitHub Pages URL for this repository once deployment is enabled.

## Features
- Video preview in the browser (video is not uploaded).
- Four rubric categories with levels 1–4.
- Per-category score and written rationale.
- Evidence log with timestamps captured from video playback.
- Overall strengths, improvement areas, and comments.
- Automatic draft saving in browser local storage.
- JSON export and print / Save as PDF.

## Optional free local AI grader
An experimental companion app is in [local_ai_grader](local_ai_grader/README.md). It uses local Whisper transcription and Ollama text/vision models; presentation files are processed on the instructor's computer. No paid API is used.

The prototype can recommend Fluency and Word Choice scores from a transcript and make cautious observations from sampled video frames. It intentionally leaves Vocal Delivery unscored because it does not listen to vocal tone. It may also leave Nonverbal Communication unscored when sparse frames are insufficient. AI scores are provisional and must be reviewed by the instructor.

## Important limitation
The hosted website is an evaluation workspace. AI analysis runs only in the separately installed local companion app. The app does not infer whether a pause is strategic based only on its duration.

## Privacy
Video is previewed locally on the website. The companion app also processes presentation content locally. Draft fields are saved in this browser's local storage. Export a JSON copy for backup. Clearing browser data can erase the local draft. Avoid confidential student data on shared devices.

## Deploy
GitHub Actions workflow in `.github/workflows/deploy.yml` deploys the static site to GitHub Pages. In repository Settings → Pages, set Build and deployment → Source to **GitHub Actions**.

No API keys, paid AI services, subscriptions, or external processing services are used.
