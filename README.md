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

## Important limitation
This release is an evaluation workspace, **not an automatic AI grader**. It does not transcribe speech, automatically analyze audio or video, or generate grades. It keeps scoring instructor-led and does not infer whether a pause is strategic based only on its duration.

## Privacy
Video is previewed locally. Draft fields are saved in this browser's local storage. Export a JSON copy for backup. Clearing browser data can erase the local draft. Avoid confidential student data on shared devices.

## Deploy
GitHub Actions workflow in `.github/workflows/deploy.yml` deploys the static site to GitHub Pages. In repository Settings → Pages, set Build and deployment → Source to **GitHub Actions**.

No API keys, paid AI services, subscriptions, or external processing services are used in this version.