#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "First install: python3 -m venv .venv"
  echo "Then run: .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi
.venv/bin/python -m streamlit run app.py
