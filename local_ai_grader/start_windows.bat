@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First install: py -m venv .venv
  echo Then run: .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app.py
pause
