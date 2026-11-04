from __future__ import annotations

import hashlib
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "python" / "python.exe"
APP_DIR = ROOT / "local_ai_grader"
OLLAMA_INSTALLER_URL = "https://ollama.com/download/OllamaSetup.exe"
REQUIRED_MODELS = ("qwen2.5:7b", "qwen2.5vl:7b")
API = "http://127.0.0.1:11434"
LOG_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "PresentationGrader" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[Presentation Grader] {message}", flush=True)


def update_application() -> None:
    """Apply verified code-only updates from the public GitHub main branch."""
    repo = "vbistricky/presentation-grader"
    api = "https://api.github.com/repos/" + repo
    raw = "https://raw.githubusercontent.com/" + repo + "/main/"
    targets = {
        "local_ai_grader/launcher.py": ROOT / "launcher.py",
        "local_ai_grader/app.py": APP_DIR / "app.py",
        "local_ai_grader/media_analysis.py": APP_DIR / "media_analysis.py",
    }
    session = requests.Session()
    session.headers.update({"Accept": "application/vnd.github+json", "User-Agent": "Presentation-Grader-Updater"})
    try:
        commit_response = session.get(api + "/commits/main", timeout=12)
        commit_response.raise_for_status()
        commit = commit_response.json()
        latest_commit = commit["sha"]
        tree_sha = commit["commit"]["tree"]["sha"]
        tree_response = session.get(api + "/git/trees/" + tree_sha + "?recursive=1", timeout=20)
        tree_response.raise_for_status()
        entries = {item["path"]: item["sha"] for item in tree_response.json().get("tree", []) if item.get("type") == "blob"}
    except (requests.RequestException, KeyError, ValueError) as exc:
        log(f"Update check unavailable; continuing with installed version ({exc}).")
        return

    changed = []
    for relative, destination in targets.items():
        expected_sha = entries.get(relative)
        if not expected_sha:
            continue
        try:
            current = destination.read_bytes() if destination.is_file() else b""
            current_sha = hashlib.sha1(b"blob " + str(len(current)).encode("ascii") + b"\0" + current).hexdigest() if current else ""
            if current_sha == expected_sha:
                continue
            response = session.get(raw + relative, timeout=45)
            response.raise_for_status()
            payload = response.content
            actual_sha = hashlib.sha1(b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()
            if actual_sha != expected_sha:
                raise RuntimeError("Git object checksum did not match GitHub's repository tree.")
            changed.append((destination, payload))
        except (requests.RequestException, OSError, RuntimeError) as exc:
            log(f"Could not safely update {relative}; keeping installed copy ({exc}).")
            return

    if not changed:
        log("Presentation Grader is up to date.")
        return

    backups = []
    staged = []
    try:
        for destination, payload in changed:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".update-tmp")
            temporary.write_bytes(payload)
            staged.append((destination, temporary))
        for destination, temporary in staged:
            backup = destination.with_name(destination.name + ".update-backup")
            if destination.exists():
                shutil.copy2(destination, backup)
                backups.append((destination, backup))
            os.replace(temporary, destination)
        for _, backup in backups:
            backup.unlink(missing_ok=True)
        log(f"Updated {len(changed)} application file(s) from GitHub ({latest_commit[:7]}).")
    except OSError as exc:
        for destination, backup in reversed(backups):
            try:
                os.replace(backup, destination)
            except OSError:
                pass
        for _, temporary in staged:
            temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Application update failed and rollback was attempted: {exc}") from exc


def find_ollama() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Ollama" / "ollama.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def download_ollama_installer() -> Path:
    destination = Path(os.environ.get("TEMP", str(Path.home()))) / "PresentationGrader-OllamaSetup.exe"
    log("Ollama is not installed. Downloading the official Windows installer…")
    def report(block_count, block_size, total_size):
        if total_size > 0:
            percent = min(100, int(block_count * block_size * 100 / total_size))
            print(f"\rDownloading Ollama installer: {percent}%", end="", flush=True)
    urllib.request.urlretrieve(OLLAMA_INSTALLER_URL, destination, reporthook=report)
    print()
    if not destination.is_file() or destination.stat().st_size < 10_000_000:
        raise RuntimeError("The Ollama installer download appears incomplete.")
    return destination


def ensure_ollama_installed() -> str:
    executable = find_ollama()
    if executable:
        log(f"Found Ollama: {executable}")
        return executable
    installer = download_ollama_installer()
    log("Installing Ollama for the current Windows user…")
    completed = subprocess.run(
        [str(installer), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"],
        check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Ollama installer exited with code {completed.returncode}.")
    for _ in range(30):
        executable = find_ollama()
        if executable:
            return executable
        time.sleep(2)
    raise RuntimeError("Ollama installation finished, but ollama.exe could not be located. Restart Windows and try again.")


def api_ready() -> bool:
    try:
        response = requests.get(f"{API}/api/tags", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def ensure_ollama_server(executable: str) -> None:
    if api_ready():
        log("Ollama service is already running.")
        return
    log("Starting the local Ollama service…")
    log_file = open(LOG_DIR / "ollama-server.log", "a", encoding="utf-8")
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    subprocess.Popen(
        [executable, "serve"],
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
        close_fds=True
    )
    for _ in range(45):
        if api_ready():
            log("Ollama service is ready.")
            return
        time.sleep(1)
    raise RuntimeError("Ollama did not start. See the Ollama log in %LOCALAPPDATA%\\PresentationGrader\\logs.")


def installed_models() -> set[str]:
    response = requests.get(f"{API}/api/tags", timeout=15)
    response.raise_for_status()
    return {item.get("name", "") for item in response.json().get("models", [])}


def ensure_models(ollama_executable: str) -> None:
    installed = installed_models()
    for model in REQUIRED_MODELS:
        if model in installed or any(name.split(":")[0] == model.split(":")[0] and name.endswith(model.split(":")[-1]) for name in installed):
            log(f"Model already installed: {model}")
            continue
        log(f"Downloading AI model {model}. This may take a while and requires several GB of free disk space…")
        completed = subprocess.run([ollama_executable, "pull", model], check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Downloading {model} failed. Check your internet connection and run Presentation Grader again.")


def choose_port() -> int:
    for port in range(8501, 8521):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available local port was found in the range 8501–8520.")


def launch_app() -> None:
    import requests as req
    port = choose_port()
    url = f"http://127.0.0.1:{port}"
    log("Starting Presentation Grader…")
    log_file = open(LOG_DIR / "presentation-grader.log", "a", encoding="utf-8")
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    process = subprocess.Popen(
        [
            str(PYTHON), "-m", "streamlit", "run", str(APP_DIR / "app.py"),
            "--server.headless", "true",
            "--server.address", "127.0.0.1",
            "--server.port", str(port),
            "--browser.gatherUsageStats", "false"
        ],
        cwd=str(APP_DIR),
        stdin=subprocess.DEVNULL,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
        close_fds=True
    )
    for _ in range(60):
        if process.poll() is not None:
            raise RuntimeError("The local app stopped while starting. See %LOCALAPPDATA%\\PresentationGrader\\logs\\presentation-grader.log.")
        try:
            response = req.get(f"{url}/_stcore/health", timeout=1)
            if response.ok:
                log("Opening the app in your browser.")
                os.startfile(url)
                return
        except req.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("Presentation Grader took too long to start. See the local app log for details.")


def main() -> int:
    if not PYTHON.is_file():
        raise RuntimeError("The bundled Python runtime is missing. Reinstall Presentation Grader.")
    log("Checking for application updates. Only changed application files are downloaded; settings and AI models are kept.")
    update_application()
    log("Checking local AI components. The first launch downloads Ollama and its AI models automatically.")
    ollama = ensure_ollama_installed()
    ensure_ollama_server(ollama)
    ensure_models(ollama)
    launch_app()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"ERROR: {exc}")
        input("Press Enter to close this window…")
        raise SystemExit(1)
