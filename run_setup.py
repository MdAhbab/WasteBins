import os
import subprocess
import sys
import threading
import time
import signal

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "waste_manager")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
VENV_DIR = os.path.join(BASE_DIR, ".venv")

# ─── Helper Functions ────────────────────────────────────────────────────────
def get_python_exe():
    """Find the best python executable (prefers .venv)."""
    if os.name == 'nt':
        venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(VENV_DIR, "bin", "python")
        
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable

def run_command(cmd, cwd=None, env=None):
    """Run a synchronous command and check for errors."""
    print(f"\n[EXEC] {' '.join(cmd)} (cwd={cwd or '.'})")
    try:
        subprocess.check_call(cmd, cwd=cwd, env=env, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)

# ─── Setup Phase ─────────────────────────────────────────────────────────────
def setup():
    print(f"=== WasteBins Unified Setup ===")
    
    # 1. Venv Creation
    if not os.path.exists(VENV_DIR):
        print("\n--- Creating Virtual Environment ---")
        subprocess.check_call([sys.executable, "-m", "venv", ".venv"], cwd=BASE_DIR)
        print("  Virtual environment created successfully.")
    
    py_exe = get_python_exe()
    print(f"Using Python: {py_exe}")

    # 2. Backend: Requirements
    requirements_path = os.path.join(BACKEND_DIR, "requirements.txt")
    if os.path.exists(requirements_path):
        print("\n--- Installing Backend Dependencies ---")
        run_command([py_exe, "-m", "pip", "install", "--upgrade", "pip"], cwd=BACKEND_DIR)
        run_command([py_exe, "-m", "pip", "install", "-r", "requirements.txt"], cwd=BACKEND_DIR)

    # 3. Backend: Migrations
    print("\n--- Running Database Migrations ---")
    run_command([py_exe, "manage.py", "migrate"], cwd=BACKEND_DIR)

    # 4. Backend: Node Bootstrapping (Mirpur)
    relocate_script = os.path.join(BACKEND_DIR, "help_relocate.py")
    if os.path.exists(relocate_script):
        print("\n--- Bootstrapping Mirpur Nodes ---")
        run_command([py_exe, "help_relocate.py"], cwd=BACKEND_DIR)

    # 5. Frontend: Dependencies
    print("\n--- Installing Frontend Dependencies ---")
    if os.path.exists(os.path.join(FRONTEND_DIR, "package.json")):
        run_command(["npm", "install"], cwd=FRONTEND_DIR)
    else:
        print("[SKIP] frontend/package.json not found.")

# ─── Execution Phase ──────────────────────────────────────────────────────────
processes = []

def signal_handler(sig, frame):
    print("\n\n=== Shutting down servers... ===")
    for p in processes:
        try:
            p.terminate()
        except:
            pass
    sys.exit(0)

def stream_logs(process, prefix):
    """Stream output from a process with a prefix."""
    for line in iter(process.stdout.readline, b''):
        print(f"[{prefix}] {line.decode(errors='replace').strip()}")

def run_servers():
    py_exe = get_python_exe()
    signal.signal(signal.SIGINT, signal_handler)

    print("\n=== Starting WasteBins Services ===")
    print("Press Ctrl+C to stop all servers.\n")

    # Start Backend
    backend_proc = subprocess.Popen(
        [py_exe, "manage.py", "runserver"],
        cwd=BACKEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True
    )
    processes.append(backend_proc)

    # Start Frontend
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True
    )
    processes.append(frontend_proc)

    # Start Dummy Data Sender (Wait a bit for backend to start)
    print("\n[INFO] Waiting 5s for backend to initialize before starting dummy sender...")
    time.sleep(5)
    dummy_proc = subprocess.Popen(
        [py_exe, "send_dummy_data.py", "--username", "admin", "--password", "ahbab123", "--interval", "10"],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True
    )
    processes.append(dummy_proc)

    # Start logging threads
    t1 = threading.Thread(target=stream_logs, args=(backend_proc, "BACKEND"), daemon=True)
    t2 = threading.Thread(target=stream_logs, args=(frontend_proc, "FRONTEND"), daemon=True)
    t3 = threading.Thread(target=stream_logs, args=(dummy_proc, "DUMMY_SENDER"), daemon=True)
    t1.start()
    t2.start()
    t3.start()

    # Keep main thread alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    try:
        setup()
        run_servers()
    except KeyboardInterrupt:
        signal_handler(None, None)
