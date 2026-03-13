# run.py
"""
Entry point — run this to launch the Streamlit UI.
Usage: python run.py
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    ui_path = os.path.join("ui", "app.py")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", ui_path,
        "--server.port", "8501",
        "--browser.gatherUsageStats", "false"
    ])