# ingestion/loader.py
import os
import git
import tempfile
from pathlib import Path

# File types we care about
SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".go": "go",
    ".rb": "ruby",
    ".rs": "rust",
}

# Folders to always skip
SKIP_DIRS = {
    "venv", ".venv", "node_modules", ".git",
    "__pycache__", ".pytest_cache", "dist", "build",
    ".idea", ".vscode", "migrations"
}


def load_from_github(github_url: str) -> list[dict]:
    """
    Clone a GitHub repo into a temp folder and load all code files.
    Returns list of file dicts.
    """
    print(f"📥 Cloning repo: {github_url}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        git.Repo.clone_from(github_url, tmp_dir)
        files = load_from_directory(tmp_dir)

    print(f"✅ Loaded {len(files)} files from GitHub")
    return files


def load_from_directory(directory: str) -> list[dict]:
    """
    Walk a local directory and load all supported code files.
    Returns list of file dicts with: path, language, content
    """
    files = []
    root = Path(directory)

    for filepath in root.rglob("*"):

        # Skip unwanted directories
        if any(skip in filepath.parts for skip in SKIP_DIRS):
            continue

        # Only process supported file types
        ext = filepath.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        # Skip empty or huge files
        try:
            size = filepath.stat().st_size
            if size == 0 or size > 500_000:  # skip files > 500KB
                continue

            content = filepath.read_text(encoding="utf-8", errors="ignore")

            files.append({
                "file_path": str(filepath.relative_to(root)),
                "absolute_path": str(filepath),
                "language": SUPPORTED_EXTENSIONS[ext],
                "content": content,
                "size_bytes": size
            })

        except Exception as e:
            print(f"⚠️  Could not read {filepath}: {e}")
            continue

    print(f"📂 Found {len(files)} code files in: {directory}")
    return files


def summarize_loaded_files(files: list[dict]) -> None:
    """Print a nice summary of what was loaded."""
    from collections import Counter
    lang_counts = Counter(f["language"] for f in files)

    print("\n📊 Files loaded by language:")
    for lang, count in lang_counts.most_common():
        print(f"   {lang:15} → {count} files")
    print(f"   {'TOTAL':15} → {len(files)} files\n")