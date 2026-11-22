"""Fail-closed repository hygiene checks for the deployed WealthPilot path."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOYED_PREFIXES = ("src/", "app.py", "streamlit_app.py")
SECRET_PATTERNS = (
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bsk-(?:proj-|live-|test-)?[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?:['\"])(?:/home/|/Users/|/content/|[A-Za-z]:\\)"
)


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / name for name in result.stdout.decode().split("\0") if name]


def main() -> int:
    failures: list[str] = []
    for path in tracked_files():
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            if not relative.startswith(("tests/", "evals/")) and any(
                pattern.search(line) for pattern in SECRET_PATTERNS
            ):
                failures.append(f"secret-like value at {relative}:{line_number}")
            if relative.startswith(DEPLOYED_PREFIXES) and ABSOLUTE_PATH_PATTERN.search(line):
                failures.append(f"machine-specific path at {relative}:{line_number}")
            if relative.startswith(DEPLOYED_PREFIXES) and re.search(r"\bprint\s*\(", line):
                failures.append(f"stdout print in deployed path at {relative}:{line_number}")

    if failures:
        print("Hygiene check failed:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("H1/H3/H6 repository hygiene checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
