"""
Safe, surgical .env editing: update exactly one KEY=value line, leaving
every other line in the file byte-for-byte untouched (other variables,
comments, blank lines, secrets — all preserved exactly, including line
endings). Used by the voice audition web UI's "Save Selected Voice"
action, which must only ever touch CARTESIA_VOICE_ID.

Not a general .env parser/library — deliberately minimal, since the
only thing writing here is the audition UI, and the only guarantee that
matters is "don't corrupt or drop anything else in this file".
"""

from __future__ import annotations

import re
from pathlib import Path


def update_env_value(env_path: Path, key: str, value: str) -> None:
    """Sets `key=value` in the .env file at `env_path`, replacing an
    existing `key=...` line in place (preserving its line-ending style)
    or appending a new one if the key isn't present yet. Every other
    line is copied through unchanged. Writes atomically (temp file +
    rename) so a crash mid-write can't leave a half-written .env."""
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
        raise ValueError(f"Refusing to write an unsafe-looking env key: {key!r}")

    original = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    lines = original.splitlines(keepends=True)

    key_pattern = re.compile(rf"^{re.escape(key)}=")
    new_lines: list[str] = []
    found = False
    for line in lines:
        body = line.splitlines()[0] if line else line  # strip trailing newline for matching
        if key_pattern.match(body):
            ending = line[len(body):] or "\n"
            new_lines.append(f"{key}={value}{ending}")
            found = True
        else:
            new_lines.append(line)

    if not found:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] = new_lines[-1] + "\n"
        new_lines.append(f"{key}={value}\n")

    tmp_path = env_path.with_name(env_path.name + ".tmp")
    tmp_path.write_text("".join(new_lines), encoding="utf-8")
    tmp_path.replace(env_path)  # atomic on the same filesystem
