"""
Tiny SQLite storage layer — no ORM, just enough to back the dashboard.

Two tables:
  calls: one row per call, holds the raw transcript + raw state JSON
  leads: one row per call, holds the structured extraction (bonus feature)
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "sia.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS calls (
                call_id TEXT PRIMARY KEY,
                phone_number TEXT,
                started_at REAL,
                ended_at REAL,
                state_json TEXT,
                transcript_json TEXT,
                recording_url TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                call_id TEXT PRIMARY KEY,
                extracted_json TEXT,
                created_at REAL,
                FOREIGN KEY (call_id) REFERENCES calls(call_id)
            )
            """
        )


def save_call(call_id: str, phone_number: str, state_dict: dict, recording_url: str | None = None) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO calls (call_id, phone_number, started_at, ended_at, state_json, transcript_json, recording_url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(call_id) DO UPDATE SET
                ended_at=excluded.ended_at,
                state_json=excluded.state_json,
                transcript_json=excluded.transcript_json,
                recording_url=COALESCE(excluded.recording_url, calls.recording_url)
            """,
            (
                call_id,
                phone_number,
                state_dict.get("started_at"),
                state_dict.get("ended_at") or time.time(),
                json.dumps(state_dict),
                json.dumps(state_dict.get("turns", [])),
                recording_url,
            ),
        )


def save_lead(call_id: str, extracted: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO leads (call_id, extracted_json, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(call_id) DO UPDATE SET extracted_json=excluded.extracted_json
            """,
            (call_id, json.dumps(extracted), time.time()),
        )


def get_leads() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.call_id, c.phone_number, c.started_at, c.ended_at, c.recording_url,
                   l.extracted_json
            FROM calls c
            LEFT JOIN leads l ON l.call_id = c.call_id
            ORDER BY c.started_at DESC
            """
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["extracted"] = json.loads(d.pop("extracted_json")) if d.get("extracted_json") else None
            result.append(d)
        return result


def get_call(call_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM calls WHERE call_id = ?", (call_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["state"] = json.loads(d.pop("state_json"))
        d["transcript"] = json.loads(d.pop("transcript_json"))
        lead_row = conn.execute("SELECT extracted_json FROM leads WHERE call_id = ?", (call_id,)).fetchone()
        d["extracted"] = json.loads(lead_row["extracted_json"]) if lead_row else None
        return d
