from __future__ import annotations

import sqlite3
from pathlib import Path


DATABASE_PATH = Path(__file__).resolve().parent / "leaderboard.sqlite3"


SEED_ROWS = [
    {
        "rank": 1,
        "model_name": "GPT-4o",
        "organization": "OpenAI",
        "representation": "Files",
        "accuracy": 0.82,
        "full_accuracy": 0.64,
        "latency_seconds": 18.5,
        "status": "Verified",
    },
    {
        "rank": 2,
        "model_name": "Gemini 3 Flash Preview",
        "organization": "Google",
        "representation": "Raw",
        "accuracy": 0.76,
        "full_accuracy": 0.51,
        "latency_seconds": 21.2,
        "status": "Verified",
    },
    {
        "rank": 3,
        "model_name": "Claude Sonnet 4.5",
        "organization": "Anthropic",
        "representation": "Files",
        "accuracy": 0.74,
        "full_accuracy": 0.49,
        "latency_seconds": 24.9,
        "status": "Pending review",
    },
]


def get_connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS leaderboard_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rank INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                organization TEXT NOT NULL,
                representation TEXT NOT NULL,
                accuracy REAL NOT NULL,
                full_accuracy REAL NOT NULL,
                latency_seconds REAL NOT NULL,
                status TEXT NOT NULL
            )
            """
        )
        row_count = connection.execute(
            "SELECT COUNT(*) FROM leaderboard_entries"
        ).fetchone()[0]
        if row_count:
            return
        connection.executemany(
            """
            INSERT INTO leaderboard_entries (
                rank,
                model_name,
                organization,
                representation,
                accuracy,
                full_accuracy,
                latency_seconds,
                status
            ) VALUES (
                :rank,
                :model_name,
                :organization,
                :representation,
                :accuracy,
                :full_accuracy,
                :latency_seconds,
                :status
            )
            """,
            SEED_ROWS,
        )

