#!/usr/bin/env python3
"""
db_manager.py
-------------
Single source of truth for all SQLite access in the palm vein system.
No other module in this project imports sqlite3 or touches palm_vein.db.
"""

import os
import zlib
import sqlite3

import numpy as np

# ---------------------------------------------------------------------------
# Database location
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "palm_vein.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    username     TEXT UNIQUE NOT NULL COLLATE NOCASE,
    enrolled_at  TEXT DEFAULT (datetime('now')),
    active       INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS templates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(id),
    sample_idx   INTEGER NOT NULL DEFAULT 0,
    vr_blob      BLOB NOT NULL,
    vi_blob      BLOB NOT NULL,
    signature    BLOB NOT NULL,
    vr_mean      REAL,
    vi_mean      REAL,
    enrolled_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(user_id, sample_idx)
);

CREATE TABLE IF NOT EXISTS access_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER,
    score        REAL NOT NULL,
    accepted     INTEGER NOT NULL,
    scan_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS meta (
    key          TEXT PRIMARY KEY,
    value        TEXT
);

CREATE INDEX IF NOT EXISTS idx_templates_user ON templates(user_id);
CREATE INDEX IF NOT EXISTS idx_users_active   ON users(active, username);
"""


# ---------------------------------------------------------------------------
# Signature helper (module-level so search_engine.py can import it directly)
# ---------------------------------------------------------------------------

def compute_signature(VR: np.ndarray) -> np.ndarray:
    """
    Compute a compact 16-float signature from a 256x256 binary VR array.
    Used for fast Layer 1 pre-filtering.

    Steps:
        1. block_means = VR.reshape(8, 32, 8, 32).mean(axis=(1, 3))  -> (8,8)
        2. sig = block_means.reshape(4, 2, 4, 2).mean(axis=(1, 3))   -> (4,4)
        3. return sig.flatten().astype(np.float32)                    -> (16,)
    """
    block_means = VR.reshape(8, 32, 8, 32).mean(axis=(1, 3))
    sig = block_means.reshape(4, 2, 4, 2).mean(axis=(1, 3))
    return sig.flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db():
    """
    Create tables, indexes, and meta row if they don't exist.
    Call this once at application startup before anything else.
    Creates the data/ directory if needed.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(_SCHEMA)
        conn.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', '1')")
        conn.commit()


# ---------------------------------------------------------------------------
# User existence check
# ---------------------------------------------------------------------------

def user_exists(username: str) -> bool:
    """Return True if username already enrolled and active."""
    username = username.strip().lower()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id FROM users WHERE username = ? AND active = 1",
            (username,)
        ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

def enroll_user(username: str, veincode_list: list) -> int:
    """
    Enroll a new user with multiple VeinCode templates.

    username     : str — must be unique, will be lowercased and stripped
    veincode_list: list of dicts [{'VR': ndarray, 'VI': ndarray}, ...]
                   Minimum 1, recommended 3.

    Returns user_id (int).
    Raises ValueError if username already active.
    """
    username = username.strip().lower()

    if user_exists(username):
        raise ValueError(f"User '{username}' is already enrolled and active.")

    with sqlite3.connect(DB_PATH) as conn:
        try:
            cursor = conn.execute(
                "INSERT INTO users (username) VALUES (?)", (username,)
            )
            user_id = cursor.lastrowid

            for sample_idx, code in enumerate(veincode_list):
                VR = code['VR'].astype(np.uint8)
                VI = code['VI'].astype(np.uint8)

                vr_blob   = zlib.compress(VR.tobytes())
                vi_blob   = zlib.compress(VI.tobytes())
                sig       = compute_signature(VR)
                sig_blob  = sig.tobytes()
                vr_mean   = float(VR.mean())
                vi_mean   = float(VI.mean())

                conn.execute(
                    """
                    INSERT INTO templates
                        (user_id, sample_idx, vr_blob, vi_blob,
                         signature, vr_mean, vi_mean)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, sample_idx, vr_blob, vi_blob,
                     sig_blob, vr_mean, vi_mean)
                )

            conn.commit()
        except Exception:
            conn.rollback()
            raise

    return user_id


# ---------------------------------------------------------------------------
# Signature cache loader
# ---------------------------------------------------------------------------

def get_all_signatures() -> dict:
    """
    Load all active users' template signatures from DB into memory.

    Returns:
    {
        'matrix':       np.ndarray shape (N, 16) float32,
        'template_ids': list of int,
        'user_ids':     list of int,
    }
    N = total number of templates across all active users.
    If no templates exist, returns empty arrays (shape (0,16), [], []).
    """
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT t.id, t.user_id, t.signature
            FROM   templates t
            JOIN   users u ON u.id = t.user_id
            WHERE  u.active = 1
            ORDER  BY t.id
            """
        ).fetchall()

    if not rows:
        return {
            'matrix':       np.zeros((0, 16), dtype=np.float32),
            'template_ids': [],
            'user_ids':     [],
        }

    template_ids = []
    user_ids     = []
    sigs         = []

    for tid, uid, sig_blob in rows:
        template_ids.append(tid)
        user_ids.append(uid)
        sigs.append(np.frombuffer(sig_blob, dtype=np.float32))

    return {
        'matrix':       np.stack(sigs, axis=0),
        'template_ids': template_ids,
        'user_ids':     user_ids,
    }


# ---------------------------------------------------------------------------
# Template loader
# ---------------------------------------------------------------------------

def get_templates_by_ids(template_ids: list) -> list:
    """
    Load and decompress full VeinCode templates for given template IDs.
    Returns list of dicts [{'VR': ndarray, 'VI': ndarray}, ...] in same
    order as template_ids input.
    """
    if not template_ids:
        return []

    placeholders = ",".join("?" for _ in template_ids)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"SELECT id, vr_blob, vi_blob FROM templates WHERE id IN ({placeholders})",
            template_ids
        ).fetchall()

    row_map = {}
    for tid, vr_blob, vi_blob in rows:
        VR = np.frombuffer(zlib.decompress(vr_blob), dtype=np.uint8).reshape(256, 256)
        VI = np.frombuffer(zlib.decompress(vi_blob), dtype=np.uint8).reshape(256, 256)
        row_map[tid] = {'VR': VR, 'VI': VI}

    return [row_map[tid] for tid in template_ids if tid in row_map]


# ---------------------------------------------------------------------------
# Username lookup
# ---------------------------------------------------------------------------

def get_username(user_id: int) -> str:
    """Return username for a given user_id. Raises KeyError if not found."""
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT username FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if row is None:
        raise KeyError(f"No user found with id={user_id}")
    return row[0]


# ---------------------------------------------------------------------------
# Access log
# ---------------------------------------------------------------------------

def log_access(user_id, score: float, accepted: bool):
    """
    Insert one row into access_log.
    user_id: int or None (None if rejected / unknown)
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO access_log (user_id, score, accepted) VALUES (?, ?, ?)",
            (user_id, float(score), int(accepted))
        )
        conn.commit()


# ---------------------------------------------------------------------------
# User listing
# ---------------------------------------------------------------------------

def list_users() -> list:
    """
    Return list of dicts for all active users:
    [{'username': str, 'sample_count': int, 'enrolled_at': str}, ...]
    Ordered by enrolled_at DESC.
    """
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT u.username,
                   COUNT(t.id) AS sample_count,
                   u.enrolled_at
            FROM   users u
            LEFT JOIN templates t ON t.user_id = u.id
            WHERE  u.active = 1
            GROUP  BY u.id
            ORDER  BY u.enrolled_at DESC
            """
        ).fetchall()

    return [
        {'username': row[0], 'sample_count': row[1], 'enrolled_at': row[2]}
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Soft-delete
# ---------------------------------------------------------------------------

def delete_user(username: str):
    """
    Soft-delete: SET active=0 for the user.
    Templates remain in DB (audit trail). They are excluded from searches
    because get_all_signatures() only fetches active users.
    Raises ValueError if username not found or already inactive.
    """
    username = username.strip().lower()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id, active FROM users WHERE username = ?", (username,)
        ).fetchone()

        if row is None:
            raise ValueError(f"User '{username}' not found.")
        if row[1] == 0:
            raise ValueError(f"User '{username}' is already inactive.")

        conn.execute(
            "UPDATE users SET active = 0 WHERE username = ?", (username,)
        )
        conn.commit()
