#!/usr/bin/env python3
"""
lab - Lightweight research notebook with optional semantic search.

A CLI tool for logging experiments, ideas, and decisions during research
and development. Designed for efficient context retrieval by AI assistants.

Copyright (c) 2026 Vincent Pacelli
Licensed under the MIT License. See LICENSE file in the project root.
"""


import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# === Configuration ===

DEFAULT_NOTEBOOK = "default"
LAB_DIR_NAME = ".lab"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# === Lazy-loaded optional dependencies ===

_embedding_model = None
_dateparser = None


def get_embedding_model():
    """Lazy-load the sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print(
                "Error: Semantic search requires sentence-transformers.\n"
                "Install with: pip install sentence-transformers",
                file=sys.stderr,
            )
            sys.exit(1)
        model_name = os.environ.get("LAB_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string, supporting natural language."""
    if date_str is None:
        return None

    # Try ISO format first
    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass

    # Try relative expressions
    lower = date_str.lower().strip()
    now = datetime.now()

    if lower == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif lower == "yesterday":
        return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif lower == "last week":
        return now - timedelta(weeks=1)
    elif lower == "last month":
        return now - timedelta(days=30)

    # Try "N days/weeks/months ago"
    match = re.match(r"(\d+)\s*(days?|weeks?|months?)\s*ago", lower)
    if match:
        n = int(match.group(1))
        unit = match.group(2)
        if unit.startswith("day"):
            return now - timedelta(days=n)
        elif unit.startswith("week"):
            return now - timedelta(weeks=n)
        elif unit.startswith("month"):
            return now - timedelta(days=n * 30)

    # Try dateparser if available
    global _dateparser
    if _dateparser is None:
        try:
            import dateparser as dp
            _dateparser = dp
        except ImportError:
            _dateparser = False  # Mark as unavailable

    if _dateparser:
        parsed = _dateparser.parse(date_str)
        if parsed:
            return parsed

    print(f"Warning: Could not parse date '{date_str}'", file=sys.stderr)
    return None


# === Database utilities ===


def find_lab_root() -> Optional[Path]:
    """Find the .lab directory, searching up from cwd."""
    if "LAB_PATH" in os.environ:
        return Path(os.environ["LAB_PATH"])

    current = Path.cwd()
    while current != current.parent:
        lab_dir = current / LAB_DIR_NAME
        if lab_dir.is_dir():
            return lab_dir
        current = current.parent

    return None


def get_db_path(notebook: str = None, require_exists: bool = True) -> Path:
    """Get path to the database file for a specific notebook."""
    lab_root = find_lab_root()
    
    # Determine notebook name
    if notebook is None:
        notebook = os.environ.get("LAB_NOTEBOOK", DEFAULT_NOTEBOOK)
    
    db_name = f"{notebook}.db"
    
    if lab_root is None:
        if require_exists:
            print(
                "Error: No lab notebook found. Run 'lab init' first.",
                file=sys.stderr,
            )
            sys.exit(1)
        return Path.cwd() / LAB_DIR_NAME / db_name
    return lab_root / db_name


def get_connection(notebook: str = None, require_exists: bool = True) -> sqlite3.Connection:
    """Get a database connection."""
    db_path = get_db_path(notebook, require_exists)
    
    # Auto-create notebook if .lab/ exists but this specific notebook doesn't
    if not db_path.exists():
        lab_root = find_lab_root()
        if lab_root is not None:
            # .lab/ exists, auto-create this notebook
            init_notebook_db(db_path)
        elif require_exists:
            print(
                "Error: No lab notebook found. Run 'lab init' first.",
                file=sys.stderr,
            )
            sys.exit(1)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_notebook_db(db_path: Path):
    """Initialize a notebook database schema."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            branch TEXT,
            tags TEXT,
            title TEXT,
            body TEXT NOT NULL,
            commit_hash TEXT,
            experiment_id TEXT,
            embedding BLOB,
            meta TEXT
        );

        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
            title, body, tags,
            content=entries,
            content_rowid=id
        );

        CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
            INSERT INTO entries_fts (rowid, title, body, tags)
            VALUES (new.id, new.title, new.body, new.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
            INSERT INTO entries_fts (entries_fts, rowid, title, body, tags)
            VALUES ('delete', old.id, old.title, old.body, old.tags);
        END;

        CREATE TRIGGER IF NOT EXISTS entries_au AFTER UPDATE ON entries BEGIN
            INSERT INTO entries_fts (entries_fts, rowid, title, body, tags)
            VALUES ('delete', old.id, old.title, old.body, old.tags);
            INSERT INTO entries_fts (rowid, title, body, tags)
            VALUES (new.id, new.title, new.body, new.tags);
        END;

        CREATE INDEX IF NOT EXISTS idx_timestamp ON entries(timestamp);
        CREATE INDEX IF NOT EXISTS idx_tags ON entries(tags);
        CREATE INDEX IF NOT EXISTS idx_branch ON entries(branch);
        CREATE INDEX IF NOT EXISTS idx_experiment ON entries(experiment_id);

        INSERT OR IGNORE INTO config (key, value) VALUES ('embeddings', 'off');
        INSERT OR IGNORE INTO config (key, value) VALUES ('auto_branch', 'on');
        INSERT OR IGNORE INTO config (key, value) VALUES ('auto_commit', 'off');
        """
    )
    conn.commit()
    conn.close()


def get_config(conn: sqlite3.Connection, key: str, default: str = None) -> Optional[str]:
    """Get a configuration value."""
    row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_config(conn: sqlite3.Connection, key: str, value: str):
    """Set a configuration value."""
    conn.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
        (key, value),
    )
    conn.commit()


# === Git utilities ===


def get_git_branch() -> Optional[str]:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


# === Embedding utilities ===


def embed_text(text: str) -> bytes:
    """Compute embedding for text, return as bytes."""
    import numpy as np

    model = get_embedding_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.astype(np.float32).tobytes()


def bytes_to_vec(b: bytes):
    """Convert bytes back to numpy vector."""
    import numpy as np

    return np.frombuffer(b, dtype=np.float32)


def compute_entry_embedding(title: str, body: str, meta: dict = None) -> bytes:
    """Compute embedding for an entry, including metadata."""
    text = f"{title or ''}\n{body}"
    if meta:
        meta_str = " ".join(f"{k}: {v}" for k, v in meta.items())
        text += f"\n{meta_str}"
    return embed_text(text)


# === Metadata utilities ===


def parse_meta_arg(arg: str) -> tuple[str, any]:
    """
    Parse a single -m key=value argument.
    Supports type hints: key:int=42, key:float=3.14, key:bool=true, key:json=[1,2,3]
    """
    if "=" not in arg:
        raise ValueError(f"Invalid meta format: {arg} (expected key=value)")

    key, value = arg.split("=", 1)

    # Check for type hint
    if ":" in key:
        key, type_hint = key.rsplit(":", 1)
        if type_hint == "int":
            value = int(value)
        elif type_hint == "float":
            value = float(value)
        elif type_hint == "bool":
            value = value.lower() in ("true", "1", "yes")
        elif type_hint == "json":
            value = json.loads(value)
        else:
            raise ValueError(f"Unknown type hint: {type_hint}")
    else:
        # Auto-detect numbers
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # keep as string

    return key, value


def parse_meta_args(meta_args: list[str]) -> dict:
    """Parse multiple -m arguments into a dict."""
    if not meta_args:
        return None
    result = {}
    for arg in meta_args:
        key, value = parse_meta_arg(arg)
        result[key] = value
    return result


# === Commands ===


def cmd_init(args):
    """Initialize a new notebook."""
    lab_dir = Path(args.path) / LAB_DIR_NAME if args.path else Path.cwd() / LAB_DIR_NAME
    notebook = args.notebook or DEFAULT_NOTEBOOK
    db_path = lab_dir / f"{notebook}.db"

    lab_dir.mkdir(parents=True, exist_ok=True)
    
    if db_path.exists():
        print(f"Notebook '{notebook}' already exists at {lab_dir}")
        return

    init_notebook_db(db_path)
    print(f"Initialized notebook '{notebook}' at {lab_dir}")


def cmd_notebooks(args):
    """List available notebooks."""
    lab_root = find_lab_root()
    
    if lab_root is None:
        print("No lab directory found. Run 'lab init' first.", file=sys.stderr)
        sys.exit(1)
    
    notebooks = []
    for db_file in lab_root.glob("*.db"):
        name = db_file.stem
        # Get entry count
        try:
            conn = sqlite3.connect(db_file)
            count = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            conn.close()
            notebooks.append((name, count))
        except sqlite3.Error:
            notebooks.append((name, "?"))
    
    if not notebooks:
        print("No notebooks found.")
        return
    
    current = os.environ.get("LAB_NOTEBOOK", DEFAULT_NOTEBOOK)
    
    for name, count in sorted(notebooks):
        marker = "*" if name == current else " "
        print(f" {marker} {name}: {count} entries")


def cmd_add(args):
    """Add a new entry."""
    conn = get_connection(args.notebook)

    # Read body from stdin if -
    body = args.body
    if body == "-":
        body = sys.stdin.read()

    # Auto-detect git info
    branch = args.branch
    commit_hash = None

    if branch is None and get_config(conn, "auto_branch", "on") == "on":
        branch = get_git_branch()

    if args.commit or get_config(conn, "auto_commit", "off") == "on":
        commit_hash = get_git_commit()

    # Parse metadata
    meta = parse_meta_args(args.meta)
    meta_json = json.dumps(meta) if meta else None

    # Compute embedding if enabled
    embedding = None
    if get_config(conn, "embeddings", "off") == "on":
        embedding = compute_entry_embedding(args.title, body, meta)

    # Insert entry
    cur = conn.execute(
        """INSERT INTO entries 
           (timestamp, title, body, tags, branch, commit_hash, experiment_id, meta, embedding)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(),
            args.title,
            body,
            args.tags,
            branch,
            commit_hash,
            args.experiment,
            meta_json,
            embedding,
        ),
    )
    conn.commit()

    print(f"Added entry {cur.lastrowid}")
    conn.close()


def cmd_show(args):
    """Show a single entry."""
    conn = get_connection(args.notebook)

    row = conn.execute(
        "SELECT * FROM entries WHERE id = ?", (args.id,)
    ).fetchone()

    if row is None:
        print(f"Entry {args.id} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"ID:         {row['id']}")
    print(f"Timestamp:  {row['timestamp']}")
    if row["title"]:
        print(f"Title:      {row['title']}")
    if row["tags"]:
        print(f"Tags:       {row['tags']}")
    if row["branch"]:
        print(f"Branch:     {row['branch']}")
    if row["commit_hash"]:
        print(f"Commit:     {row['commit_hash']}")
    if row["experiment_id"]:
        print(f"Experiment: {row['experiment_id']}")
    if row["meta"]:
        meta = json.loads(row["meta"])
        print(f"Metadata:   {json.dumps(meta, indent=2)}")
    print(f"Embedded:   {'yes' if row['embedding'] else 'no'}")
    print()
    print(row["body"])

    conn.close()


def cmd_edit(args):
    """Edit an existing entry."""
    conn = get_connection(args.notebook)

    # Fetch existing entry
    row = conn.execute(
        "SELECT * FROM entries WHERE id = ?", (args.id,)
    ).fetchone()

    if row is None:
        print(f"Entry {args.id} not found.", file=sys.stderr)
        sys.exit(1)

    # Determine new values
    title = args.title if args.title is not None else row["title"]
    body = args.body if args.body is not None else row["body"]
    
    # Handle tags: --tags replaces, --add-tag appends, --remove-tag removes
    existing_tags = set(t.strip() for t in (row["tags"] or "").split(",") if t.strip())
    
    if args.tags is not None:
        # Full replacement
        tags = args.tags
    else:
        # Incremental updates
        if args.add_tag:
            for tag in args.add_tag:
                existing_tags.add(tag.strip())
        if args.remove_tag:
            for tag in args.remove_tag:
                existing_tags.discard(tag.strip())
        tags = ",".join(sorted(existing_tags)) if existing_tags else None

    # Handle metadata update (merge with existing)
    existing_meta = json.loads(row["meta"]) if row["meta"] else {}
    if args.meta:
        new_meta = parse_meta_args(args.meta)
        existing_meta.update(new_meta)
    meta_json = json.dumps(existing_meta) if existing_meta else None

    # Recompute embedding if content changed and embeddings enabled
    embedding = row["embedding"]
    content_changed = (args.title is not None or args.body is not None)
    if content_changed and get_config(conn, "embeddings", "off") == "on":
        embedding = compute_entry_embedding(title, body, existing_meta)

    conn.execute(
        """UPDATE entries SET title = ?, body = ?, tags = ?, meta = ?, embedding = ?
           WHERE id = ?""",
        (title, body, tags, meta_json, embedding, args.id),
    )
    conn.commit()

    print(f"Updated entry {args.id}")
    conn.close()


def cmd_delete(args):
    """Delete an entry."""
    conn = get_connection(args.notebook)

    if not args.force:
        row = conn.execute(
            "SELECT title, timestamp FROM entries WHERE id = ?", (args.id,)
        ).fetchone()

        if row is None:
            print(f"Entry {args.id} not found.", file=sys.stderr)
            sys.exit(1)

        title = row["title"] or "(untitled)"
        print(f"Delete entry {args.id}: '{title}' from {row['timestamp']}?")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() != "yes":
            print("Cancelled.")
            return

    conn.execute("DELETE FROM entries WHERE id = ?", (args.id,))
    conn.commit()

    print(f"Deleted entry {args.id}")
    conn.close()


def cmd_list(args):
    """List entries."""
    conn = get_connection(args.notebook)

    conditions = []
    params = []

    if args.after:
        date = parse_date(args.after)
        if date:
            conditions.append("timestamp >= ?")
            params.append(date.isoformat())

    if args.before:
        date = parse_date(args.before)
        if date:
            conditions.append("timestamp <= ?")
            params.append(date.isoformat())

    if args.tag:
        for tag in args.tag:
            conditions.append("tags LIKE ?")
            params.append(f"%{tag}%")

    if args.branch:
        conditions.append("branch = ?")
        params.append(args.branch)

    if args.has_meta:
        conditions.append(f"json_extract(meta, '$.{args.has_meta}') IS NOT NULL")

    where = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
        SELECT id, timestamp, title, tags, branch
        FROM entries
        WHERE {where}
        ORDER BY timestamp DESC
        LIMIT ?
    """
    params.append(args.limit)

    rows = conn.execute(query, params).fetchall()

    if not rows:
        print("No entries found.")
        return

    for row in rows:
        title = row["title"] or "(untitled)"
        tags = f" [{row['tags']}]" if row["tags"] else ""
        branch = f" @{row['branch']}" if row["branch"] else ""
        ts = row["timestamp"][:16]  # Trim to minute
        print(f"{row['id']:4d}  {ts}  {title}{tags}{branch}")

    conn.close()


def cmd_recent(args):
    """Show recent entries."""
    args.limit = args.n
    args.after = None
    args.before = None
    args.tag = None
    args.branch = None
    args.has_meta = None
    cmd_list(args)


def cmd_search(args):
    """Search entries."""
    conn = get_connection(args.notebook)

    if args.where:
        # Metadata search
        results = search_by_meta(conn, args.where, args.limit)
    elif args.semantic:
        # Semantic search
        results = search_semantic(conn, args.query, args.limit)
    else:
        # FTS keyword search
        results = search_fts(conn, args.query, args.limit)

    if not results:
        print("No results found.")
        return

    for r in results:
        title = r.get("title") or "(untitled)"
        score = f" ({r['score']:.3f})" if "score" in r else ""
        tags = f" [{r.get('tags')}]" if r.get("tags") else ""
        ts = r["timestamp"][:16]
        print(f"{r['id']:4d}  {ts}  {title}{tags}{score}")
        if r.get("snippet"):
            # Clean up snippet
            snippet = r["snippet"].replace("\n", " ")[:100]
            print(f"      {snippet}")

    conn.close()


def search_fts(conn: sqlite3.Connection, query: str, limit: int) -> list[dict]:
    """Full-text search using FTS5."""
    rows = conn.execute(
        """SELECT e.id, e.timestamp, e.title, e.tags,
                  snippet(entries_fts, 1, '>>>', '<<<', '...', 32) as snippet
           FROM entries_fts f
           JOIN entries e ON f.rowid = e.id
           WHERE entries_fts MATCH ?
           ORDER BY rank
           LIMIT ?""",
        (query, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def search_semantic(conn: sqlite3.Connection, query: str, limit: int) -> list[dict]:
    """Semantic similarity search."""
    import numpy as np

    # Check coverage
    total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    with_emb = conn.execute(
        "SELECT COUNT(*) FROM entries WHERE embedding IS NOT NULL"
    ).fetchone()[0]

    if with_emb == 0:
        print("No entries have embeddings. Run 'lab embed' first.", file=sys.stderr)
        return []

    if with_emb < total:
        print(
            f"Note: {with_emb}/{total} entries have embeddings. "
            f"Run 'lab embed' for full coverage.\n",
            file=sys.stderr,
        )

    q_vec = bytes_to_vec(embed_text(query))

    rows = conn.execute(
        """SELECT id, timestamp, title, tags, body, embedding 
           FROM entries WHERE embedding IS NOT NULL"""
    ).fetchall()

    scored = []
    for row in rows:
        e_vec = bytes_to_vec(row["embedding"])
        score = float(np.dot(q_vec, e_vec))
        scored.append(
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "title": row["title"],
                "tags": row["tags"],
                "snippet": (row["body"] or "")[:150],
                "score": score,
            }
        )

    scored.sort(key=lambda x: -x["score"])
    return scored[:limit]


def search_by_meta(conn: sqlite3.Connection, conditions: list[str], limit: int) -> list[dict]:
    """Search by metadata conditions."""
    where_clauses = []

    for cond in conditions:
        # Convert meta.field to json_extract
        cond = re.sub(
            r"meta\.(\w+)",
            r"json_extract(meta, '$.\1')",
            cond,
        )
        where_clauses.append(cond)

    query = f"""
        SELECT id, timestamp, title, tags
        FROM entries
        WHERE {' AND '.join(where_clauses)}
        ORDER BY timestamp DESC
        LIMIT ?
    """

    try:
        rows = conn.execute(query, (limit,)).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.OperationalError as e:
        print(f"Query error: {e}", file=sys.stderr)
        return []


def cmd_tags(args):
    """List all tags with counts."""
    conn = get_connection(args.notebook)

    rows = conn.execute("SELECT tags FROM entries WHERE tags IS NOT NULL").fetchall()

    tag_counts = {}
    for row in rows:
        for tag in row["tags"].split(","):
            tag = tag.strip()
            if tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    if not tag_counts:
        print("No tags found.")
        return

    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")

    conn.close()


def cmd_meta_keys(args):
    """List metadata keys in use."""
    conn = get_connection(args.notebook)

    rows = conn.execute(
        "SELECT meta FROM entries WHERE meta IS NOT NULL"
    ).fetchall()

    key_counts = {}
    for row in rows:
        meta = json.loads(row["meta"])
        for key in meta.keys():
            key_counts[key] = key_counts.get(key, 0) + 1

    if not key_counts:
        print("No metadata keys found.")
        return

    for key, count in sorted(key_counts.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")

    conn.close()


def cmd_meta_values(args):
    """List values for a metadata key."""
    conn = get_connection(args.notebook)

    rows = conn.execute(
        f"SELECT json_extract(meta, '$.{args.key}') as val FROM entries WHERE meta IS NOT NULL"
    ).fetchall()

    value_counts = {}
    for row in rows:
        val = row["val"]
        if val is not None:
            value_counts[str(val)] = value_counts.get(str(val), 0) + 1

    if not value_counts:
        print(f"No values found for key '{args.key}'.")
        return

    for val, count in sorted(value_counts.items(), key=lambda x: -x[1]):
        print(f"  {val}: {count}")

    conn.close()


def cmd_config(args):
    """Get or set configuration."""
    conn = get_connection(args.notebook)

    if args.value is None:
        # Get config
        value = get_config(conn, args.key)
        if value is None:
            print(f"Config key '{args.key}' not set.")
        else:
            print(f"{args.key} = {value}")
    else:
        # Set config
        set_config(conn, args.key, args.value)
        print(f"Set {args.key} = {args.value}")

    conn.close()


def cmd_embed(args):
    """Manage embeddings."""
    conn = get_connection(args.notebook)

    if args.clear:
        conn.execute("UPDATE entries SET embedding = NULL")
        conn.commit()
        print("Cleared all embeddings.")
        conn.close()
        return

    # Determine which entries to embed
    if args.ids:
        id_list = [int(x.strip()) for x in args.ids.split(",")]
        placeholders = ",".join("?" * len(id_list))
        rows = conn.execute(
            f"SELECT id, title, body, meta FROM entries WHERE id IN ({placeholders})",
            id_list,
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, title, body, meta FROM entries WHERE embedding IS NULL"
        ).fetchall()

    if not rows:
        print("No entries need embedding.")
        conn.close()
        return

    print(f"Embedding {len(rows)} entries...")

    for i, row in enumerate(rows):
        meta = json.loads(row["meta"]) if row["meta"] else None
        emb = compute_entry_embedding(row["title"], row["body"], meta)
        conn.execute("UPDATE entries SET embedding = ? WHERE id = ?", (emb, row["id"]))

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(rows)}")
            conn.commit()

    conn.commit()
    print("Done.")
    conn.close()


def cmd_context(args):
    """Output context for AI assistant."""
    conn = get_connection(args.notebook)

    limit = args.limit or 20

    if args.query:
        # Use search (semantic if available, else FTS)
        total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        with_emb = conn.execute(
            "SELECT COUNT(*) FROM entries WHERE embedding IS NOT NULL"
        ).fetchone()[0]

        if with_emb > total * 0.5:
            # Majority have embeddings, use semantic
            try:
                results = search_semantic(conn, args.query, limit)
            except Exception:
                results = search_fts(conn, args.query, limit)
        else:
            results = search_fts(conn, args.query, limit)

        entry_ids = [r["id"] for r in results]
        if not entry_ids:
            print("No relevant entries found.")
            return

        placeholders = ",".join("?" * len(entry_ids))
        rows = conn.execute(
            f"SELECT * FROM entries WHERE id IN ({placeholders}) ORDER BY timestamp DESC",
            entry_ids,
        ).fetchall()
    else:
        # Recent entries
        conditions = []
        params = []

        if args.after:
            date = parse_date(args.after)
            if date:
                conditions.append("timestamp >= ?")
                params.append(date.isoformat())
        else:
            # Default: last 7 days
            date = datetime.now() - timedelta(days=7)
            conditions.append("timestamp >= ?")
            params.append(date.isoformat())

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        rows = conn.execute(
            f"SELECT * FROM entries WHERE {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()

    if not rows:
        print("No entries found.")
        return

    # Output based on format
    fmt = args.format or "summary"

    if fmt == "titles":
        for row in rows:
            title = row["title"] or "(untitled)"
            print(f"[{row['id']}] {row['timestamp'][:10]} - {title}")

    elif fmt == "summary":
        for row in rows:
            title = row["title"] or "(untitled)"
            tags = f" [{row['tags']}]" if row["tags"] else ""
            print(f"## [{row['id']}] {title}{tags}")
            print(f"*{row['timestamp'][:16]}*")
            if row["meta"]:
                meta = json.loads(row["meta"])
                print(f"Meta: {json.dumps(meta)}")
            # Truncate body
            body = row["body"] or ""
            if len(body) > 300:
                body = body[:300] + "..."
            print(body)
            print()

    elif fmt == "full":
        for row in rows:
            title = row["title"] or "(untitled)"
            tags = f" [{row['tags']}]" if row["tags"] else ""
            print(f"## [{row['id']}] {title}{tags}")
            print(f"*{row['timestamp']}*")
            if row["branch"]:
                print(f"Branch: {row['branch']}")
            if row["commit_hash"]:
                print(f"Commit: {row['commit_hash']}")
            if row["meta"]:
                meta = json.loads(row["meta"])
                print(f"Meta: {json.dumps(meta, indent=2)}")
            print()
            print(row["body"])
            print()
            print("---")
            print()

    conn.close()


def cmd_export(args):
    """Export notebook contents."""
    conn = get_connection(args.notebook)

    rows = conn.execute(
        "SELECT id, timestamp, title, body, tags, branch, commit_hash, experiment_id, meta FROM entries ORDER BY timestamp"
    ).fetchall()

    entries = []
    for row in rows:
        entry = {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "title": row["title"],
            "body": row["body"],
            "tags": row["tags"],
            "branch": row["branch"],
            "commit_hash": row["commit_hash"],
            "experiment_id": row["experiment_id"],
            "meta": json.loads(row["meta"]) if row["meta"] else None,
        }
        entries.append(entry)

    fmt = args.format or "json"

    if fmt == "json":
        output = json.dumps(entries, indent=2)
    elif fmt == "csv":
        import csv
        import io

        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=["id", "timestamp", "title", "body", "tags", "branch", "meta"],
        )
        writer.writeheader()
        for e in entries:
            e["meta"] = json.dumps(e["meta"]) if e["meta"] else ""
            writer.writerow(e)
        output = buf.getvalue()
    elif fmt == "markdown":
        lines = []
        for e in entries:
            title = e["title"] or "(untitled)"
            lines.append(f"## {title}")
            lines.append(f"*{e['timestamp']}*")
            if e["tags"]:
                lines.append(f"Tags: {e['tags']}")
            lines.append("")
            lines.append(e["body"] or "")
            lines.append("")
            lines.append("---")
            lines.append("")
        output = "\n".join(lines)
    else:
        print(f"Unknown format: {fmt}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Exported to {args.output}")
    else:
        print(output)

    conn.close()


# === CLI Parser ===


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab",
        description="Lightweight research notebook with optional semantic search.",
    )
    
    # Global options
    parser.add_argument(
        "--notebook", "-n",
        help=f"Notebook name (default: '{DEFAULT_NOTEBOOK}', or LAB_NOTEBOOK env var)",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p = subparsers.add_parser("init", help="Initialize a new notebook")
    p.add_argument("--path", help="Directory to initialize in (default: current)")
    p.set_defaults(func=cmd_init)

    # notebooks
    p = subparsers.add_parser("notebooks", help="List available notebooks")
    p.set_defaults(func=cmd_notebooks)

    # add
    p = subparsers.add_parser("add", help="Add a new entry")
    p.add_argument("--body", "-b", required=True, help="Entry body (use '-' for stdin)")
    p.add_argument("--title", "-t", help="Entry title")
    p.add_argument("--tags", help="Comma-separated tags")
    p.add_argument("--branch", help="Git branch (auto-detected if not specified)")
    p.add_argument("--commit", action="store_true", help="Attach current git commit")
    p.add_argument("--experiment", "-e", help="Experiment ID")
    p.add_argument("--meta", "-m", action="append", help="Metadata key=value (repeatable)")
    p.set_defaults(func=cmd_add)

    # show
    p = subparsers.add_parser("show", help="Show an entry")
    p.add_argument("id", type=int, help="Entry ID")
    p.set_defaults(func=cmd_show)

    # edit
    p = subparsers.add_parser("edit", help="Edit an entry")
    p.add_argument("id", type=int, help="Entry ID")
    p.add_argument("--title", "-t", help="New title")
    p.add_argument("--body", "-b", help="New body")
    p.add_argument("--tags", help="Replace all tags")
    p.add_argument("--add-tag", action="append", help="Add a tag (repeatable)")
    p.add_argument("--remove-tag", action="append", help="Remove a tag (repeatable)")
    p.add_argument("--meta", "-m", action="append", help="Metadata to update (repeatable)")
    p.set_defaults(func=cmd_edit)

    # delete
    p = subparsers.add_parser("delete", help="Delete an entry")
    p.add_argument("id", type=int, help="Entry ID")
    p.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    p.set_defaults(func=cmd_delete)

    # list
    p = subparsers.add_parser("list", help="List entries")
    p.add_argument("--limit", "-n", type=int, default=20, help="Maximum entries")
    p.add_argument("--after", help="Entries after date")
    p.add_argument("--before", help="Entries before date")
    p.add_argument("--tag", action="append", help="Filter by tag (repeatable)")
    p.add_argument("--branch", help="Filter by branch")
    p.add_argument("--has-meta", help="Entries with metadata key")
    p.set_defaults(func=cmd_list)

    # recent
    p = subparsers.add_parser("recent", help="Show recent entries")
    p.add_argument("n", type=int, nargs="?", default=10, help="Number of entries")
    p.set_defaults(func=cmd_recent)

    # search
    p = subparsers.add_parser("search", help="Search entries")
    p.add_argument("query", nargs="?", help="Search query")
    p.add_argument("--semantic", "-s", action="store_true", help="Use semantic search")
    p.add_argument("--where", "-w", action="append", help="Metadata condition (repeatable)")
    p.add_argument("--limit", "-n", type=int, default=10, help="Maximum results")
    p.set_defaults(func=cmd_search)

    # tags
    p = subparsers.add_parser("tags", help="List all tags")
    p.set_defaults(func=cmd_tags)

    # meta-keys
    p = subparsers.add_parser("meta-keys", help="List metadata keys")
    p.set_defaults(func=cmd_meta_keys)

    # meta-values
    p = subparsers.add_parser("meta-values", help="List values for a metadata key")
    p.add_argument("key", help="Metadata key")
    p.set_defaults(func=cmd_meta_values)

    # config
    p = subparsers.add_parser("config", help="Get or set configuration")
    p.add_argument("key", help="Config key")
    p.add_argument("value", nargs="?", help="Config value (omit to get current)")
    p.set_defaults(func=cmd_config)

    # embed
    p = subparsers.add_parser("embed", help="Manage embeddings")
    p.add_argument("--clear", action="store_true", help="Remove all embeddings")
    p.add_argument("--ids", help="Only embed specific IDs (comma-separated)")
    p.set_defaults(func=cmd_embed)

    # context
    p = subparsers.add_parser("context", help="Output context for AI assistant")
    p.add_argument("query", nargs="?", help="Search query for relevant context")
    p.add_argument("--after", help="Only entries after date")
    p.add_argument("--limit", "-n", type=int, help="Maximum entries")
    p.add_argument(
        "--format", "-f", choices=["full", "summary", "titles"], help="Output format"
    )
    p.set_defaults(func=cmd_context)

    # export
    p = subparsers.add_parser("export", help="Export notebook")
    p.add_argument("--format", "-f", choices=["json", "csv", "markdown"], help="Format")
    p.add_argument("--output", "-o", help="Output file")
    p.set_defaults(func=cmd_export)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()