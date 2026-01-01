# Lab Notebook

A lightweight CLI research notebook for logging experiments, decisions, and development context. Designed for efficient context retrieval by AI coding assistants (Claude Code) without ingesting large files.

The instructions in this README are written for Claude Code, but the script `lab.py` itself is nothing more than a command line utility. Similar procedures can be followed for your coding assistant of choice. It can also be used on its own without any AI integration.

## Features

- **Persistent logging** — Entries stored in SQLite, survives across sessions
- **Full-text search** — FTS5-powered keyword search with snippets
- **Semantic search** (optional) — Find conceptually similar entries via embeddings
- **Flexible metadata** — Arbitrary key-value pairs with typed values
- **Multiple notebooks** — Separate namespaces for different concerns
- **Zero required dependencies** — Core functionality works with Python 3.10+ only

## Installation

```bash
# Copy script to PATH
chmod +x lab.py
ln -s /path/to/lab.py ~/.local/bin/lab

# Ensure ~/.local/bin is in PATH (add to ~/.zshrc or ~/.bashrc)
export PATH="$HOME/.local/bin:$PATH"

# Initialize in your project
cd /your/project
lab init
```

This creates `.lab/default.db`.

### Optional Dependencies

```bash
# For semantic search
pip install sentence-transformers numpy

# For natural language date parsing ("last week", "3 days ago")
pip install dateparser
```

### Add to Claude's Permissions

To allow Claude Code to run `lab` without prompting for permission, add to your project's `CLAUDE.md` or `~/.claude/CLAUDE.md`:
```markdown
## Allowed Commands

Run without asking:
- lab
```

Or add to `.claude/settings.json`:
```json
{
  "allowedTools": ["lab", "lab *"]
}
```

## Quick Start

```bash
# Add an entry
lab add --title "Initial experiment" --body "Tested baseline model. Loss: 0.45" --tags "experiment,baseline"

# Add with metadata
lab add --title "Hyperparameter sweep" --body "..." --tags "experiment" \
    -m lr=0.001 -m batch_size=64 -m epochs=100

# List recent entries
lab recent 10

# View specific entry
lab show 1

# Search
lab search "baseline"
lab search --where "meta.lr < 0.01"

# Update entry (add/remove tags)
lab edit 1 --add-tag completed --remove-tag todo

# Get context for AI assistant
lab context --after "7 days ago"
lab context "gradient issues"
```

## Command Reference

### Entry Management

| Command | Purpose |
|---------|---------|
| `lab add --body "..." [--title "..."] [--tags "..."] [-m key=value]` | Create entry |
| `lab show <id>` | View entry |
| `lab edit <id> [--title] [--body] [--add-tag] [--remove-tag] [-m]` | Modify entry |
| `lab delete <id>` | Remove entry |

### Listing & Search

| Command | Purpose |
|---------|---------|
| `lab list [--limit N] [--after DATE] [--tag TAG]` | List entries |
| `lab recent [N]` | Show N most recent (default: 10) |
| `lab search "query"` | Full-text keyword search |
| `lab search --semantic "query"` | Semantic similarity search |
| `lab search --where "meta.field = value"` | Metadata query |

### Introspection

| Command | Purpose |
|---------|---------|
| `lab tags` | List all tags with counts |
| `lab meta-keys` | List metadata keys in use |
| `lab meta-values <key>` | List values for a key |
| `lab notebooks` | List all notebooks |

### Context & Export

| Command | Purpose |
|---------|---------|
| `lab context [query] [--after DATE] [--format FORMAT]` | Output for AI ingestion |
| `lab export [--format json\|csv\|markdown]` | Export notebook |

### Configuration

| Command | Purpose |
|---------|---------|
| `lab config embeddings on\|off` | Toggle auto-embedding for new entries |
| `lab embed` | Backfill embeddings for existing entries |
| `lab embed --clear` | Remove all embeddings |

## Multiple Notebooks

```bash
# Use a specific notebook
lab -n experiments add --body "..."
lab -n todo add --title "Fix bug" --tags "todo"

# List notebooks
lab notebooks

# Set default via environment
export LAB_NOTEBOOK=experiments
```

## Metadata

Attach typed key-value pairs:

```bash
lab add --body "..." \
    -m lr=0.001 \                    # auto-detect float
    -m epochs:int=100 \              # explicit int
    -m converged:bool=true \         # explicit bool
    -m layers:json='[128,256]'       # JSON

# Query
lab search --where "meta.lr < 0.01"
lab search --where "meta.converged = true"
```

## Semantic Search

Find entries by meaning, not just keywords:

```bash
# Enable
lab config embeddings on
lab embed  # backfill existing entries

# Search
lab search --semantic "training became unstable"
# Finds: "gradient explosion", "loss went to NaN", etc.
```

Requires `sentence-transformers` (~90MB model download on first use).

## Claude Code Integration

Add to your project's `CLAUDE.md`:

```markdown
## Lab Notebook

### Session Start
Run `lab context --after "7 days ago"` to review recent work.

### When to Log
- After experiments (successful or failed)
- When discovering bugs or their root causes  
- When making design decisions

### Format
lab add --title "..." --body "..." --tags "..." --commit -m key=value
```

## File Structure

```
your-project/
├── .lab/
│   ├── default.db       # Default notebook
│   └── experiments.db   # Additional notebooks
├── CLAUDE.md            # Project-specific logging instructions
└── ...
```

## Files in This Repo

| File | Purpose |
|------|---------|
| `lab.py` | Main CLI script |
| `SKILL.md` | Skill file for Claude Code (place in skill directory) |
| `docs/reference.md` | A "man page" style reference document |

## Environment Notes

- Uses `#!/usr/bin/env python3` — runs with active mamba/conda environment's Python
- Dependencies are per-environment; semantic search requires `sentence-transformers` in the active env
- `~/.local/bin/lab` works across all environments if PATH is configured

## License

Private / internal use.