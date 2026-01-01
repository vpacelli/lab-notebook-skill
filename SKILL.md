---
name: lab-notebook
description: Persistent research notebook for logging experiments, decisions, debugging sessions, and development context. Use when Claude Code needs to (1) record findings, failed attempts, or decisions for future reference, (2) retrieve context from prior work sessions, (3) search for previous experiments or solutions to similar problems, or (4) maintain continuity across sessions. Provides efficient context retrieval without ingesting large files.
---

# Lab Notebook Skill

A lightweight research notebook for logging experiments, decisions, and context during development. Designed for efficient retrieval by Claude Code without ingesting large files.

## Setup

The `lab` command should be available in PATH. If not installed, see the project README for setup instructions.

### Optional Dependencies

```bash
# For semantic search
pip install sentence-transformers numpy

# For natural language dates
pip install dateparser
```

### Initialize in Project

```bash
cd /your/project
lab init
```

This creates `.lab/default.db`.

## Multiple Notebooks

You can maintain separate notebooks for different purposes:

```bash
# Create/use a specific notebook
lab -n experiments add --body "..."
lab -n debugging add --body "..."

# List all notebooks
lab notebooks
#  * default: 15 entries
#    experiments: 42 entries
#    debugging: 7 entries

# Set default via environment variable
export LAB_NOTEBOOK=experiments
lab add --body "..."  # goes to experiments
```

Use cases:
- **default**: General project notes
- **experiments**: Hyperparameter sweeps, model comparisons
- **debugging**: Bug investigations, stack traces
- **literature**: Paper summaries, references

## Usage for Claude Code

### At Session Start

Retrieve recent context:
```bash
lab context --after "7 days ago"
```

Or just titles for orientation:
```bash
lab context --format titles --limit 20
```

### Before Starting Work

Check for relevant prior work:
```bash
lab context "gradient clipping experiments"
```

### After Significant Work

Log findings, decisions, and failed attempts:
```bash
lab add \
    --title "AdamW vs Adam comparison" \
    --body "Tested both optimizers on PTB. AdamW converged faster but final loss was similar. AdamW: 0.34 @ epoch 45, Adam: 0.35 @ epoch 52. Decided to use AdamW for future experiments." \
    --tags "experiment,optimizer,adamw" \
    --commit \
    -m lr=0.001 -m batch_size=64 -m dataset=ptb
```

### Logging Failures

Failed experiments are valuable context:
```bash
lab add \
    --title "Attempted gradient checkpointing" \
    --body "Tried to add gradient checkpointing to reduce memory. Got OOM anyway because the checkpoint boundaries were wrong. Need to checkpoint at attention layers, not FFN. Reverting for now." \
    --tags "failed,memory,gradient-checkpointing"
```

## Commands Reference

### Core Commands

| Command | Purpose |
|---------|---------|
| `lab add --body "..." [--title "..."]` | Create entry |
| `lab show <id>` | View entry |
| `lab edit <id> [--title "..."] [--body "..."]` | Modify entry |
| `lab edit <id> --add-tag TAG` | Add tag to entry |
| `lab edit <id> --remove-tag TAG` | Remove tag from entry |
| `lab delete <id>` | Remove entry |
| `lab list [--after DATE] [--tag TAG]` | List entries |
| `lab recent [N]` | Show N most recent |
| `lab notebooks` | List all notebooks |
| `lab -n NAME <command>` | Run command on specific notebook |

### Search Commands

| Command | Purpose |
|---------|---------|
| `lab search "query"` | Keyword search (FTS5) |
| `lab search --semantic "query"` | Semantic similarity search |
| `lab search --where "meta.lr < 0.01"` | Metadata query |

### Context Commands (for AI assistants)

| Command | Purpose |
|---------|---------|
| `lab context` | Recent entries (last 7 days) |
| `lab context "query"` | Relevant entries for query |
| `lab context --format titles` | Just titles/dates |
| `lab context --format full` | Complete entries |

### Introspection

| Command | Purpose |
|---------|---------|
| `lab tags` | List all tags with counts |
| `lab meta-keys` | List metadata keys in use |
| `lab meta-values <key>` | List values for a key |

### Configuration

| Command | Purpose |
|---------|---------|
| `lab config embeddings on` | Enable auto-embedding |
| `lab config embeddings off` | Disable auto-embedding |
| `lab embed` | Backfill missing embeddings |
| `lab embed --clear` | Remove all embeddings |

## Metadata

Attach arbitrary key-value pairs to entries:

```bash
lab add --body "..." \
    -m lr=0.001 \                    # auto-detected as float
    -m epochs:int=100 \              # explicit int
    -m converged:bool=true \         # explicit bool
    -m layers:json='[128,256,128]'   # JSON array
```

Query by metadata:
```bash
lab search --where "meta.lr < 0.01"
lab search --where "meta.converged = true"
lab list --has-meta gpu
```

## Semantic Search

Semantic search finds conceptually similar entries even without keyword matches.

### Enable

```bash
lab config embeddings on   # new entries get embedded automatically
lab embed                  # backfill existing entries
```

### Use

```bash
lab search --semantic "training became unstable"
# Finds: "gradient norms exploded", "loss went to NaN", etc.
```

### Disable / Reduce Storage

```bash
lab config embeddings off  # stop auto-embedding
lab embed --clear          # remove all embeddings
```

## Best Practices

### What to Log

- **Decisions**: Why you chose approach A over B
- **Failed experiments**: What didn't work and why
- **Hyperparameters**: Specific values that worked
- **Bugs found**: Root causes and fixes
- **External resources**: Papers, docs, Stack Overflow links that were useful

### Tagging Strategy

Use consistent tags:
- `experiment`, `failed`, `success` — outcome
- `hypothesis`, `decision`, `todo` — type
- `model-name`, `dataset-name` — specifics

### Todo Workflow Example

```bash
# Add a todo
lab add --title "Refactor data loader" --body "..." --tags "todo,priority-high"

# List open todos
lab list --tag todo

# Mark complete (adds tag, removes todo)
lab edit 12 --add-tag completed --remove-tag todo

# List completed
lab list --tag completed
```

### When to Use Metadata vs Tags

- **Tags**: Categories, filterable labels
- **Metadata**: Numeric values, structured data you'll query

### Periodic Maintenance

```bash
# Review what's accumulated
lab tags
lab meta-keys

# Export for backup
lab export --format json --output backup.json
```

## Integration with CLAUDE.md

Logging behavior is project-specific. Define policies in your project's CLAUDE.md, for example:

```markdown
## Lab Notebook

After completing any experiment or debugging session, log findings with:

    lab add --title "..." --body "..." --tags "..."

Always include the --commit flag when logging experiments.

Before starting work on a new feature, check for prior context:

    lab context "relevant topic"
```

The skill provides the tool; your CLAUDE.md defines when and how to use it.

## Dependencies Note

The script runs without optional dependencies—semantic search is simply unavailable:

| Dependency | Required for | Behavior if missing |
|------------|--------------|---------------------|
| `sentence-transformers` | Semantic search, embeddings | Clear error when running `lab embed` |
| `dateparser` | Natural language dates | Falls back to basic parsing (`yesterday`, `N days ago`) |
| `numpy` | Embeddings | Installed with sentence-transformers |

All core functionality (add, search, list, context) works with zero dependencies beyond Python 3.10+ and SQLite.

## File Structure

```
your-project/
├── .lab/
│   ├── default.db       # Default notebook
│   ├── experiments.db   # Optional additional notebooks
│   └── ...
└── CLAUDE.md            # Project-specific logging policies
```

The `.lab/` directory can be added to `.gitignore` or committed depending on whether you want notebook history shared.