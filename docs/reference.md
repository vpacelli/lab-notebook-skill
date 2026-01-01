# LAB(1) - Research Lab Notebook

## NAME

**lab** — lightweight research notebook with optional semantic search

## SYNOPSIS

```
lab init [--path <dir>]
lab add --body <text> [--title <text>] [--tags <tags>] [--branch <name>] [-m <key=value>...]
lab show <id>
lab edit <id> [--title <text>] [--body <text>] [--tags <tags>] [-m <key=value>...]
lab delete <id> [--force]
lab list [--limit <n>] [--after <date>] [--before <date>] [--tag <tag>] [--branch <name>]
lab recent [<n>]
lab search <query> [--semantic] [--limit <n>]
lab search --where <condition>... [--limit <n>]
lab tags
lab meta-keys
lab meta-values <key>
lab config <key> [<value>]
lab embed [--clear] [--ids <id,...>]
lab context [<query>] [--after <date>] [--limit <n>]
lab export [--format <fmt>] [--output <file>]
```

## DESCRIPTION

**lab** is a command-line research notebook designed for logging experiments, ideas, and decisions during software development and research. It stores entries in a local SQLite database with full-text search, optional semantic search via embeddings, and flexible metadata.

The notebook is optimized for use with AI coding assistants (e.g., Claude Code), providing efficient context retrieval without requiring ingestion of the entire history.

## COMMANDS

### Initialization

**init** [--path *dir*]
:   Initialize a new notebook. Creates `.lab/notebook.db` in the specified directory (default: current directory). Safe to run multiple times; will not overwrite existing data.

### Entry Management

**add** --body *text* [options]
:   Create a new entry. Returns the entry ID.

    **--body** *text*
    :   Entry body content (required). Use `-` to read from stdin.

    **--title** *text*
    :   Short summary title.

    **--tags** *tags*
    :   Comma-separated tags (e.g., `experiment,failed,lstm`).

    **--branch** *name*
    :   Git branch name. If omitted and inside a git repo, auto-detects current branch.

    **--commit**
    :   Attach current git commit hash.

    **--experiment** *id*
    :   Group entries under an experiment identifier.

    **-m**, **--meta** *key=value*
    :   Arbitrary metadata (repeatable). See **METADATA** section.

**show** *id*
:   Display full entry contents and metadata.

**edit** *id* [options]
:   Modify an existing entry. Only specified fields are updated. Recomputes embedding if body/title changed and embeddings are enabled.

    **--title** *text*
    :   New title.

    **--body** *text*
    :   New body content.

    **--tags** *tags*
    :   Replace all tags (comma-separated).

    **--add-tag** *tag*
    :   Add a single tag (repeatable). Ignored if `--tags` is also specified.

    **--remove-tag** *tag*
    :   Remove a single tag (repeatable). Ignored if `--tags` is also specified.

    **-m**, **--meta** *key=value*
    :   Update metadata (repeatable, merges with existing).

**delete** *id* [--force]
:   Delete an entry. Prompts for confirmation unless `--force` is given.

### Listing and Filtering

**list** [options]
:   List entries matching criteria. Displays ID, timestamp, title, and tags.

    **--limit** *n*
    :   Maximum entries to return (default: 20).

    **--after** *date*
    :   Entries after this date (ISO 8601 or natural language: `yesterday`, `2025-01-01`, `"last week"`).

    **--before** *date*
    :   Entries before this date.

    **--tag** *tag*
    :   Filter by tag (repeatable for AND logic).

    **--branch** *name*
    :   Filter by branch.

    **--has-meta** *key*
    :   Entries that have a specific metadata key.

**recent** [*n*]
:   Shorthand for `lab list --limit <n>` ordered by most recent. Default: 10.

**tags**
:   List all tags in use with entry counts.

### Search

**search** *query* [options]
:   Full-text keyword search using SQLite FTS5. Supports prefix matching (`optim*`), phrases (`"learning rate"`), and boolean operators (`gradient AND explode`).

    **--semantic**
    :   Use embedding-based semantic similarity instead of keyword matching. Requires embeddings to be computed. See **SEMANTIC SEARCH** section.

    **--limit** *n*
    :   Maximum results (default: 10).

**search** --where *condition*...
:   Query by metadata conditions. Conditions use SQL-like syntax with `meta.` prefix for metadata fields.

    Examples:
    ```
    lab search --where "meta.lr < 0.01"
    lab search --where "meta.dataset = 'ptb'" --where "meta.converged = true"
    ```

### Metadata Introspection

**meta-keys**
:   List all metadata keys in use across entries, with counts.

**meta-values** *key*
:   List all distinct values for a metadata key, with counts.

### Configuration

**config** *key* [*value*]
:   Get or set configuration. Without *value*, prints current setting.

    **embeddings** (`on`|`off`)
    :   Enable/disable automatic embedding computation for new entries. Default: `off`.

    **auto-branch** (`on`|`off`)
    :   Auto-detect git branch when adding entries. Default: `on`.

    **auto-commit** (`on`|`off`)
    :   Auto-attach git commit hash when adding entries. Default: `off`.

### Embedding Management

**embed** [options]
:   Compute or manage embeddings for semantic search.

    (no options)
    :   Backfill embeddings for all entries that lack them.

    **--clear**
    :   Remove all embeddings from the database.

    **--ids** *id,...*
    :   Only embed specific entries (comma-separated IDs).

    **--model** *name*
    :   Embedding model to use (default: `all-MiniLM-L6-v2`).

### Context Retrieval (for AI assistants)

**context** [*query*] [options]
:   Output a condensed summary suitable for ingestion by an AI assistant. Designed to provide relevant context without transferring the entire notebook.

    (no arguments)
    :   Returns recent entries (default: last 7 days or 20 entries, whichever is more).

    *query*
    :   Natural language query. Uses semantic search if embeddings available, otherwise FTS.

    **--after** *date*
    :   Only include entries after this date.

    **--limit** *n*
    :   Maximum entries to include.

    **--format** (`full`|`summary`|`titles`)
    :   Output verbosity. `summary` (default) includes title, date, tags, and truncated body. `titles` gives one-line-per-entry. `full` includes complete body text.

### Export

**export** [options]
:   Export notebook contents.

    **--format** (`json`|`csv`|`markdown`)
    :   Output format (default: `json`).

    **--output** *file*
    :   Write to file instead of stdout.

## METADATA

Metadata allows arbitrary key-value pairs on entries. Values can be typed:

| Syntax | Type | Example |
|--------|------|---------|
| `key=value` | auto-detect | `epochs=100` → int |
| `key:int=value` | integer | `batch:int=32` |
| `key:float=value` | float | `lr:float=1e-4` |
| `key:bool=value` | boolean | `converged:bool=true` |
| `key:json=value` | JSON | `dims:json=[128,256]` |

Auto-detection tries integer, then float, then keeps as string.

**Examples:**

```bash
lab add --body "..." -m lr=0.001 -m model=transformer -m layers:json=[6,6,6]
lab search --where "meta.lr < 0.01"
lab list --has-meta gpu
```

## SEMANTIC SEARCH

Semantic search uses embedding vectors to find conceptually similar entries, even without keyword matches.

**Setup:**

1. Install dependencies: `pip install sentence-transformers`
2. Enable embeddings: `lab config embeddings on`
3. Backfill existing entries: `lab embed`

**Usage:**

```bash
lab search --semantic "training became unstable"
```

This finds entries about gradient explosions, NaN losses, etc., even if those exact words aren't used.

**Notes:**

- Embeddings add ~1.5KB per entry
- First semantic operation downloads the model (~90MB)
- Queries take ~50ms (negligible for interactive use)
- Partial coverage works; search warns about missing embeddings

## FILES

**.lab/notebook.db**
:   SQLite database containing all entries, configuration, and embeddings.

**.lab/config.json**
:   (Optional) Additional configuration overrides.

## ENVIRONMENT VARIABLES

**LAB_PATH**
:   Override notebook location. Default: searches current directory and parents for `.lab/`.

**LAB_EMBEDDING_MODEL**
:   Override default embedding model.

## EXAMPLES

**Initialize and add first entry:**

```bash
lab init
lab add --title "Project kickoff" --body "Starting work on the new optimizer. Initial plan is to try Adam variants." --tags "planning"
```

**Log an experiment with metadata:**

```bash
lab add \
    --title "AdamW baseline" \
    --body "Ran baseline with AdamW. Converged after 50 epochs. Final loss: 0.34." \
    --tags "experiment,baseline,adamw" \
    --commit \
    -m lr=0.001 -m batch_size=64 -m final_loss:float=0.34 -m converged:bool=true
```

**Find entries about learning rate issues:**

```bash
lab search "learning rate"
lab search --semantic "training was too slow to converge"
```

**Get context for Claude Code:**

```bash
# At start of session
lab context --after "3 days ago"

# When working on specific problem
lab context "gradient clipping experiments"
```

**Track todos:**

```bash
# Add a todo
lab add --title "Refactor data loader" --body "..." --tags "todo,priority-high"

# List open todos
lab list --tag todo

# Mark complete
lab edit 12 --add-tag completed --remove-tag todo
```

**Review experiment metadata:**

```bash
lab meta-keys
lab meta-values lr
lab search --where "meta.lr < 0.001" --where "meta.converged = true"
```

## INTEGRATION WITH CLAUDE CODE

Add to your project's `SKILL.md` or `CLAUDE.md`:

```markdown
## Lab Notebook

This project uses `lab` for research logging. At session start, run:

    lab context --after "7 days ago"

Before starting work on a problem, check for relevant history:

    lab context "brief description of the problem"

After significant experiments, log results:

    lab add --title "..." --body "..." --tags "..." -m key=value
```

## SEE ALSO

sqlite3(1), git(1)

## AUTHOR

Designed collaboratively between human researcher and Claude.

## BUGS

Report issues and feature requests via project repository.