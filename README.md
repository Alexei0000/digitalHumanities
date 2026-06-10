# Novel SNA Pipeline

A large-scale LLM-assisted pipeline for Social Network Analysis of literary fiction. Processes thousands of novels in `.txt` or `.md` format, extracts characters and dialogue, attributes speakers and listeners, scores sentiment, and produces graph-ready CSV files for analysis in Gephi, Cytoscape, or NetworkX.

---

## What It Does

For each novel the pipeline produces:

| File | Contents |
|---|---|
| `characters.json` | All named characters with aliases |
| `dialogues.json` | Every extracted speech act |
| `interaction_log.json` | Speaker → listener events with sentiment scores |
| `nodes.csv` | Character node table for graph tools |
| `edges.csv` | Weighted, directed edge table |

**Node table**

| CharacterID | Name | Aliases | Degree |
|---|---|---|---|
| c_3f2a1b | Elizabeth Bennet | Elizabeth; Lizzy; Miss Bennet | 14 |

**Edge table**

| Source | Target | Interaction_Count | Mean_Sentiment | Co_Occurrence | Weight | Evidence |
|---|---|---|---|---|---|---|
| Darcy | Elizabeth Bennet | 57 | 1.8 | 23 | 73.5 | You must allow me... |

Edge weight formula:

```
W = 1.0 × dialogue_count
  + 0.3 × co_occurrence_count
  + 0.5 × abs(mean_sentiment)
```

Weights are tunable in `src/config.py`.

---

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com) running on a local or remote machine with at least one supported model pulled

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup

**1. Set your Ollama server URL**

Open `src/config.py` and set:

```python
OLLAMA_URL = "http://YOUR_SERVER_IP:11434"
```

**2. Choose a model profile**

```python
ACTIVE_PROFILE = "gemma4_31b"   # best quality
```

See the [Model Profiles](#model-profiles) section for all options.

**3. Add your novels**

Drop `.txt` or `.md` files anywhere inside the `data/` folder. Subdirectories at any depth are supported — the corpus loader walks the entire tree recursively.

```
data/
├── fantasy/
│   ├── tolkien/
│   │   └── fellowship.txt
│   └── sanderson/
│       └── mistborn.txt
└── classic/
    └── pride_and_prejudice.txt
```

**4. Run**

```bash
cd src
python main.py
```

Outputs are written to `output/<novel_id>/` for each novel.

---

## Model Profiles

Switch the entire pipeline by changing one line in `config.py`:

```python
ACTIVE_PROFILE = "gemma4_31b"
```

| Profile | Models | Chunk Size | Best For |
|---|---|---|---|
| `gemma4_31b` | gemma4:31b (all stages) | 8,000 chars | Final research runs, best accuracy |
| `gemma4_mixed` | gemma4:12b extraction + gemma4:31b attribution | 5,000 chars | Good balance of speed and quality |
| `gemma4_12b` | gemma4:12b (all stages) | 3,500 chars | Fast exploratory runs |
| `qwen3_32b` | qwen3:32b (all stages) | 8,000 chars | Alternative high-quality option |
| `qwen3_14b` | qwen3:14b (all stages) | 4,000 chars | Faster Qwen option |

Each profile also sets the chunk size automatically — smaller models get shorter chunks so they can produce reliable JSON output.

To add a custom profile:

```python
MODEL_PROFILES["my_profile"] = {
    "character_model":     "llama3:8b",
    "dialogue_model":      "llama3:8b",
    "speaker_model":       "llama3:8b",
    "listener_model":      "llama3:8b",
    "sentiment_model":     "llama3:8b",
    "max_chars_per_chunk": 3_000,
    "overlap_chars":       400,
}
ACTIVE_PROFILE = "my_profile"
```

---

## Pipeline Stages

Each novel passes through four stages. Completed stages are saved to `pipeline.db` and skipped on re-runs, so the pipeline is safe to interrupt and resume at any point.

```
Novel (.txt/.md)
      │
      ▼
1. Scene Segmentation
   Splits text into scenes using explicit markers (Chapter, ***, ---)
   or paragraph-count grouping as fallback.
      │
      ▼
2. Chunking
   Groups paragraphs into overlapping chunks that fit the model's
   context window. Breaks only at paragraph boundaries.
      │
      ▼
3. LLM Extraction (five sequential passes per chunk)
   Pass 1 — Character extraction → characters.json
   Pass 2 — Dialogue extraction  → dialogues.json
   Pass 3 — Speaker attribution  ─┐
   Pass 4 — Listener attribution  ├─ interaction_log.json
   Pass 5 — Sentiment scoring    ─┘
      │
      ▼
4. Graph Construction
   Aggregates interactions into weighted directed edges.
   Outputs nodes.csv and edges.csv.
```

---

## Resuming and Resetting

The pipeline tracks per-novel, per-stage completion in `pipeline.db`. If a run is interrupted, simply re-run — completed stages are skipped automatically.

**Reset a specific novel** (re-runs all stages):

```bash
python main.py --reset f33b44f6
```

You can use just the first 8 characters of the novel ID (visible in the log).

**Reset a specific stage only:**

```bash
python main.py --reset f33b44f6 --stage dialogues
```

Valid stage names: `characters`, `dialogues`, `interactions`, `all` (default).

**Reset multiple novels:**

```bash
python main.py --reset f33b44f6 2484413e 9101ac3e
```

**Reset everything** (equivalent to deleting `pipeline.db`):

```bash
python main.py --reset all
```

---

## Concurrency and Timeout Tuning

If you see frequent timeout warnings, adjust these settings in `config.py`:

```python
MAX_CONCURRENT_REQUESTS = 2    # parallel Ollama calls — raise gradually (2 → 4 → 6)
REQUEST_TIMEOUT         = 600  # seconds per call — raise for large models or slow servers
MAX_RETRIES             = 2    # retries on network failure before skipping a chunk
```

The pipeline includes an **adaptive throttle**: if more than 40% of recent calls fail, it automatically adds a delay between requests and logs a warning. It recovers gradually as calls succeed. You will see lines like:

```
WARNING  Throttle: 45% failure rate over last 20 calls. Adding 1.0s delay between requests.
INFO     Progress: 100/380 chunks (0.9/s, ~311s remaining, throttle delay: 1.0s)
```

---

## Project Structure

```
novel_sna/
├── data/                      ← put novels here (any subfolder depth)
├── output/                    ← per-novel results written here
├── pipeline.db                ← SQLite resume state (auto-created)
├── requirements.txt
└── src/
    ├── main.py                ← entry point
    ├── config.py              ← all settings, model profiles
    ├── database.py            ← SQLite layer, resume support
    ├── corpus_loader.py       ← recursive .txt/.md discovery
    ├── scene_segmenter.py     ← scene boundary detection
    ├── chunker.py             ← paragraph-aware chunking with overlap
    ├── character_registry.py  ← alias merging and name resolution
    ├── llm_client.py          ← async Ollama client, retry logic
    ├── json_utils.py          ← JSON extraction and truncation repair
    ├── schemas.py             ← Pydantic validation for LLM responses
    ├── models.py              ← internal data transfer objects
    ├── extractors/
    │   ├── character_extractor.py
    │   ├── dialogue_extractor.py
    │   ├── speaker_extractor.py
    │   ├── listener_extractor.py
    │   └── sentiment_extractor.py
    ├── pipelines/
    │   ├── character_pipeline.py
    │   ├── dialogue_pipeline.py
    │   ├── interaction_pipeline.py
    │   └── _queue.py          ← bounded worker pool, adaptive throttle
    ├── prompts/
    │   ├── character_prompt.py
    │   ├── dialogue_prompt.py
    │   ├── speaker_prompt.py
    │   ├── listener_prompt.py
    │   └── sentiment_prompt.py
    ├── graph/
    │   └── graph_builder.py   ← hybrid edge weighting, nodes/edges CSV
    └── exporters/
        ├── json_exporter.py
        └── csv_exporter.py
```

---

## Troubleshooting

**`discovered 0 novels`**
The `data/` folder is empty or contains no `.txt`/`.md` files. Check that your files have the correct extension.

**`Model X returned an empty response`**
The model name in your profile doesn't match exactly what Ollama has. Run `ollama list` on your server and update `config.py` to match.

**Frequent timeouts on large corpora**
Lower `MAX_CONCURRENT_REQUESTS` to 1 or 2. The adaptive throttle will also kick in automatically. If a single call times out repeatedly, the chunk may be too long for the model — reduce `max_chars_per_chunk` in the profile.

**`0 dialogues extracted` for many novels**
The dialogue prompt covers standard quotes, single quotes, em-dash dialogue, and indirect speech. If your corpus uses an unusual convention (e.g. no quotation marks at all), edit `src/prompts/dialogue_prompt.py` to describe the style explicitly.

**`0 edges` in output**
This means interactions were extracted but all speakers or listeners resolved to `UNKNOWN`. Check `interaction_log.json` — if `speaker_confidence` values are consistently low, the model is struggling with attribution. Try switching to a larger model profile.

---

## Output Files Reference

### `nodes.csv`

| Column | Description |
|---|---|
| CharacterID | Stable hash-based ID |
| Name | Canonical character name |
| Aliases | Semicolon-separated list of known aliases |
| Degree | Number of distinct edge connections |

### `edges.csv`

| Column | Description |
|---|---|
| Source | Speaker (canonical name) |
| Target | Listener (canonical name) |
| Interaction_Count | Number of dialogue events |
| Mean_Sentiment | Average sentiment score (−5 to +5) |
| Co_Occurrence | Number of shared scenes |
| Weight | Hybrid weight score |
| Evidence | Up to 3 sample quotes |

### `interaction_log.json`

The source of truth. Every entry represents one attributed speech act with full provenance (novel ID, scene ID, chunk ID, confidence scores). Use this for auditing, debugging, or building alternative graph representations.

---

## Sentiment Scale

| Score | Label |
|---|---|
| +5 | Deeply affectionate / euphoric |
| +4 | Affectionate / loving |
| +3 | Strongly positive |
| +2 | Warm / friendly |
| +1 | Mildly positive |
| 0 | Neutral |
| −1 | Mildly negative |
| −2 | Cold / distant |
| −3 | Strongly negative |
| −4 | Hostile |
| −5 | Very hostile / hateful |
