# Inverted Index Search Engine - CS121

This project builds a positional inverted index from the ICS crawl dataset and supports boolean AND retrieval with TF-IDF ranking.

Group member(s):
Isaac Zhang
Owen Yang
Kent Ziti
David Deng

Student Ids: 63919428, 92644763, 40689120, 38943002

## Components

### 1. Indexer (`indexer.py`)
- Parses HTML content and boosts important tags (`h1`, `h2`, `h3`, `b`, `strong`, `title`).
- Uses Krovetz stemming.
- Builds positional postings in format `(doc_id, weighted_tf, positions)`.
- Performs partial index offloading during indexing and merges partial indexes into a final on-disk index.
- Saves final index files into `index/`:
  - `index/index.msgpack` (postings + index-of-index + doc map + metadata)
  - `index/qgram2.msgpack` (2-gram vocabulary index for typo correction)

### 2. Search (`search.py`)
- Loads only needed term postings from `index/index.msgpack` using byte offsets.
- Supports boolean AND retrieval.
- Supports TF-IDF ranking.
- Supports quoted phrase queries using positional postings.
- Supports 2-gram typo correction for missing query terms.
- Prints query latency in milliseconds.

### 3. Web Interface (`search_web.py`)
- Local Flask web interface for the search engine.
- Reuses the same search pipeline as `search.py`.
- Opens a browser automatically on launch.

### 4. Report Generator (`generate_report.py`)
- Reads `index/index.msgpack`.
- Reports index analytics:
  - indexed document count
  - unique token count
  - index bucket count
  - total postings
  - index size on disk
  - top 10 most frequent tokens by document frequency

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python indexer.py
python search.py
python search_web.py
python generate_report.py
```
