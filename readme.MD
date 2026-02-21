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
- Uses stemming (Krovetz if available, otherwise Porter).
- Builds positional postings in format `(doc_id, weighted_tf, positions)`.
- Saves sharded index files into `index/` using 2-character shard prefixes:
  - `index/index_<2chars>.pkl`
  - `index/meta.pkl`
  - `index/doc_map.pkl`

### 2. Search (`search.py`)
- Loads only needed shards for query terms.
- Supports boolean AND retrieval.
- Supports TF-IDF ranking.
- Supports quoted phrase queries using positional postings.
- Prints query latency in milliseconds.

### 3. Report Generator (`generate_report.py`)
- Reads `index/meta.pkl` and shard files.
- Reports index analytics:
  - indexed document count
  - unique token count
  - shard file count
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
python generate_report.py
```
