# Inverted Index Builder - CS121 HW3

This project builds an inverted index from HTML web pages stored in JSON format.

Group member(s):
Isaac Zhang
Owen Yang
Kent Ziti
David Deng

Student Ids: 63919428, 92644763, 40689120, 38943002

## Components

### 1. Indexer (`indexer.py`)
- **InvertedIndex Class**: Builds and manages the inverted index
  - Parses HTML and extracts normal/important text
  - Uses stemming during indexing (Krovetz if installed, otherwise Porter)
  - Stores postings as `(doc_id, weighted_tf, positions)`
  - `add_document()`: Processes a single document
  - `build_from_directory()`: Processes all JSON files in the DEV folder
  - `save_sharded_index()`: Saves sharded index files into `index/`
  - Also saves `index/meta.pkl` and `index/doc_map.pkl`

### 2. Search (`search.py`)
- Loads only needed shard files on demand
- Supports boolean AND retrieval
- Supports TF-IDF ranking
- Supports quoted phrase matching using positional postings
- Prints query latency in milliseconds

### 3. Report Generator (`generate_report.py`)
- Loads index metadata and shard files from `index/`
- Generates comprehensive analytics including:
  - Number of indexed documents
  - Number of unique tokens
  - Total index size in KB
  - Additional statistics (total postings, average postings per token)
  - Top 10 most common tokens

## Usage

### Step 1: Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
