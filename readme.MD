# Inverted Index Builder - CS121 HW3

This project builds an inverted index from HTML web pages stored in JSON format.

Group member(s):
Isaac Zhang
Owen Yang
Kent Ziti
David Deng

Student Ids: 63919428, 92644763, 40689120, 38943002

## Project Structure

```
HW3/
├── DEV/                    # Directory containing JSON files with HTML content
├── indexer.py              # Main script to build the inverted index
├── generate_report.py      # Script to generate analytics report
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── inverted_index.pkl     # Generated inverted index (after running)
└── report.txt             # Generated analytics report (after running)
```

## Components

### 1. Indexer (`indexer.py`)
- **Posting Class**: Represents a token occurrence in a document
  - `doc_id`: Unique document identifier (filename without extension)
  - `term_frequency`: Number of times the token appears in the document

- **InvertedIndex Class**: Builds and manages the inverted index
  - `extract_text_from_html()`: Uses BeautifulSoup to parse HTML and extract text
  - `tokenize()`: Splits text on non-alphanumeric characters (keeps only A-Z, a-z, 0-9)
  - `compute_term_frequency()`: Calculates term frequency for each token
  - `add_document()`: Processes a single document
  - `build_from_directory()`: Processes all JSON files in the DEV folder
  - `save_to_disk()`: Saves the index using pickle

### 2. Report Generator (`generate_report.py`)
- Loads the inverted index from disk
- Generates comprehensive analytics including:
  - Number of indexed documents
  - Number of unique tokens
  - Total index size in KB
  - Additional statistics (total postings, average postings per token)
  - Top 10 most common tokens

## Usage

### Step 1: Create and activate a virtual environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies used in this project:
- `beautifulsoup4`
- `tqdm`
- `nltk`

### Step 3: Build the Inverted Index

```bash
python indexer.py
```

This will:
- Process all JSON files in the DEV folder
- Extract text from HTML content
- Tokenize the text (split on non-alphanumeric characters)
- Build the inverted index with term frequencies
- Save the index to `inverted_index.pkl`
- Display analytics

Expected output:
```
Building index from: DEV
Found 55393 JSON files to process
Processed 1000/55393 documents...
Processed 2000/55393 documents...
...
Index building complete!
Total documents indexed: 55393
Total unique tokens: [number]
Index saved to: inverted_index.pkl

==================================================
INVERTED INDEX ANALYTICS
==================================================
Number of indexed documents: 55393
Number of unique tokens: [number]
Total size of index on disk: [size] KB
==================================================
```

### Step 4: Generate Analytics Report

```bash
python generate_report.py
```

This will:
- Load the inverted index from disk
- Generate a detailed analytics report
- Save to `report.txt`
- Display the report in console

## Index Structure

The inverted index is a dictionary where:
- **Key**: Token (normalized to lowercase)
- **Value**: List of Posting objects

Example:
```python
{
    'uci': [
        Posting(doc_id='4b479247...', term_frequency=15),
        Posting(doc_id='8ef6d99d...', term_frequency=8),
        ...
    ],
    'computer': [
        Posting(doc_id='906c24a2...', term_frequency=5),
        ...
    ]
}
```

## Tokenization Rules

The tokenizer splits text based on:
- **Keep**: Letters (A-Z, a-z) and digits (0-9)
- **Remove**: All other characters (spaces, punctuation, special characters)
- **Normalization**: Convert to lowercase

Examples:
- `"Hello, World!"` → `['hello', 'world']`
- `"CS121-2024"` → `['cs121', '2024']`
- `"user@example.com"` → `['user', 'example', 'com']`
