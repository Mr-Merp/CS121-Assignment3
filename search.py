import math
import os
import pickle
import re
import time
from nltk.stem import PorterStemmer
from collections import defaultdict

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def build_stemmer():
    """Prefer Krovetz stemmer when available; fallback to Porter."""
    try:
        from krovetzstemmer import Stemmer as KrovetzStemmer
        return KrovetzStemmer(), "krovetz"
    except Exception:
        return PorterStemmer(), "porter"


def stem_token(stemmer, token):
    """Stem one token."""
    return stemmer.stem(token)


def load_meta(index_dir="index"):
    """Load metadata for the sharded index."""
    with open(os.path.join(index_dir, "meta.pkl"), "rb") as f:
        return pickle.load(f)


def load_doc_map(index_dir="index"):
    """Load doc_id -> (url, total_words) mapping."""
    with open(os.path.join(index_dir, "doc_map.pkl"), "rb") as f:
        return pickle.load(f)


def parse_query(query, stemmer):
    """Parse query into single terms and quoted phrases."""
    raw_phrases = re.findall(r'"([^"]+)"', query)
    query_without_phrases = re.sub(r'"[^"]+"', " ", query)

    terms = [stem_token(stemmer, token) for token in TOKEN_PATTERN.findall(query_without_phrases.lower())]
    phrases = []
    for phrase in raw_phrases:
        phrase_terms = [stem_token(stemmer, token) for token in TOKEN_PATTERN.findall(phrase.lower())]
        if phrase_terms:
            phrases.append(phrase_terms)

    all_terms = list(terms)
    for phrase_terms in phrases:
        all_terms.extend(phrase_terms)

    return {
        "terms": terms,
        "phrases": phrases,
        "all_terms": all_terms,
    }


def load_term_postings(index_dir, term, shard_cache):
    """Load postings for one term by opening only its shard file."""
    first = term[0] if term else "_"
    shard_key = first if first.isalnum() else "_"
    shard_file = f"index_{shard_key}.pkl"
    shard_path = os.path.join(index_dir, shard_file)

    if shard_file not in shard_cache:
        if os.path.exists(shard_path):
            with open(shard_path, "rb") as f:
                shard_cache[shard_file] = pickle.load(f)
        else:
            shard_cache[shard_file] = {}

    return shard_cache[shard_file].get(term, [])


def parse_posting(posting):
    """Normalize posting tuple shape across index versions."""
    if len(posting) == 3:
        return posting[0], posting[1], posting[2]
    return posting[0], posting[1], []


def load_postings_for_terms(index_dir, terms, shard_cache):
    """Load postings for each unique term."""
    term_postings = {}
    for term in dict.fromkeys(terms):
        term_postings[term] = load_term_postings(index_dir, term, shard_cache)
    return term_postings


def boolean_and_retrieve(term_postings, query_terms):
    """Return doc ids that contain every query term."""
    if not query_terms:
        return set()

    unique_terms = list(dict.fromkeys(query_terms))
    unique_terms.sort(key=lambda term: len(term_postings.get(term, [])))

    common_docs = None
    for term in unique_terms:
        postings = term_postings.get(term, [])
        docs_for_term = {doc_id for doc_id, _, _ in (parse_posting(posting) for posting in postings)}
        if common_docs is None:
            common_docs = docs_for_term
        else:
            common_docs &= docs_for_term

    return common_docs if common_docs is not None else set()


def filter_phrase_candidates(term_postings, phrase_terms, candidate_doc_ids):
    """Filter docs that contain the exact phrase using positional postings."""
    if not phrase_terms:
        return candidate_doc_ids

    term_position_maps = []
    phrase_docs = None
    for term in phrase_terms:
        postings = term_postings.get(term, [])
        doc_to_positions = {}
        for posting in postings:
            doc_id, _, positions = parse_posting(posting)
            doc_to_positions[doc_id] = positions
        term_position_maps.append(doc_to_positions)
        term_docs = set(doc_to_positions.keys())
        phrase_docs = term_docs if phrase_docs is None else phrase_docs & term_docs

    if phrase_docs is None:
        return set()

    filtered_docs = phrase_docs & candidate_doc_ids
    result_docs = set()
    for doc_id in filtered_docs:
        starts = set(term_position_maps[0].get(doc_id, []))
        if not starts:
            continue

        matched = True
        for offset in range(1, len(term_position_maps)):
            positions = term_position_maps[offset].get(doc_id, [])
            if not positions:
                matched = False
                break
            position_set = set(positions)
            starts = {start for start in starts if (start + offset) in position_set}
            if not starts:
                matched = False
                break

        if matched:
            result_docs.add(doc_id)

    return result_docs


def compute_idf(doc_count, doc_freq):
    """Compute IDF for one term."""
    return math.log((doc_count + 1) / (doc_freq + 1)) + 1


def rank_with_tfidf(meta, query_terms, candidate_doc_ids, term_postings, doc_map):
    """Return doc scores using tf-idf with length normalization."""
    scores = defaultdict(float)
    query_tf = defaultdict(int)
    for term in query_terms:
        query_tf[term] += 1

    doc_count = meta.get("doc_count", 0)
    candidate_set = set(candidate_doc_ids)

    for term, qf in query_tf.items():
        postings = term_postings.get(term, [])
        if not postings:
            continue

        doc_freq = len(postings)
        idf = compute_idf(doc_count, doc_freq)
        query_weight = (1 + math.log(qf)) * idf

        for posting in postings:
            doc_id, term_freq, _ = parse_posting(posting)
            if doc_id not in candidate_set:
                continue
            doc_weight = 1 + math.log(term_freq)
            scores[doc_id] += query_weight * doc_weight

    for doc_id in list(scores.keys()):
        total_words = doc_map.get(doc_id, ("", 0))[1]
        if total_words > 0:
            scores[doc_id] /= (1 + math.log(total_words))

    return scores


def main():
    """Prompt for queries and print top 5 URLs."""
    index_dir = "index"
    meta = load_meta(index_dir)
    stemmer, stemmer_name = build_stemmer()
    shard_cache = {}
    doc_map = None

    print(f"Stemmer: {stemmer_name}")
    print("Search ready. Type a query, or 'exit' to quit.")
    while True:
        query = input("query> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Bye.")
            break

        start_time = time.perf_counter()
        parsed = parse_query(query, stemmer)
        all_terms = parsed["all_terms"]
        if not all_terms:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print("No results.\n")
            print(f"Search takes {elapsed_ms:.2f} ms\n")
            continue

        term_postings = load_postings_for_terms(index_dir, all_terms, shard_cache)
        candidates = boolean_and_retrieve(term_postings, all_terms)

        for phrase_terms in parsed["phrases"]:
            candidates = filter_phrase_candidates(term_postings, phrase_terms, candidates)
            if not candidates:
                break

        if not candidates:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print("No results.\n")
            print(f"Search takes {elapsed_ms:.2f} ms\n")
            continue

        if doc_map is None:
            doc_map = load_doc_map(index_dir)

        scores = rank_with_tfidf(meta, all_terms, candidates, term_postings, doc_map)
        ranked_ids = sorted(candidates, key=lambda d: scores.get(d, 0.0), reverse=True)

        result_lines = []
        for doc_id in ranked_ids:
            if doc_id in doc_map:
                url = doc_map[doc_id][0]
                result_lines.append(url)
            if len(result_lines) == 5:
                break

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        print(f"Top {len(result_lines)} results:")
        for i, url in enumerate(result_lines, start=1):
            print(f"{i}. {url}")
        print("")
        print(f"Search takes {elapsed_ms:.2f} ms\n")


if __name__ == "__main__":
    main()
