import math
import os
import struct
import msgpack
import re
import time
from krovetzstemmer import Stemmer as KrovetzStemmer
from collections import defaultdict

_HEADER_FMT = ">QQQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 24

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
PHRASE_PATTERN = re.compile(r'"([^"]+)"')
QUOTED_PATTERN = re.compile(r'"[^"]+"')


def build_stemmer():
    return KrovetzStemmer(), "krovetz"



def stem_token(stemmer, token):
    """Stem one token."""
    return stemmer.stem(token)


def load_all_metadata(index_dir="index"):
    """Read the 24-byte header then load meta, doc_map, and index-of-index in one pass."""
    path = os.path.join(index_dir, "index.msgpack")
    with open(path, "rb") as f:
        ioi_offset, docmap_offset, meta_offset = struct.unpack(_HEADER_FMT, f.read(_HEADER_SIZE))

        f.seek(meta_offset)
        meta = next(msgpack.Unpacker(f, raw=False))

        # idk why but strict_map_key needs to be false
        f.seek(docmap_offset)
        doc_map = next(msgpack.Unpacker(f, raw=False, strict_map_key=False))

        f.seek(ioi_offset)
        index_of_index = next(msgpack.Unpacker(f, raw=False))

    return meta, doc_map, index_of_index


def parse_query(query, stemmer):
    """Parse query into single terms and quoted phrases."""
    raw_phrases = PHRASE_PATTERN.findall(query)
    query_without_phrases = QUOTED_PATTERN.sub(" ", query)

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


def load_term_postings(index_file, term, term_cache, index_of_index):
    """Seek to a term's byte offset in index.msgpack and unpack only that entry.

    index_of_index maps term -> absolute byte offset written by the indexer.
    Results are cached in term_cache so repeated queries avoid redundant I/O.
    """
    if term in term_cache:
        return term_cache[term]

    offset = index_of_index.get(term)
    if offset is None:
        term_cache[term] = []
        return []

    with open(index_file, "rb") as f:
        f.seek(offset)
        _, postings_list = next(msgpack.Unpacker(f, raw=False))

    term_cache[term] = postings_list
    return postings_list


def parse_posting(posting):
    """Normalize posting tuple shape across index versions."""
    if len(posting) == 3:
        return posting[0], posting[1], posting[2]
    return posting[0], posting[1], []


def load_postings_for_terms(index_file, terms, term_cache, index_of_index):
    """Load postings for each unique term."""
    term_postings = {}
    for term in dict.fromkeys(terms):
        term_postings[term] = load_term_postings(index_file, term, term_cache, index_of_index)
    return term_postings


def boolean_and_retrieve(term_postings, query_terms):
    """Return doc ids that contain every query term."""
    if not query_terms:
        return set()

    unique_terms = list(dict.fromkeys(query_terms))
    unique_terms.sort(key=lambda term: len(term_postings.get(term, [])))
    if not unique_terms:
        return set()

    first_postings = term_postings.get(unique_terms[0], [])
    common_docs = {posting[0] for posting in first_postings}
    if not common_docs:
        return set()

    for term in unique_terms[1:]:
        postings = term_postings.get(term, [])
        next_common_docs = set()
        for posting in postings:
            doc_id = posting[0]
            if doc_id in common_docs:
                next_common_docs.add(doc_id)
        common_docs = next_common_docs
        if not common_docs:
            break

    return common_docs


def has_phrase_positions(position_lists):
    """Check whether positional lists contain at least one exact phrase match."""
    if not position_lists:
        return False
    current_starts = position_lists[0]
    if not current_starts:
        return False

    for offset in range(1, len(position_lists)):
        next_positions = position_lists[offset]
        if not next_positions:
            return False

        i = 0
        j = 0
        matched_starts = []
        while i < len(current_starts) and j < len(next_positions):
            target = current_starts[i] + offset
            pos = next_positions[j]
            if pos == target:
                matched_starts.append(current_starts[i])
                i += 1
                j += 1
            elif pos < target:
                j += 1
            else:
                i += 1

        if not matched_starts:
            return False
        current_starts = matched_starts

    return True


def filter_phrase_candidates(term_postings, phrase_terms, candidate_doc_ids):
    """Filter docs that contain the exact phrase using positional postings."""
    if not phrase_terms:
        return candidate_doc_ids

    filtered_docs = set(candidate_doc_ids)
    term_position_maps = {}
    for term in phrase_terms:
        postings = term_postings.get(term, [])
        doc_to_positions = {}
        for posting in postings:
            doc_id = posting[0]
            positions = posting[2] if len(posting) > 2 else []
            if doc_id in filtered_docs:
                doc_to_positions[doc_id] = positions
        term_position_maps[term] = doc_to_positions
        filtered_docs.intersection_update(doc_to_positions.keys())
        if not filtered_docs:
            return set()

    result_docs = set()
    for doc_id in filtered_docs:
        position_lists = []
        for term in phrase_terms:
            positions = term_position_maps[term].get(doc_id)
            if not positions:
                position_lists = []
                break
            position_lists.append(positions)

        if position_lists and has_phrase_positions(position_lists):
            result_docs.add(doc_id)

    return result_docs


def compute_idf(doc_count, doc_freq):
    """Compute IDF for one term."""
    return math.log((doc_count + 1) / (doc_freq + 1)) + 1


def binary_search_term_freq(postings, target_doc_id):
    """Binary-search one doc id in a sorted postings list and return tf."""
    left = 0
    right = len(postings) - 1

    while left <= right:
        mid = (left + right) // 2
        mid_post = postings[mid]
        mid_doc_id = mid_post[0]
        if mid_doc_id == target_doc_id:
            return mid_post[1]
        if mid_doc_id < target_doc_id:
            left = mid + 1
        else:
            right = mid - 1

    return None


def rank_with_tfidf(meta, query_terms, candidate_doc_ids, term_postings, doc_map, idf_cache, doc_norm_cache):
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

        if term in idf_cache:
            idf = idf_cache[term]
        else:
            idf = compute_idf(doc_count, len(postings))
            idf_cache[term] = idf

        query_weight = (1 + math.log(qf)) * idf

        # For very common terms, binary search on candidate docs is faster.
        if len(candidate_set) * math.log2(len(postings) + 1) < len(postings):
            for doc_id in candidate_set:
                term_freq = binary_search_term_freq(postings, doc_id)
                if term_freq is None:
                    continue
                doc_weight = 1 + math.log(term_freq)
                scores[doc_id] += query_weight * doc_weight
        else:
            for posting in postings:
                doc_id = posting[0]
                term_freq = posting[1]
                if doc_id not in candidate_set:
                    continue
                doc_weight = 1 + math.log(term_freq)
                scores[doc_id] += query_weight * doc_weight

    for doc_id in list(scores.keys()):
        if doc_id in doc_norm_cache:
            scores[doc_id] *= doc_norm_cache[doc_id]
            continue

        total_words = doc_map.get(doc_id, ("", 0))[1]
        if total_words > 0:
            norm = 1 / (1 + math.log(total_words))
            doc_norm_cache[doc_id] = norm
            scores[doc_id] *= norm

    return scores


def main():
    """Prompt for queries and print top 5 URLs."""
    index_dir = "index"
    index_file = os.path.join(index_dir, "index.msgpack")
    meta, doc_map, index_of_index = load_all_metadata(index_dir)
    stemmer, _ = build_stemmer()
    term_cache = {}
    idf_cache = {}
    doc_norm_cache = {}

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

        term_postings = load_postings_for_terms(index_file, all_terms, term_cache, index_of_index)
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

        scores = rank_with_tfidf(meta, all_terms, candidates, term_postings, doc_map, idf_cache, doc_norm_cache)
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
