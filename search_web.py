import argparse
import os
import threading
import time
import webbrowser
from urllib.parse import urlparse

from flask import Flask, jsonify, render_template, request

import search as search_core


class SearchRuntime:
    def __init__(self, index_dir="index"):
        self.index_dir = index_dir
        self.index_file = os.path.join(index_dir, "index.msgpack")
        self.meta, self.doc_map, self.index_of_index = search_core.load_all_metadata(index_dir)
        self.qgram_index = search_core.load_qgram_index(index_dir)
        self.stemmer, _ = search_core.build_stemmer()
        self.term_cache = {}
        self.idf_cache = {}
        self.doc_norm_cache = {}

    def query(self, raw_query):
        start_time = time.perf_counter()
        parsed = search_core.parse_query(raw_query, self.stemmer)
        parsed = search_core.apply_qgram_corrections(parsed, self.qgram_index, self.index_of_index)
        all_terms = parsed["all_terms"]

        if not all_terms:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return {
                "query": raw_query,
                "results": [],
                "elapsed_ms": round(elapsed_ms, 2),
                "count": 0,
            }

        term_postings = search_core.load_postings_for_terms(
            self.index_file, all_terms, self.term_cache, self.index_of_index
        )
        candidates = search_core.boolean_and_retrieve(term_postings, all_terms)

        for phrase_terms in parsed["phrases"]:
            candidates = search_core.filter_phrase_candidates(term_postings, phrase_terms, candidates)
            if not candidates:
                break

        if not candidates:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return {
                "query": raw_query,
                "results": [],
                "elapsed_ms": round(elapsed_ms, 2),
                "count": 0,
            }

        scores = search_core.rank_with_tfidf(
            self.meta,
            all_terms,
            candidates,
            term_postings,
            self.doc_map,
            self.idf_cache,
            self.doc_norm_cache,
        )
        ranked_ids = sorted(candidates, key=lambda d: scores.get(d, 0.0), reverse=True)

        results = []
        for doc_id in ranked_ids[:5]:
            url = self.doc_map.get(doc_id, ("", 0))[0]
            if not url:
                continue
            parsed_url = urlparse(url)
            host = parsed_url.netloc
            title = url
            results.append(
                {
                    "doc_id": int(doc_id),
                    "url": url,
                    "title": title,
                    "host": host,
                    "score": float(scores.get(doc_id, 0.0)),
                }
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return {
            "query": raw_query,
            "results": results,
            "elapsed_ms": round(elapsed_ms, 2),
            "count": len(results),
        }


def create_app(index_dir="index"):
    app = Flask(__name__)
    runtime = SearchRuntime(index_dir=index_dir)

    @app.route("/")
    def home():
        return render_template("home.html")

    @app.route("/search")
    def search_page():
        query = request.args.get("q", "").strip()
        payload = runtime.query(query) if query else {"query": "", "results": [], "elapsed_ms": 0.0, "count": 0}
        return render_template("results.html", data=payload)

    @app.route("/api/search")
    def api_search():
        query = request.args.get("q", "").strip()
        payload = runtime.query(query)
        return jsonify(payload)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--index-dir", default="index")
    args = parser.parse_args()

    app = create_app(index_dir=args.index_dir)
    url = f"http://{args.host}:{args.port}/"

    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
