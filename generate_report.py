import os
import pickle


def load_meta(index_dir):
    """Load index metadata from disk."""
    with open(os.path.join(index_dir, "meta.pkl"), "rb") as f:
        return pickle.load(f)


def load_shard(index_dir, shard_file):
    """Load one shard file."""
    with open(os.path.join(index_dir, shard_file), "rb") as f:
        return pickle.load(f)


def generate_report(index_dir="index", output_file="report.txt"):
    """Generate a detailed analytics report"""

    meta = load_meta(index_dir)
    doc_count = meta["doc_count"]
    shard_files = meta.get("shards", [])

    size_bytes = 0
    for file in os.listdir(index_dir):
        file_path = os.path.join(index_dir, file)
        if os.path.isfile(file_path):
            size_bytes += os.path.getsize(file_path)
    size_kb = size_bytes / 1024

    unique_tokens = meta.get("unique_tokens", 0)

    total_postings = 0
    token_doc_counts = []
    for shard_file in shard_files:
        shard_data = load_shard(index_dir, shard_file)
        for token, postings in shard_data.items():
            posting_count = len(postings)
            total_postings += posting_count
            token_doc_counts.append((token, posting_count))

    avg_postings_per_token = total_postings / unique_tokens if unique_tokens > 0 else 0

    # Find top 10 most common tokens (by number of documents they appear in)
    token_doc_counts.sort(key=lambda x: x[1], reverse=True)
    top_10_tokens = token_doc_counts[:10]

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("INVERTED INDEX ANALYTICS REPORT")
    report_lines.append("=" * 70)

    report_lines.append("+" + "-" * 68 + "+")
    report_lines.append("| Metric                              | Value                        |")
    report_lines.append("+" + "-" * 68 + "+")
    report_lines.append(f"| Number of indexed documents         | {doc_count:>28,} |")
    report_lines.append(f"| Number of unique tokens             | {unique_tokens:>28,} |")
    report_lines.append(f"| Total size of index on disk (KB)    | {size_kb:>28,.2f} |")
    report_lines.append(f"| Total postings                      | {total_postings:>28,} |")
    report_lines.append(f"| Average postings per token          | {avg_postings_per_token:>28,.2f} |")
    report_lines.append("+" + "-" * 68 + "+")
    report_lines.append("")

    report_lines.append("=" * 70)
    report_lines.append("TOP 10 MOST COMMON TOKENS")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("Token                              | Documents")
    report_lines.append("-" * 70)
    for token, doc_count_token in top_10_tokens:
        report_lines.append(f"{token:<35} | {doc_count_token:>10,}")
    report_lines.append("")

    report_lines.append("=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"Report generated: {output_file}")
    print("\nReport Preview:")
    print(report_text)

    return report_text


if __name__ == "__main__":
    index_dir = "index"

    if not os.path.exists(index_dir):
        print(f"Error: Index directory '{index_dir}' not found.")
        print("Please run indexer.py first to build the index.")
    else:
        generate_report(index_dir=index_dir)
