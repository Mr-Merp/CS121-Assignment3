import io
import os
import struct
import msgpack
import heapq

_HEADER_FMT = ">QQQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 24


def generate_report(index_dir="index", output_file="report.txt"):
    """Generate a detailed analytics report from a single index.msgpack file."""
    index_file = os.path.join(index_dir, "index.msgpack")

    with open(index_file, "rb") as f:
        ioi_offset, _, meta_offset = struct.unpack(_HEADER_FMT, f.read(_HEADER_SIZE))

        f.seek(meta_offset)
        meta = next(msgpack.Unpacker(f, raw=False))

        # Postings section runs from byte _HEADER_SIZE to ioi_offset.
        # Read it all at once so we can iterate without holding the file open.
        f.seek(_HEADER_SIZE)
        postings_bytes = f.read(ioi_offset - _HEADER_SIZE)

    doc_count = meta["doc_count"]
    unique_tokens = meta["unique_tokens"]
    size_kb = os.path.getsize(index_file) / 1024
    bucket_count = len(meta.get("buckets", []))

    total_postings = 0
    top_10_heap = []

    for term, postings_list in msgpack.Unpacker(io.BytesIO(postings_bytes), raw=False):
        posting_count = len(postings_list)
        total_postings += posting_count

        if len(top_10_heap) < 10:
            heapq.heappush(top_10_heap, (posting_count, term))
        elif posting_count > top_10_heap[0][0]:
            heapq.heapreplace(top_10_heap, (posting_count, term))

    avg_postings_per_token = total_postings / unique_tokens if unique_tokens > 0 else 0
    top_10_tokens = [(token, count) for count, token in sorted(top_10_heap, reverse=True)]

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("INVERTED INDEX ANALYTICS REPORT")
    report_lines.append("=" * 70)

    report_lines.append("+" + "-" * 68 + "+")
    report_lines.append("| Metric                              | Value                        |")
    report_lines.append("+" + "-" * 68 + "+")
    report_lines.append(f"| Number of indexed documents         | {doc_count:>28,} |")
    report_lines.append(f"| Number of unique tokens             | {unique_tokens:>28,} |")
    report_lines.append(f"| Number of index buckets             | {bucket_count:>28,} |")
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
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Report generated: {output_file}")
    print("\nReport Preview:")
    print(report_text)

    return report_text


if __name__ == "__main__":
    index_dir = "index"

    if not os.path.exists(os.path.join(index_dir, "index.msgpack")):
        print(f"Error: '{index_dir}/index.msgpack' not found.")
        print("Please run indexer.py first to build the index.")
    else:
        generate_report(index_dir=index_dir)
