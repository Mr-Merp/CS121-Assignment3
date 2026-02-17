import pickle
import os
from indexer import Posting

def load_index(index_file):
    """Load the inverted index from disk"""
    with open(index_file, 'rb') as f:
        data = pickle.load(f)
    return data


def generate_report(index_file, output_file="report.txt"):
    """Generate a detailed analytics report"""

    index_data = load_index(index_file)
    index = index_data['index']
    doc_count = index_data['doc_count']

    size_bytes = os.path.getsize(index_file)
    size_kb = size_bytes / 1024

    unique_tokens = len(index)

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
    report_lines.append("+" + "-" * 68 + "+")
    report_lines.append("")

    report_lines.append("=" * 70)
    report_lines.append("INDEX STRUCTURE")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("The inverted index is structured as follows:")
    report_lines.append("")
    report_lines.append("  Token -> List of Postings")
    report_lines.append("")
    report_lines.append("Each Posting contains:")
    report_lines.append("  - Document ID: Unique identifier for the document")
    report_lines.append("  - Term Frequency (TF): Number of times the token appears in the document")
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
    index_file = "inverted_index.pkl"

    if not os.path.exists(index_file):
        print(f"Error: Index file '{index_file}' not found.")
        print("Please run indexer.py first to build the index.")
    else:
        generate_report(index_file)
