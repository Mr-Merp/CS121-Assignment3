import json
import os
import re
import struct
from collections import defaultdict
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
import msgpack
import warnings
from tqdm import tqdm
from krovetzstemmer import Stemmer as KrovetzStemmer

# Using msgpack, its 3 unit64int numbers basically which equates to 24 bits
_HEADER_FMT = ">QQQ"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

class Posting:
    """Represents a single posting in the inverted index"""

    def __init__(self, doc_id, term_frequency):
        self.doc_id = doc_id
        self.term_frequency = term_frequency

    def __repr__(self):
        return f"Posting(doc_id={self.doc_id}, tf={self.term_frequency})"


class InvertedIndex:
    """Builds and manages the inverted index"""

    def __init__(self):
        self.index = defaultdict(list)
        self.doc_count = 0
        self.doc_id_to_url_total_words = {}
        self.stemmer, self.stemmer_name = self.build_stemmer()
        self.unique_token_count = 0
        self.token_pattern = re.compile(r"[a-z0-9]+")

    def build_stemmer(self):
        return KrovetzStemmer(), "krovetz"


    def stem_token(self, token):
        """Stem one token using the configured stemmer."""
        return self.stemmer.stem(token)

    def extract_text_with_weights(self, html_content):
        """Extract text from HTML with importance weights in a single pass"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            for script in soup(['script', 'style', 'meta', 'link']):
                script.decompose()

            important_tags = ['h1', 'h2', 'h3', 'b', 'strong', 'title']
            important_text = []
            for tag in soup.find_all(important_tags):
                important_text.append(tag.get_text())

            all_text = soup.get_text(separator=' ')

            return all_text, ' '.join(important_text)
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return "", ""

    def compute_stemmed_term_frequency(self, text):
        """Tokenize, stem, and count terms directly from raw text."""
        term_freq = defaultdict(int)
        stem = self.stem_token
        for token in self.token_pattern.findall(text.lower()):
            term_freq[stem(token)] += 1
        return term_freq

    def compute_positions_and_term_frequency(self, text):
        """Tokenize, stem, count frequency, and store token positions."""
        term_freq = defaultdict(int)
        term_positions = defaultdict(list)
        stem = self.stem_token
        position = 0
        for token in self.token_pattern.findall(text.lower()):
            stemmed = stem(token)
            term_freq[stemmed] += 1
            term_positions[stemmed].append(position)
            position += 1
        return term_freq, term_positions

    def add_document(self, doc_id, url, html_content):
        """Process a single document and add it to the index"""
        all_text, important_text = self.extract_text_with_weights(html_content)

        term_freq, term_positions = self.compute_positions_and_term_frequency(all_text)
        important_term_freq = self.compute_stemmed_term_frequency(important_text)
        for token, freq in important_term_freq.items():
            term_freq[token] += freq

        total = 0
        for token, freq in term_freq.items():
            positions = term_positions.get(token, [])
            self.index[token].append((doc_id, freq, positions))
            total += freq

        self.doc_id_to_url_total_words[doc_id] = (url, total)
        self.doc_count += 1

    def build_from_directory(self, directory_path):
        """
        Build index from all JSON files in the directory.
        """
        print(f"Building index from: {directory_path}")

        json_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))

        total_files = len(json_files)
        print(f"Found {total_files} JSON files to process")

        for doc_id, json_file in tqdm(enumerate(json_files), total=len(json_files)):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:

                    data = json.load(f)
                    url = data.get('url', '')
                    content = data.get('content', '')

                    self.add_document(doc_id, url, content)

            except Exception as e:
                print(f"Error processing {json_file}: {e}")

        self.unique_token_count = len(self.index)
        print(f"Index building complete!")
        print(f"Total documents indexed: {self.doc_count}")
        print(f"Total unique tokens: {self.unique_token_count}")

    def get_partial_index_name(self, term):
        """Map a term to its partial index name based on first character.

        Splits tokens into 4 buckets, a-i, j-r, s-z, and numbers
        """
        if not term or not term[0].isalpha():
            return "numbers"
        c = term[0]
        if 'a' <= c <= 'i':
            return "a_i"
        elif 'j' <= c <= 'r':
            return "j_r"
        else:
            return "s_z"

    def clear_old_index_files(self, index_dir):
        """Remove old generated index files before writing new ones."""
        if not os.path.exists(index_dir):
            return

        for file in os.listdir(index_dir):
            file_path = os.path.join(index_dir, file)
            if file.endswith(".msgpack") or (file.startswith("index_") and file.endswith(".pkl")):
                os.remove(file_path)
            elif file in {"doc_map.pkl", "meta.pkl"}:
                os.remove(file_path)

    def save_sharded_index(self, index_dir):
        """Write the entire index into a single index/index.msgpack file.
        On load, read the 24-byte header first, then seek to each section as needed.
        To fetch one term's postings, look up its offset in the index-of-index,
        seek to that position, and unpack exactly one msgpack entry.
        """
        os.makedirs(index_dir, exist_ok=True)
        self.clear_old_index_files(index_dir)

        # bucket grouping
        groups = defaultdict(dict)
        for term, postings in self.index.items():
            groups[self.get_partial_index_name(term)][term] = postings

        index_of_index = {}
        out_path = os.path.join(index_dir, "index.msgpack")

        with open(out_path, "wb") as f:
            # get space for the 24 bytes talked about above
            f.write(b"\x00" * _HEADER_SIZE)

            # postings for each bucket
            for bucket in sorted(groups):
                for term in sorted(groups[bucket]):
                    postings = groups[bucket][term]
                    postings_list = [
                        [doc_id, tf, positions]
                        for doc_id, tf, positions in postings
                    ]
                    index_of_index[term] = f.tell()
                    f.write(msgpack.packb([term, postings_list], use_bin_type=True))

            # byte offset
            ioi_offset = f.tell()
            f.write(msgpack.packb(index_of_index, use_bin_type=True))

            # map doc_id to url and total_words
            docmap_offset = f.tell()
            f.write(msgpack.packb(self.doc_id_to_url_total_words, use_bin_type=True))

            # store metadata, so if we need to change things, in the future, its easy
            # no hard coding needed
            meta_offset = f.tell()
            metadata = {
                "doc_count": self.doc_count,
                "unique_tokens": len(self.index),
                "buckets": sorted(groups.keys()),
            }
            f.write(msgpack.packb(metadata, use_bin_type=True))

            # go back to real header location and write
            f.seek(0)
            f.write(struct.pack(_HEADER_FMT, ioi_offset, docmap_offset, meta_offset))

        print(f"Index saved to: {out_path}")

    def get_index_size(self, file_path):
        """Get the size of the index file in KB"""
        size_bytes = os.path.getsize(file_path)
        size_kb = size_bytes / 1024
        return size_kb

    def get_directory_size_kb(self, directory_path):
        """Get total size of files in a directory in KB."""
        total_bytes = 0
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_bytes += os.path.getsize(file_path)
        return total_bytes / 1024

    def print_analytics(self, index_path):
        """Print analytics about the index"""
        index_file = os.path.join(index_path, "index.msgpack") if os.path.isdir(index_path) else index_path
        size_kb = self.get_index_size(index_file)

        unique_tokens = self.unique_token_count if self.unique_token_count else len(self.index)

        print("\n" + "=" * 50)
        print("INVERTED INDEX ANALYTICS")
        print("=" * 50)
        print(f"Number of indexed documents: {self.doc_count}")
        print(f"Number of unique tokens: {unique_tokens}")
        print(f"Total size of index on disk: {size_kb:.2f} KB")
        print("=" * 50)


def main():
    """Main function to build the inverted index"""
    dev_folder = "DEV"
    output_dir = "index"

    index = InvertedIndex()
    print(f"Using stemmer: {index.stemmer_name}")

    index.build_from_directory(dev_folder)

    index.save_sharded_index(output_dir)

    index.print_analytics(output_dir)


if __name__ == "__main__":
    main()
