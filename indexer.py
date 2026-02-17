import json
import os
import re
from collections import defaultdict
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, MarkupResemblesLocatorWarning
import pickle
import warnings
from tqdm import tqdm

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

    def extract_text_from_html(self, html_content):
        """Extract text from HTML using BeautifulSoup"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            for script in soup(['script', 'style', 'meta', 'link']):
                script.decompose()

            text = soup.get_text(separator=' ')
            return text
        except Exception as e:
            print(f"Error parsing HTML: {e}")
            return ""

    def tokenize(self, text):
        """
        Tokenize text by splitting on any character that's not alphanumeric
        Returns a list of tokens
        """
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return tokens

    def compute_term_frequency(self, tokens):
        """Compute term frequency for each token in a document"""
        term_freq = defaultdict(int)
        for token in tokens:
            term_freq[token] += 1
        return term_freq

    def add_document(self, doc_id, url, html_content):
        """Process a single document and add it to the index"""
        text = self.extract_text_from_html(html_content)

        tokens = self.tokenize(text)

        term_freq = self.compute_term_frequency(tokens)

        total = 0
        for token, freq in term_freq.items():
            posting = Posting(doc_id, freq)
            self.index[token].append(posting)
            total += freq

        self.doc_id_to_url_total_words[doc_id] = (url, total)
        self.doc_count += 1

    def build_from_directory(self, directory_path):
        """
        Build index from all JSON files in the directory
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

        print(f"Index building complete!")
        print(f"Total documents indexed: {self.doc_count}")
        print(f"Total unique tokens: {len(self.index)}")

    def save_to_disk(self, output_path):
        """Save the inverted index to disk using pickle"""
        index_data = {
            'index': dict(self.index),
            'doc_count': self.doc_count,
            'doc_id_to_url_total_words': self.doc_id_to_url_total_words
        }

        with open(output_path, 'wb') as f:
            pickle.dump(index_data, f)

        print(f"Index saved to: {output_path}")

    def get_index_size(self, file_path):
        """Get the size of the index file in KB"""
        size_bytes = os.path.getsize(file_path)
        size_kb = size_bytes / 1024
        return size_kb

    def print_analytics(self, index_file_path):
        """Print analytics about the index"""
        size_kb = self.get_index_size(index_file_path)

        print("\n" + "=" * 50)
        print("INVERTED INDEX ANALYTICS")
        print("=" * 50)
        print(f"Number of indexed documents: {self.doc_count}")
        print(f"Number of unique tokens: {len(self.index)}")
        print(f"Total size of index on disk: {size_kb:.2f} KB")
        print("=" * 50)


def main():
    """Main function to build the inverted index"""
    dev_folder = "DEV"
    output_file = "inverted_index.pkl"

    index = InvertedIndex()

    index.build_from_directory(dev_folder)

    index.save_to_disk(output_file)

    index.print_analytics(output_file)


if __name__ == "__main__":
    main()
