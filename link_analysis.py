import os
import re
from collections import defaultdict
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import msgpack
import math


class LinkAnalyzer:
    """Extract and analyze hyperlinks from documents"""

    def __init__(self):
        self.graph = defaultdict(set)  # adjacency list: doc_id -> set of doc_ids
        self.url_to_doc_id = {}  # url -> doc_id mapping
        self.doc_id_to_url = {}  # doc_id -> url mapping

    def extract_links_from_html(self, html_content, base_url, all_urls):
        """Extract absolute URLs from HTML that point to documents in corpus"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = set()
            
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href'].strip()
                if not href:
                    continue
                
                # Convert relative URLs to absolute
                try:
                    absolute_url = urljoin(base_url, href)
                    # Normalize URL (remove fragments)
                    absolute_url = absolute_url.split('#')[0]
                    
                    # Check if this URL is in our corpus
                    if absolute_url in all_urls:
                        links.add(absolute_url)
                except:
                    pass
            
            return links
        except Exception as e:
            print(f"Error extracting links from {base_url}: {e}")
            return set()

    def build_graph(self, doc_id_to_url, doc_id_to_content):
        """Build the link graph from documents"""
        # First, create url -> doc_id mapping
        self.url_to_doc_id = {url: doc_id for doc_id, url in doc_id_to_url.items()}
        self.doc_id_to_url = doc_id_to_url
        all_urls = set(doc_id_to_url.values())
        
        # Extract links for each document
        for doc_id, url in doc_id_to_url.items():
            content = doc_id_to_content.get(doc_id, "")
            links = self.extract_links_from_html(content, url, all_urls)
            
            for target_url in links:
                target_doc_id = self.url_to_doc_id.get(target_url)
                if target_doc_id is not None:
                    self.graph[doc_id].add(target_doc_id)

    def pagerank(self, iterations=20, damping_factor=0.85, tolerance=1e-6):
        """
        Calculate PageRank scores for all documents
        Returns: dict of doc_id -> pagerank_score
        """
        num_docs = len(self.doc_id_to_url)
        if num_docs == 0:
            return {}
        
        # Initialize ranks
        ranks = {doc_id: 1.0 / num_docs for doc_id in self.doc_id_to_url.keys()}
        
        for iteration in range(iterations):
            new_ranks = {}
            
            for doc_id in self.doc_id_to_url.keys():
                # Find all pages linking to this doc
                rank = (1 - damping_factor) / num_docs
                
                for source_doc_id, targets in self.graph.items():
                    if doc_id in targets:
                        # Divide source's rank by its out-degree
                        out_degree = len(targets)
                        rank += damping_factor * ranks[source_doc_id] / out_degree
                
                new_ranks[doc_id] = rank
            
            # Check for convergence
            max_diff = max(abs(new_ranks[doc_id] - ranks[doc_id]) 
                          for doc_id in ranks.keys())
            ranks = new_ranks
            
            if max_diff < tolerance:
                print(f"PageRank converged after {iteration + 1} iterations")
                break
        
        return ranks

    def hits(self, iterations=10, tolerance=1e-6):
        """
        Calculate HITS (Hub and Authority) scores
        Returns: tuple of (hubs, authorities) dicts mapping doc_id -> score
        """
        num_docs = len(self.doc_id_to_url)
        if num_docs == 0:
            return {}, {}
        
        # Initialize scores
        hubs = {doc_id: 1.0 for doc_id in self.doc_id_to_url.keys()}
        authorities = {doc_id: 1.0 for doc_id in self.doc_id_to_url.keys()}
        
        for iteration in range(iterations):
            # Authority update: sum of hub scores of pages pointing to it
            new_authorities = {}
            for doc_id in self.doc_id_to_url.keys():
                score = 0
                for source_doc_id, targets in self.graph.items():
                    if doc_id in targets:
                        score += hubs[source_doc_id]
                new_authorities[doc_id] = score
            
            # Hub update: sum of authority scores of pages it points to
            new_hubs = {}
            for doc_id in self.doc_id_to_url.keys():
                score = sum(authorities[target_doc_id] 
                           for target_doc_id in self.graph.get(doc_id, set()))
                new_hubs[doc_id] = score
            
            # Normalize
            auth_sum = sum(new_authorities.values())
            if auth_sum > 0:
                new_authorities = {k: v / auth_sum for k, v in new_authorities.items()}
            
            hub_sum = sum(new_hubs.values())
            if hub_sum > 0:
                new_hubs = {k: v / hub_sum for k, v in new_hubs.items()}
            
            # Check convergence
            max_auth_diff = max(abs(new_authorities[doc_id] - authorities[doc_id]) 
                               for doc_id in authorities.keys())
            max_hub_diff = max(abs(new_hubs[doc_id] - hubs[doc_id]) 
                              for doc_id in hubs.keys())
            
            hubs = new_hubs
            authorities = new_authorities
            
            if max(max_auth_diff, max_hub_diff) < tolerance:
                print(f"HITS converged after {iteration + 1} iterations")
                break
        
        return hubs, authorities

    def save_scores(self, output_dir, pagerank_scores, hits_hubs, hits_authorities):
        """Save link analysis scores to msgpack files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save PageRank
        pr_path = os.path.join(output_dir, "pagerank.msgpack")
        with open(pr_path, "wb") as f:
            f.write(msgpack.packb(pagerank_scores, use_bin_type=True))
        print(f"PageRank scores saved to: {pr_path}")
        
        # Save HITS
        hits_data = {
            "hubs": hits_hubs,
            "authorities": hits_authorities
        }
        hits_path = os.path.join(output_dir, "hits.msgpack")
        with open(hits_path, "wb") as f:
            f.write(msgpack.packb(hits_data, use_bin_type=True))
        print(f"HITS scores saved to: {hits_path}")

    def load_scores(self, output_dir):
        """Load pre-computed link analysis scores"""
        pagerank_scores = {}
        hits_scores = {"hubs": {}, "authorities": {}}
        
        pr_path = os.path.join(output_dir, "pagerank.msgpack")
        if os.path.exists(pr_path):
            with open(pr_path, "rb") as f:
                pagerank_scores = msgpack.unpackb(f.read(), raw=False)
        
        hits_path = os.path.join(output_dir, "hits.msgpack")
        if os.path.exists(hits_path):
            with open(hits_path, "rb") as f:
                hits_scores = msgpack.unpackb(f.read(), raw=False)
        
        return pagerank_scores, hits_scores