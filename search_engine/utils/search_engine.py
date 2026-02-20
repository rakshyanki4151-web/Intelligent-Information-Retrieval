# Core Search Engine: IR Logic Core.
# Features: Candidate Filtering, Processing Consistency, JSON Persistence, Contribution Scoring.
import math
import json
import os
from collections import defaultdict, Counter
from .preprocessor import TextPreprocessor

# Initialize global preprocessor for uniform processing
_preprocessor = TextPreprocessor()

class WeightedInvertedIndex:
    """
    Implementation of a Weighted Inverted Index.
    Weights: Title (3.0), Authors (2.5), Keywords (2.0), Year (1.5), Abstract (1.0).
    """
    
    def __init__(self):
        self.index = defaultdict(list)
        self.documents = {}
        self.doc_vectors = {}  # {doc_id: {token: weighted_tfidf}}
        self.weights = {
            'title': 3.0,
            'authors': 2.5,
            'keywords': 2.0,
            'year': 1.5,
            'abstract': 1.0
        }

    def add_document(self, doc_data, doc_id, rebuild=True):
        """Build index entry for a document with field weights"""
        self.documents[doc_id] = doc_data
        
        # Process each field according to its weight
        for field, weight in self.weights.items():
            content = doc_data.get(field, "")
            if isinstance(content, list):
                content = " ".join(content)
            
            # Use shared preprocessor for consistency
            tokens = _preprocessor.process_text(str(content))
            token_counts = Counter(tokens)
            
            for token, freq in token_counts.items():
                # Store (doc_id, weighted_freq, field)
                self.index[token].append({
                    'id': doc_id,
                    'w_freq': freq * weight,
                    'field': field
                })
        
        # Note: Vector update can be deferred for batch indexing performance
        if rebuild:
            self._rebuild_vectors()

    def _rebuild_vectors(self):
        """Internal: Recalculate TF-IDF vectors across all candidate documents"""
        N = len(self.documents)
        if N == 0: return
        
        self.doc_vectors = defaultdict(lambda: defaultdict(float))
        
        for token, postings in self.index.items():
            # Calculate IDF for this token
            df = len(set(p['id'] for p in postings))
            idf = math.log(N / (df + 1)) + 1
            
            for p in postings:
                self.doc_vectors[p['id']][token] += p['w_freq'] * idf

    def search(self, query, top_k=50):
        """
        1. Performance Optimization: Candidate Filtering
        2. Logic: Contribution-based Ranking
        """
        if not query.strip(): return []
        
        # Processing Consistency: Process query exactly like the index
        query_tokens = _preprocessor.process_text(query)
        if not query_tokens: return []
        
        query_counts = Counter(query_tokens)
        N = len(self.documents)
        
        # 1. Filter Candidate Set (Only docs containing at least one query token)
        candidate_ids = set()
        query_vector = {}
        
        for token, tf in query_counts.items():
            if token in self.index:
                postings = self.index[token]
                df = len(set(p['id'] for p in postings))
                idf = math.log(N / (df + 1)) + 1
                query_vector[token] = tf * idf
                
                for p in postings:
                    candidate_ids.add(p['id'])

        if not candidate_ids: return []

        # 2. Similarity Calculation (Contribution math)
        results = []
        for doc_id in candidate_ids:
            doc_data = self.documents[doc_id]
            doc_vec = self.doc_vectors[doc_id]
            
            # Calculate total score (Simplified Cosine or Weighted Sum)
            total_score = 0
            field_scores = defaultdict(float) # Contribution Tracking
            
            for token, q_tfidf in query_vector.items():
                if token in doc_vec:
                    # Contribution breakdown per token/field
                    # we must look at the postings for this doc specifically
                    postings = [p for p in self.index[token] if p['id'] == doc_id]
                    token_idf = query_vector[token] / query_counts[token]
                    
                    for p in postings:
                        # Logic: Field Contribution = TF * FieldWeight * IDF
                        contribution = p['w_freq'] * token_idf
                        field_scores[p['field']] += contribution
                        total_score += contribution
            
            # Normalize field scores for the scoring requirement
            total_sum = sum(field_scores.values()) or 1.0
            contribution_percentages = {
                f: round((s / total_sum) * 100, 1) for f, s in field_scores.items()
            }
            
            results.append({
                'id': doc_id,
                'data': self.documents[doc_id],
                'score': round(total_score, 4),
                'contribution': contribution_percentages,
                'snippet': self._generate_snippet(self.documents[doc_id].get('abstract', ''), query_tokens)
            })

        # Sort by total rank
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def _generate_snippet(self, text, query_tokens, window=20):
        """Enhanced Feature: Keyword-in-Context Snippet Generation"""
        if not text: return ""
        
        words = text.split()
        clean_words = [_preprocessor.process_text(w)[0] if _preprocessor.process_text(w) else "" for w in words]
        
        # Find first occurrence of any query token
        match_idx = -1
        q_set = set(query_tokens)
        for i, cw in enumerate(clean_words):
            if cw in q_set:
                match_idx = i
                break
        
        if match_idx == -1:
            return " ".join(words[:window]) + "..."
            
        start = max(0, match_idx - window // 2)
        end = min(len(words), start + window)
        
        snippet_words = words[start:end]
        
        # Highlight matches
        highlighted = []
        for i in range(start, end):
            word = words[i]
            clean_word = clean_words[i]
            if clean_word in q_set:
                highlighted.append(f"<mark>{word}</mark>")
            else:
                highlighted.append(word)
        
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(words) else ""
        return prefix + " ".join(highlighted) + suffix

    def save(self, filepath):
        """3. Modernized Persistence (JSON over Pickle)"""
        data = {
            'documents': self.documents,
            'index': dict(self.index),
            'doc_vectors': {k: dict(v) for k, v in self.doc_vectors.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
            
    def load(self, filepath):
        """Load from JSON with re-initialization"""
        if not os.path.exists(filepath): return False
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.documents = data['documents']
                self.index = defaultdict(list, data['index'])
                self.doc_vectors = defaultdict(lambda: defaultdict(float))
                for k, v in data['doc_vectors'].items():
                    self.doc_vectors[k] = defaultdict(float, v)
            return True
        except:
            return False

    def get_document_count(self):
        return len(self.documents)

# Global search index instance
search_index = WeightedInvertedIndex()
