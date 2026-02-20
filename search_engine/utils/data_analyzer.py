import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from .preprocessor import TextPreprocessor

class DataAnalyzer:
    """
    Utility class for analyzing multi-label dataset quality and statistics.
    """
    
    def __init__(self, json_path):
        self.json_path = json_path
        self.data = []
        self.preprocessor = TextPreprocessor()
        self.load_data()

    def load_data(self):
        """Load data from JSON file."""
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

    def get_basic_stats(self):
        """Calculate high-level overview statistics."""
        if not self.data:
            return {}
            
        total_docs = len(self.data)
        all_labels = []
        doc_lengths = []
        dates = []
        
        for doc in self.data:
            labels = doc.get('labels', [])
            all_labels.extend(labels)
            
            text = doc.get('text', '')
            doc_lengths.append(len(text.split()))
            
            date_str = doc.get('date_collected', '')
            if date_str:
                try:
                    dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
                except ValueError:
                    pass
        
        unique_labels = sorted(list(set(all_labels)))
        avg_labels = len(all_labels) / total_docs if total_docs > 0 else 0
        
        date_range = "N/A"
        if dates:
            min_date = min(dates).strftime('%b %d')
            max_date = max(dates).strftime('%b %d, %Y')
            date_range = f"{min_date} - {max_date}"

        return {
            'total_documents': total_docs,
            'unique_labels_count': len(unique_labels),
            'unique_labels': unique_labels,
            'avg_labels_per_doc': round(avg_labels, 2),
            'date_range': date_range,
            'avg_doc_length': round(sum(doc_lengths) / total_docs if total_docs > 0 else 0, 1)
        }

    def get_category_distribution(self):
        """Calculate distribution of documents per category."""
        category_counts = Counter()
        for doc in self.data:
            for label in doc.get('labels', []):
                category_counts[label] += 1
        
        # Check for imbalance (>2x difference)
        max_count = max(category_counts.values()) if category_counts else 0
        min_count = min(category_counts.values()) if category_counts else 0
        is_imbalanced = max_count > (min_count * 2) if min_count > 0 else False
        
        return {
            'counts': dict(category_counts),
            'is_imbalanced': is_imbalanced,
            'ratio': round(max_count / min_count, 2) if min_count > 0 else 0
        }

    def get_cooccurrence_matrix(self):
        """Calculate how often labels appear together."""
        matrix = defaultdict(lambda: defaultdict(int))
        labels = set()
        
        for doc in self.data:
            doc_labels = doc.get('labels', [])
            for i, l1 in enumerate(doc_labels):
                labels.add(l1)
                for l2 in doc_labels[i+1:]:
                    matrix[l1][l2] += 1
                    matrix[l2][l1] += 1
        
        sorted_labels = sorted(list(labels))
        return {
            'labels': sorted_labels,
            'matrix': matrix
        }

    def perform_quality_checks(self):
        """Run automated quality validation checks."""
        duplicates = 0
        seen_texts = set()
        short_docs = 0
        
        for doc in self.data:
            text = doc.get('text', '')
            if text in seen_texts:
                duplicates += 1
            seen_texts.add(text)
            
            if len(text.split()) < 10:
                short_docs += 1
                
        return {
            'duplicate_count': duplicates,
            'short_doc_count': short_docs,
            'encoding_valid': True, # Assumed if JSON loaded successfully
            'total_docs': len(self.data)
        }

    def get_vocabulary_stats(self):
        """Analyze vocabulary and word frequencies."""
        token_counter = Counter()
        all_tokens = []
        
        for doc in self.data:
            text = doc.get('text', '')
            tokens = self.preprocessor.process_text(text)
            token_counter.update(tokens)
            all_tokens.extend(tokens)
            
        return {
            'unique_tokens': len(token_counter),
            'most_common': token_counter.most_common(15),
            'total_tokens': len(all_tokens)
        }

    def get_source_distribution(self):
        """Count documents per source."""
        source_counts = Counter()
        source_links = {}
        
        for doc in self.data:
            source = doc.get('source', 'Unknown')
            source_counts[source] += 1
            if source not in source_links and doc.get('url'):
                source_links[source] = doc.get('url')
                
        return {
            'counts': dict(source_counts),
            'links': source_links
        }
