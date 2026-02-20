"""
Multi-Label Document Classifier

This module implements true multi-label classification for documents,
allowing a single document to be assigned multiple categories.

Features:
- MultiOutputClassifier with MultinomialNB
- MultiLabelBinarizer for label encoding
- TfidfVectorizer with custom preprocessing
- Threshold-based prediction (default: 0.30)
- Comprehensive metrics and evaluation
- Top features extraction per label
"""

import json
import os
import joblib
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    precision_recall_fscore_support,
    multilabel_confusion_matrix,
    classification_report
)
from sklearn.multioutput import ClassifierChain
# Django setup removed to prevent conflicts when running via manage.py
# Only needed for standalone execution of this specific file
if __name__ == "__main__":
    import sys
    import django
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coventry_search.settings')
    django.setup()

from search_engine.utils.preprocessor import TextPreprocessor


class DocumentClassifier:
    """
    Multi-label document classifier using Naive Bayes.
    
    Supports assigning multiple categories to a single document
    based on probability thresholds.
    """
    
    def __init__(self, threshold: float = 0.30):
        """
        Initialize the classifier.
        
        Args:
            threshold: Probability threshold for label prediction (default: 0.30)
        """
        self.threshold = threshold
        self.preprocessor = TextPreprocessor()
        
        # Model components
        self.vectorizer = None
        self.mlb = None  # MultiLabelBinarizer
        self.classifier = None
        self.label_names = None
        
        # Training data for reference
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self, json_path: str) -> Tuple[List[str], List[List[str]]]:
        """
        Load data from classification_multilabel.json.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            Tuple of (texts, labels) where labels is a list of label lists
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for item in data:
            texts.append(item['text'])
            # Ensure labels is a list
            item_labels = item['labels']
            if isinstance(item_labels, str):
                item_labels = [item_labels]
            labels.append(item_labels)
        
        return texts, labels
    
    def train(self, json_path: str, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train the multi-label classifier.
        
        Args:
            json_path: Path to classification_multilabel.json
            test_size: Proportion of data for testing (default: 0.2)
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing training metrics
        """
        print("Loading data...")
        texts, labels = self.load_data(json_path)
        
        print(f"Loaded {len(texts)} documents")
        print(f"Label distribution:")
        label_counts = {}
        for label_list in labels:
            for label in label_list:
                label_counts[label] = label_counts.get(label, 0) + 1
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state
        )
        
        print(f"\nTraining set: {len(X_train)} documents")
        print(f"Test set: {len(X_test)} documents")
        
        # Initialize MultiLabelBinarizer
        print("\nBinarizing labels...")
        self.mlb = MultiLabelBinarizer()
        y_train_binary = self.mlb.fit_transform(y_train)
        y_test_binary = self.mlb.transform(y_test)
        
        self.label_names = self.mlb.classes_.tolist()
        print(f"Labels: {self.label_names}")
        
        # Initialize TF-IDF Vectorizer with custom preprocessor
        print("\nVectorizing text...")
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.preprocessor.process_text,
            token_pattern=None,
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        # Train ClassifierChain
        print("\nTraining classifier chain...")
        base_clf = MultinomialNB(alpha=0.1)
        self.classifier = ClassifierChain(base_clf, order='random', random_state=random_state)
        self.classifier.fit(X_train_vec, y_train_binary)
        
        # Store for later use
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Evaluate
        print("\nEvaluating model...")
        metrics = self._evaluate(X_test_vec, y_test_binary, y_test)
        
        return metrics
    
    def _evaluate(self, X_test_vec, y_test_binary, y_test_labels) -> Dict:
        """
        Evaluate the model and calculate metrics.
        
        Args:
            X_test_vec: Vectorized test features
            y_test_binary: Binary label matrix
            y_test_labels: Original label lists
            
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_pred_binary = self.classifier.predict(X_test_vec)
        
        # For ClassifierChain, we can get probabilities if the base estimator supports it
        # but predict_proba on ClassifierChain returns internal state sometimes.
        # However, for thresholding we need probabilities.
        # MultinomialNB supports predict_proba.
        y_pred_probs = self.classifier.predict_proba(X_test_vec)
        
        # Calculate metrics
        metrics = {
            'threshold': self.threshold,
            'subset_accuracy': accuracy_score(y_test_binary, y_pred_binary),
            'hamming_loss': hamming_loss(y_test_binary, y_pred_binary),
        }
        
        # Precision, Recall, F1 for different averaging methods
        for average in ['micro', 'macro', 'weighted', 'samples']:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test_binary, y_pred_binary, average=average, zero_division=0
            )
            metrics[f'precision_{average}'] = float(precision)
            metrics[f'recall_{average}'] = float(recall)
            metrics[f'f1_{average}'] = float(f1)
        
        # Per-label metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_binary, y_pred_binary, average=None, zero_division=0
        )
        
        metrics['per_label'] = {}
        for i, label in enumerate(self.label_names):
            metrics['per_label'][label] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        # Confusion matrices per label
        cm = multilabel_confusion_matrix(y_test_binary, y_pred_binary)
        metrics['confusion_matrices'] = {}
        for i, label in enumerate(self.label_names):
            metrics['confusion_matrices'][label] = cm[i].tolist()
        
        return metrics
    
    def predict(self, text: str) -> Dict:
        """
        Predict labels for a single text document.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with predictions, probabilities, and metadata
        """
        if self.classifier is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        # Preprocess
        preprocessed_tokens = self.preprocessor.process_text(text)
        preprocessed_text = ' '.join(preprocessed_tokens)
        
        # Vectorize
        X_vec = self.vectorizer.transform([text])
        
        # Get probabilities
        probs = self.classifier.predict_proba(X_vec)[0]
        
        # Create probability dictionary
        all_probabilities = {
            label: float(prob) 
            for label, prob in zip(self.label_names, probs)
        }
        
        # Apply threshold to get predicted labels
        predicted_labels = [
            label for label, prob in zip(self.label_names, probs)
            if prob >= self.threshold
        ]
        
        # If no labels meet threshold, use the highest probability
        if not predicted_labels:
            max_idx = np.argmax(probs)
            predicted_labels = [self.label_names[max_idx]]
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence(probs, predicted_labels)
        
        # Get top features
        top_features = self._get_top_features(X_vec, n=10)
        
        return {
            'predicted_labels': predicted_labels,
            'all_probabilities': all_probabilities,
            'confidence_level': confidence_level,
            'top_features': top_features,
            'text': text,
            'preprocessed_text': preprocessed_text
        }
    
    def _calculate_confidence(self, probs: np.ndarray, predicted_labels: List[str]) -> str:
        """
        Calculate confidence level based on probabilities.
        
        Args:
            probs: Array of probabilities
            predicted_labels: List of predicted labels
            
        Returns:
            Confidence level: 'high', 'medium', or 'low'
        """
        if not predicted_labels:
            return 'low'
        
        # Get probabilities of predicted labels
        predicted_probs = [
            probs[self.label_names.index(label)]
            for label in predicted_labels
        ]
        
        max_prob = max(predicted_probs)
        avg_prob = np.mean(predicted_probs)
        
        # Confidence thresholds
        if max_prob >= 0.7 and avg_prob >= 0.5:
            return 'high'
        elif max_prob >= 0.5 or avg_prob >= 0.35:
            return 'medium'
        else:
            return 'low'
    
    def _get_top_features(self, X_vec, n: int = 10) -> List[Dict]:
        """
        Get top features (words) contributing to the prediction with their scores.
        
        Args:
            X_vec: Vectorized features
            n: Number of top features to return
            
        Returns:
            List of dictionaries with 'word' and 'score'
        """
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get non-zero features for this document
        doc_features = X_vec.toarray()[0]
        
        # Get indices of top features
        top_indices = np.argsort(doc_features)[-n:][::-1]
        
        # Filter out zero values and create result list
        top_features = []
        for i in top_indices:
            if doc_features[i] > 0:
                top_features.append({
                    'word': str(feature_names[i]),
                    'score': float(doc_features[i])
                })
        
        return top_features[:n]
    
    def save_model(self, model_path: str, metrics_path: Optional[str] = None, metrics: Optional[Dict] = None):
        """
        Save the trained model and optionally metrics.
        
        Args:
            model_path: Path to save the model
            metrics_path: Optional path to save metrics JSON
            metrics: Optional metrics dictionary to save
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model components
        model_data = {
            'vectorizer': self.vectorizer,
            'mlb': self.mlb,
            'classifier': self.classifier,
            'label_names': self.label_names,
            'threshold': self.threshold,
            'preprocessor_config': self.preprocessor.get_config()
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save metrics if provided
        if metrics_path and metrics:
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to: {metrics_path}")
    
    def load_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model_data = joblib.load(model_path)
        
        self.vectorizer = model_data['vectorizer']
        self.mlb = model_data['mlb']
        self.classifier = model_data['classifier']
        self.label_names = model_data['label_names']
        self.threshold = model_data['threshold']
        
        print(f"Model loaded from: {model_path}")
        print(f"Labels: {self.label_names}")
        print(f"Threshold: {self.threshold}")
    
    def get_feature_importance(self, label: str, n: int = 20) -> List[Tuple[str, float]]:
        """
        Get most important features for a specific label.
        
        Args:
            label: Label name
            n: Number of top features to return
            
        Returns:
            List of (feature, importance) tuples
        """
        if label not in self.label_names:
            raise ValueError(f"Unknown label: {label}")
        
        label_idx = self.label_names.index(label)
        estimator = self.classifier.estimators_[label_idx]
        
        # Get feature log probabilities
        feature_log_probs = estimator.feature_log_prob_[1]  # Positive class
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top features
        num_features = min(len(feature_names), len(feature_log_probs))
        all_indices = np.argsort(feature_log_probs)
        top_indices = [i for i in all_indices if i < num_features][-n:][::-1]
        
        top_features = [
            (feature_names[i], float(feature_log_probs[i]))
            for i in top_indices
        ]
        
        return top_features
