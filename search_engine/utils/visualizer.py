import json
import numpy as np

class MetricsVisualizer:
    """Utility to generate data for interactive charts."""
    
    @staticmethod
    def get_confusion_matrix_data(metrics):
        """Format confusion matrices for Plotly heatmap."""
        cms = metrics.get('confusion_matrices', {})
        label_names = list(cms.keys())
        
        heatmaps = {}
        for label, cm in cms.items():
            # cm is [[TN, FP], [FN, TP]]
            heatmaps[label] = {
                'z': cm,
                'x': ['Predicted Negative', 'Predicted Positive'],
                'y': ['Actual Negative', 'Actual Positive'],
                'type': 'heatmap',
                'colorscale': 'Blues'
            }
        return heatmaps

    @staticmethod
    def get_feature_importance_data(classifier, n=20):
        """Get top features for all labels for bar charts."""
        importance_data = {}
        for label in classifier.label_names:
            top_features = classifier.get_feature_importance(label, n=n)
            importance_data[label] = {
                'words': [f[0] for f in top_features],
                'scores': [f[1] for f in top_features]
            }
        return importance_data

    @staticmethod
    def get_confidence_distribution_data(y_pred_probs, label_names):
        """Prepare histogram data for confidences."""
        dist_data = {}
        for i, label in enumerate(label_names):
            probs = y_pred_probs[:, i].tolist()
            dist_data[label] = probs
        return dist_data
