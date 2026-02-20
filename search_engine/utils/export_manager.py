import csv
import json
import os
import pickle
from django.conf import settings

class ExportManager:
    """
    Handles CSV and serialized Model exports.
    """
    
    @staticmethod
    def export_to_csv(data, headers, output_path):
        """Export list of dicts to CSV."""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
        return output_path

    @staticmethod
    def export_dataset_to_csv(json_path, output_path):
        """Convert classification JSON to a professional binary-labeled CSV format."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 1. Identify all unique labels in the dataset
        all_unique_labels = set()
        for item in data:
            all_unique_labels.update(item.get('labels', []))
        sorted_labels = sorted(list(all_unique_labels))
        
        # 2. Transform data into binary format
        export_rows = []
        for item in data:
            row = {
                'id': item.get('id'),
                'text': item.get('text'),
                'all_labels': ", ".join(item.get('labels', []))
            }
            # Add binary indicators (1 for present, 0 for absent)
            for label in sorted_labels:
                row[label] = 1 if label in item.get('labels', []) else 0
            export_rows.append(row)
            
        # 3. Define headers
        headers = ['id', 'text'] + sorted_labels + ['all_labels']
        
        return ExportManager.export_to_csv(export_rows, headers, output_path)

    @staticmethod
    def export_trained_model(classifier, output_dir):
        """Save .pkl files for model, vectorizer, and binarizer."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        model_path = os.path.join(output_dir, 'classifier_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(classifier.model, f)
            
        vec_path = os.path.join(output_dir, 'vectorizer.pkl')
        with open(vec_path, 'wb') as f:
            pickle.dump(classifier.vectorizer, f)
            
        lb_path = os.path.join(output_dir, 'label_binarizer.pkl')
        with open(lb_path, 'wb') as f:
            pickle.dump(classifier.mlb, f)
            
        # Add basic README
        readme_path = os.path.join(output_dir, 'README.txt')
        with open(readme_path, 'w') as f:
            f.write("Classifier Export Files\n")
            f.write("========================\n")
            f.write("Usage:\n")
            f.write("1. Load vectorizer.pkl to transform new text.\n")
            f.write("2. Load classifier_model.pkl to predict labels.\n")
            f.write("3. Use label_binarizer.pkl to decode IDs to names.\n")
            
        return output_dir
