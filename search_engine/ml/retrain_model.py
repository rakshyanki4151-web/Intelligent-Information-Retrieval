import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'coventry_search.settings')
django.setup()

from search_engine.ml.classifier import DocumentClassifier

def retrain_and_save():
    classifier = DocumentClassifier()
    
    # Paths (Unified with Assignment root)
    json_path = settings.CLASSIFIER_DATA
    model_path = os.path.join(settings.DATA_DIR, 'classifier_model.pkl')
    
    print(f"Retraining model from {json_path}...")
    metrics = classifier.train(json_path)
    
    print("Saving model...")
    classifier.save_model(model_path)
    print(f"Model saved to {model_path}")
    print("Metrics:", metrics)

if __name__ == "__main__":
    retrain_and_save()
