"""
Views for document classification interface
"""

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import json
import os

from search_engine.ml.classifier import DocumentClassifier


# Global classifier instance
_classifier = None

from django.conf import settings

def get_classifier():
    """Get or initialize the classifier"""
    global _classifier
    if _classifier is None:
        _classifier = DocumentClassifier()
        # Try to load existing model
        model_path = os.path.join(settings.DATA_DIR, 'classifier_model.pkl')
        if os.path.exists(model_path):
            try:
                _classifier.load_model(model_path)
            except Exception as e:
                print(f"Warning: Could not load model: {e}")
    return _classifier


def classify_view(request):
    """Main classification page"""
    return render(request, 'classifier/classify.html')


@csrf_exempt
@require_http_methods(["POST"])
def classify_api(request):
    """API endpoint for document classification"""
    try:
        # Parse JSON body
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        threshold = float(data.get('threshold', 0.30))
        
        if not text:
            return JsonResponse({
                'error': 'No text provided'
            }, status=400)
        
        # Get classifier
        classifier = get_classifier()
        
        # Update threshold
        classifier.threshold = threshold
        
        # Predict
        result = classifier.predict(text)
        
        # Get preprocessing steps
        from search_engine.utils.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()
        steps = preprocessor.get_preprocessing_steps(text)
        
        # Format preprocessing steps
        preprocessing_steps = {}
        for i, step in enumerate(steps, 1):
            preprocessing_steps[f'step{i}'] = {
                'name': step['step'],
                'result': step['result']
            }
        
        # Return enhanced result
        return JsonResponse({
            'predicted_labels': result['predicted_labels'],
            'all_probabilities': result['all_probabilities'],
            'confidence_level': result['confidence_level'],
            'top_features': result['top_features'],
            'text': result['text'],
            'preprocessed_text': result['preprocessed_text'],
            'preprocessing_steps': preprocessing_steps
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)
