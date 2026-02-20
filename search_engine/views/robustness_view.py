import json
import os
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from .classifier_view import get_classifier
from ..utils.preprocessor import TextPreprocessor

def robustness_dashboard(request):
    """View to display the robustness testing suite."""
    return render(request, 'search_engine/robustness.html')

def run_robustness_api(request):
    """API endpoint to run all test cases and return results."""
    try:
        # 1. Load test cases
        json_path = settings.TEST_CASES_FILE
        
        if not os.path.exists(json_path):
            return JsonResponse({'error': 'Test cases file not found'}, status=404)
            
        with open(json_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
            
        # 2. Initialize classifier and preprocessor
        classifier = get_classifier()
        preprocessor = TextPreprocessor()
        
        results = []
        passed = 0
        total_confidence = 0
        
        # 3. Process each test case
        for case in test_cases:
            prediction = classifier.predict(case['input'])
            
            # Check if prediction matches expected (multi-label check)
            predicted_set = set(prediction['predicted_labels'])
            expected_set = set(case['expected'])
            
            # A "Pass" means the predicted labels exactly match expected
            # Or if expected is empty, it shouldn't predict any of our core categories
            is_pass = predicted_set == expected_set
            
            if is_pass:
                passed += 1
                
            # Get max probability for confidence
            max_prob = max(prediction['all_probabilities'].values()) if prediction['all_probabilities'] else 0
            total_confidence += max_prob
            
            # Get preprocessing steps for transparency
            steps = preprocessor.get_preprocessing_steps(case['input'])
            
            results.append({
                'case': case,
                'predicted': prediction['predicted_labels'],
                'all_probs': prediction['all_probabilities'],
                'confidence': prediction['confidence_level'],
                'max_prob_value': round(max_prob * 100, 1),
                'pass': is_pass,
                'preprocessed': prediction['preprocessed_text'],
                'steps': steps
            })
            
        # 4. Calculate summary stats
        total = len(test_cases)
        summary = {
            'total': total,
            'passed': passed,
            'failed': total - passed,
            'accuracy': round((passed / total) * 100, 1) if total > 0 else 0,
            'avg_confidence': round((total_confidence / total) * 100, 1) if total > 0 else 0
        }
        
        return JsonResponse({
            'status': 'success',
            'summary': summary,
            'results': results
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
