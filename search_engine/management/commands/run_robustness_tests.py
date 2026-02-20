import json
import os
from django.core.management.base import BaseCommand
from search_engine.ml.classifier import DocumentClassifier
from search_engine.utils.preprocessor import TextPreprocessor

class Command(BaseCommand):
    help = 'Run scientific robustness tests for the document classifier'

    def handle(self, *args, **options):
        # 1. Load test cases
        json_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            'data', 'test_cases.json'
        )
        
        if not os.path.exists(json_path):
            self.stdout.write(self.style.ERROR(f'Test cases file not found at {json_path}'))
            return
            
        with open(json_path, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
            
        # 2. Initialize
        self.stdout.write(self.style.SUCCESS(f'--- Starting Robustness Suite ({len(test_cases)} cases) ---'))
        classifier = DocumentClassifier()
        # Initialize model components if not already done
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            'data', 'models', 'classifier_model.pkl'
        )
        if os.path.exists(model_path):
            classifier.load_model(model_path)
        else:
            self.stdout.write(self.style.WARNING("Warning: Model not found. Running with untrained classifier."))

        passed = 0
        
        # 3. Run Loop
        for case in test_cases:
            prediction = classifier.predict(case['input'])
            predicted_set = set(prediction['predicted_labels'])
            expected_set = set(case['expected'])
            is_pass = predicted_set == expected_set
            
            status = self.style.SUCCESS('PASS') if is_pass else self.style.ERROR('FAIL')
            if is_pass: passed += 1
            
            self.stdout.write(f"[{case['type']}] Input: \"{case['input'][:40]}...\"")
            self.stdout.write(f"  Expected: {case['expected']}")
            self.stdout.write(f"  Got:      {prediction['predicted_labels']} -> {status}\n")

        # 4. Summary
        accuracy = (passed / len(test_cases)) * 100
        self.stdout.write(self.style.SUCCESS(f"--- Results: {passed}/{len(test_cases)} Passed ({accuracy:.1f}%) ---"))
