"""
Django management command to test the multi-label classifier with diverse examples.

Usage:
    python manage.py test_classifier
"""

from django.core.management.base import BaseCommand
from django.conf import settings
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from search_engine.ml.classifier import DocumentClassifier


class Command(BaseCommand):
    help = 'Test the multi-label document classifier with diverse examples'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model-path',
            type=str,
            default=None,
            help='Path to trained model (default: data/models/classifier_model.pkl)'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('=' * 80))
        self.stdout.write(self.style.SUCCESS(' MULTI-LABEL CLASSIFIER TEST SUITE'))
        self.stdout.write(self.style.SUCCESS('=' * 80))
        
        # Get model path
        model_path = options['model_path']
        if model_path is None:
            model_path = os.path.join(
                settings.DATA_DIR, 'classifier_model.pkl'
            )
        
        if not os.path.exists(model_path):
            self.stdout.write(self.style.ERROR(f'\nError: Model not found: {model_path}'))
            self.stdout.write(self.style.WARNING('\nPlease train the model first using: python manage.py train_model'))
            return
        
        # Load classifier
        classifier = DocumentClassifier()
        classifier.load_model(model_path)
        
        # Test cases
        test_cases = [
            # Single-label: Business
            {
                "name": "Pure Business - Stock Market",
                "text": "Stock markets rallied today as investors reacted positively to strong quarterly earnings reports from major technology companies. The S&P 500 index gained 2.3% while the Nasdaq composite rose 3.1%.",
                "expected": ["business"]
            },
            {
                "name": "Pure Business - Corporate Merger",
                "text": "Microsoft announces $69 billion acquisition of gaming giant Activision Blizzard in the largest deal in gaming industry history. Shareholders will vote on the merger next quarter.",
                "expected": ["business"]
            },
            {
                "name": "Pure Business - IPO",
                "text": "Electric vehicle startup Rivian files for initial public offering seeking to raise $8 billion. The company plans to list on the NASDAQ exchange under the ticker symbol RIVN.",
                "expected": ["business"]
            },
            
            # Single-label: Entertainment
            {
                "name": "Pure Entertainment - Movie Release",
                "text": "The latest Marvel superhero film breaks box office records with $350 million opening weekend globally. Critics praise the stunning visual effects and compelling storyline.",
                "expected": ["entertainment"]
            },
            {
                "name": "Pure Entertainment - Music Concert",
                "text": "Beyoncé announces world tour with 50 stadium shows across North America and Europe. Tickets go on sale next Friday with prices ranging from $75 to $500.",
                "expected": ["entertainment"]
            },
            {
                "name": "Pure Entertainment - Awards Show",
                "text": "The Academy Awards ceremony celebrates cinema's biggest night with surprise wins in major categories. Best Picture went to an independent film that cost just $2 million to produce.",
                "expected": ["entertainment"]
            },
            
            # Single-label: Health
            {
                "name": "Pure Health - Vaccine Development",
                "text": "Scientists develop new mRNA vaccine showing 95% effectiveness against emerging virus strain. Clinical trials involved 40,000 participants across multiple countries.",
                "expected": ["health"]
            },
            {
                "name": "Pure Health - Medical Research",
                "text": "Breakthrough study reveals genetic markers that predict Alzheimer's disease decades before symptoms appear. Researchers hope this leads to earlier intervention and prevention strategies.",
                "expected": ["health"]
            },
            {
                "name": "Pure Health - Public Health",
                "text": "CDC reports significant decline in childhood obesity rates following nationwide nutrition education programs. Experts attribute success to school lunch reforms and increased physical activity.",
                "expected": ["health"]
            },
            
            # Multi-label: Business + Health
            {
                "name": "Multi-label: Business + Health - Pharma Profits",
                "text": "Pharmaceutical giant Pfizer reports record quarterly profits as new weight-loss drug gains FDA approval, sending stock prices soaring 15% in morning trading.",
                "expected": ["business", "health"]
            },
            {
                "name": "Multi-label: Business + Health - Healthcare Insurance",
                "text": "Healthcare insurance companies face massive losses as new regulations require coverage of mental health services, forcing industry-wide restructuring and layoffs.",
                "expected": ["business", "health"]
            },
            {
                "name": "Multi-label: Business + Health - Telemedicine Startup",
                "text": "Telemedicine startup raises $500 million in Series C funding to expand virtual doctor consultations across rural America, addressing healthcare accessibility gaps.",
                "expected": ["business", "health"]
            },
            {
                "name": "Multi-label: Business + Health - Hospital Bankruptcy",
                "text": "Major hospital chains report bankruptcy filings as rising operational costs and nurse shortages create unsustainable financial pressures across the healthcare sector.",
                "expected": ["business", "health"]
            },
            
            # Multi-label: Business + Entertainment
            {
                "name": "Multi-label: Business + Entertainment - Disney Layoffs",
                "text": "Disney reports streaming subscriber losses for third consecutive quarter, prompting 7,000 layoffs across film studios and theme park divisions.",
                "expected": ["business", "entertainment"]
            },
            {
                "name": "Multi-label: Business + Entertainment - Concert Revenue",
                "text": "Taylor Swift's Eras Tour generates $2 billion in ticket revenue, becoming highest-grossing concert series in history and boosting local economies across 50 cities.",
                "expected": ["business", "entertainment"]
            },
            {
                "name": "Multi-label: Business + Entertainment - Streaming Wars",
                "text": "Netflix stock plummets 35% after announcing password-sharing crackdown backfires, losing 2 million subscribers and $15 billion in market capitalization.",
                "expected": ["business", "entertainment"]
            },
            
            # Multi-label: Health + Entertainment
            {
                "name": "Multi-label: Health + Entertainment - Athlete Mental Health",
                "text": "Professional athletes speak out about mental health struggles, with Olympic gymnast Simone Biles withdrawing from competition to prioritize psychological well-being.",
                "expected": ["health", "entertainment"]
            },
            {
                "name": "Multi-label: Health + Entertainment - Opioid Documentary",
                "text": "Documentary exploring America's opioid crisis wins Oscar, featuring interviews with addiction survivors and families devastated by pharmaceutical company negligence.",
                "expected": ["health", "entertainment"]
            },
            {
                "name": "Multi-label: Health + Entertainment - Celebrity Wellness",
                "text": "Celebrity chef Jamie Oliver launches campaign against childhood obesity, producing cooking shows teaching families healthy meal preparation on limited budgets.",
                "expected": ["health", "entertainment"]
            },
            {
                "name": "Multi-label: Health + Entertainment - Music Festival Safety",
                "text": "Music festival implements harm reduction strategies including drug testing and medical tents after overdose deaths, balancing entertainment with public health.",
                "expected": ["health", "entertainment"]
            },
        ]
        
        # Run tests
        correct = 0
        total = len(test_cases)
        
        for i, test in enumerate(test_cases, 1):
            self.stdout.write(self.style.SUCCESS(f'\n{"=" * 80}'))
            self.stdout.write(self.style.SUCCESS(f' TEST {i}/{total}: {test["name"]}'))
            self.stdout.write(self.style.SUCCESS(f'{"=" * 80}'))
            
            self.stdout.write(f'\nText: {test["text"][:100]}...')
            self.stdout.write(f'\nExpected Labels: {test["expected"]}')
            
            # Predict
            result = classifier.predict(test["text"])
            
            self.stdout.write(f'\nPredicted Labels: {result["predicted_labels"]}')
            self.stdout.write(f'Confidence Level: {result["confidence_level"]}')
            
            self.stdout.write('\nProbabilities:')
            for label, prob in sorted(result["all_probabilities"].items(), key=lambda x: x[1], reverse=True):
                marker = "[X]" if label in result["predicted_labels"] else "[ ]"
                self.stdout.write(f'  {marker} {label}: {prob:.4f}')
            
            try:
                feature_display = [f['word'] if isinstance(f, dict) else f for f in result["top_features"][:5]]
                self.stdout.write(f'\nTop Features: {", ".join(feature_display)}')
            except Exception:
                self.stdout.write(f'\nTop Features: {result["top_features"][:5]}')
            
            # Check if correct
            predicted_set = set(result["predicted_labels"])
            expected_set = set(test["expected"])
            
            if predicted_set == expected_set:
                self.stdout.write(self.style.SUCCESS('\n[OK] CORRECT'))
                correct += 1
            else:
                self.stdout.write(self.style.WARNING('\n[X] INCORRECT'))
                missing = expected_set - predicted_set
                extra = predicted_set - expected_set
                if missing:
                    self.stdout.write(self.style.WARNING(f'  Missing: {missing}'))
                if extra:
                    self.stdout.write(self.style.WARNING(f'  Extra: {extra}'))
        
        # Summary
        self.stdout.write(self.style.SUCCESS(f'\n{"=" * 80}'))
        self.stdout.write(self.style.SUCCESS(' TEST SUMMARY'))
        self.stdout.write(self.style.SUCCESS(f'{"=" * 80}'))
        
        accuracy = correct / total * 100
        self.stdout.write(f'\nTotal Tests: {total}')
        self.stdout.write(f'Correct: {correct}')
        self.stdout.write(f'Incorrect: {total - correct}')
        self.stdout.write(f'Accuracy: {accuracy:.1f}%')
        
        if accuracy >= 80:
            self.stdout.write(self.style.SUCCESS('\n[OK] EXCELLENT PERFORMANCE'))
        elif accuracy >= 60:
            self.stdout.write(self.style.WARNING('\n[!] GOOD PERFORMANCE'))
        else:
            self.stdout.write(self.style.ERROR('\n[X] NEEDS IMPROVEMENT'))
