"""
Django management command to train the multi-label document classifier.

Usage:
    python manage.py train_model
"""

from django.core.management.base import BaseCommand
from django.conf import settings
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from search_engine.ml.classifier import DocumentClassifier


class Command(BaseCommand):
    help = 'Train the multi-label document classifier'

    def add_arguments(self, parser):
        parser.add_argument(
            '--data-file',
            type=str,
            default=None,
            help='Path to classification_multilabel.json (default: search_engine/data/raw/classification_multilabel.json)'
        )
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.30,
            help='Probability threshold for label prediction (default: 0.30)'
        )
        parser.add_argument(
            '--test-size',
            type=float,
            default=0.2,
            help='Proportion of data for testing (default: 0.2)'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('=' * 80))
        self.stdout.write(self.style.SUCCESS(' MULTI-LABEL DOCUMENT CLASSIFIER TRAINING'))
        self.stdout.write(self.style.SUCCESS('=' * 80))
        
        # Get data file path
        data_file = options['data_file']
        if data_file is None:
            # Default to parent directory
            data_file = settings.CLASSIFIER_DATA
        
        if not os.path.exists(data_file):
            self.stdout.write(self.style.ERROR(f'\nError: Data file not found: {data_file}'))
            self.stdout.write(self.style.WARNING('\nPlease specify the correct path using --data-file'))
            return
        
        self.stdout.write(f'\nData file: {data_file}')
        self.stdout.write(f'Threshold: {options["threshold"]}')
        self.stdout.write(f'Test size: {options["test_size"]}\n')
        
        # Initialize classifier
        classifier = DocumentClassifier(threshold=options['threshold'])
        
        # Train
        try:
            metrics = classifier.train(
                json_path=data_file,
                test_size=options['test_size'],
                random_state=42
            )
            
            # Display results
            self.stdout.write(self.style.SUCCESS('\n' + '=' * 80))
            self.stdout.write(self.style.SUCCESS(' TRAINING RESULTS'))
            self.stdout.write(self.style.SUCCESS('=' * 80))
            
            self.stdout.write(f'\nSubset Accuracy: {metrics["subset_accuracy"]:.4f}')
            self.stdout.write(f'Hamming Loss: {metrics["hamming_loss"]:.4f}')
            
            self.stdout.write('\n--- Micro-averaged Metrics ---')
            self.stdout.write(f'Precision: {metrics["precision_micro"]:.4f}')
            self.stdout.write(f'Recall:    {metrics["recall_micro"]:.4f}')
            self.stdout.write(f'F1 Score:  {metrics["f1_micro"]:.4f}')
            
            self.stdout.write('\n--- Macro-averaged Metrics ---')
            self.stdout.write(f'Precision: {metrics["precision_macro"]:.4f}')
            self.stdout.write(f'Recall:    {metrics["recall_macro"]:.4f}')
            self.stdout.write(f'F1 Score:  {metrics["f1_macro"]:.4f}')
            
            self.stdout.write('\n--- Weighted-averaged Metrics ---')
            self.stdout.write(f'Precision: {metrics["precision_weighted"]:.4f}')
            self.stdout.write(f'Recall:    {metrics["recall_weighted"]:.4f}')
            self.stdout.write(f'F1 Score:  {metrics["f1_weighted"]:.4f}')
            
            self.stdout.write('\n--- Samples-averaged Metrics ---')
            self.stdout.write(f'Precision: {metrics["precision_samples"]:.4f}')
            self.stdout.write(f'Recall:    {metrics["recall_samples"]:.4f}')
            self.stdout.write(f'F1 Score:  {metrics["f1_samples"]:.4f}')
            
            self.stdout.write('\n--- Per-Label Performance ---')
            for label, label_metrics in metrics['per_label'].items():
                self.stdout.write(f'\n{label}:')
                self.stdout.write(f'  Precision: {label_metrics["precision"]:.4f}')
                self.stdout.write(f'  Recall:    {label_metrics["recall"]:.4f}')
                self.stdout.write(f'  F1 Score:  {label_metrics["f1"]:.4f}')
                self.stdout.write(f'  Support:   {label_metrics["support"]}')
            
            # Save model
            model_path = os.path.join(settings.DATA_DIR, 'classifier_model.pkl')
            metrics_path = os.path.join(settings.BASE_DIR, 'search_engine', 'data', 'metrics', 'model_metrics.json')
            
            self.stdout.write(self.style.SUCCESS('\n' + '=' * 80))
            self.stdout.write(self.style.SUCCESS(' SAVING MODEL'))
            self.stdout.write(self.style.SUCCESS('=' * 80))
            
            classifier.save_model(model_path, metrics_path, metrics)
            
            self.stdout.write(self.style.SUCCESS('\n' + '=' * 80))
            self.stdout.write(self.style.SUCCESS(' TRAINING COMPLETED SUCCESSFULLY'))
            self.stdout.write(self.style.SUCCESS('=' * 80))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'\nError during training: {str(e)}'))
            import traceback
            traceback.print_exc()
