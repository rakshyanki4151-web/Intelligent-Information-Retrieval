"""
Django management command to evaluate the multi-label classifier.

Usage:
    python manage.py evaluate_model
"""

from django.core.management.base import BaseCommand
from django.conf import settings
import os
import sys
import json
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from search_engine.ml.classifier import DocumentClassifier


class Command(BaseCommand):
    help = 'Evaluate the multi-label document classifier with detailed metrics'

    def add_arguments(self, parser):
        parser.add_argument(
            '--model-path',
            type=str,
            default=None,
            help='Path to trained model (default: data/models/classifier_model.pkl)'
        )
        parser.add_argument(
            '--save-report',
            action='store_true',
            help='Save detailed evaluation report to JSON'
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('=' * 80))
        self.stdout.write(self.style.SUCCESS(' MULTI-LABEL CLASSIFIER EVALUATION'))
        self.stdout.write(self.style.SUCCESS('=' * 80))
        
        # Get model path
        model_path = options['model_path']
        if model_path is None:
            model_path = os.path.join(
                settings.BASE_DIR, 'search_engine', 'data', 'models', 'classifier_model.pkl'
            )
        
        if not os.path.exists(model_path):
            self.stdout.write(self.style.ERROR(f'\nError: Model not found: {model_path}'))
            self.stdout.write(self.style.WARNING('\nPlease train the model first using: python manage.py train_model'))
            return
        
        # Load metrics
        metrics_path = os.path.join(settings.BASE_DIR, 'search_engine', 'data', 'metrics', 'model_metrics.json')
        
        if not os.path.exists(metrics_path):
            self.stdout.write(self.style.ERROR(f'\nError: Metrics file not found: {metrics_path}'))
            self.stdout.write(self.style.WARNING('\nPlease train the model first to generate metrics'))
            return
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Load classifier for feature importance
        classifier = DocumentClassifier()
        classifier.load_model(model_path)
        
        # Display comprehensive evaluation
        self.stdout.write(self.style.SUCCESS('\n' + '=' * 80))
        self.stdout.write(self.style.SUCCESS(' OVERALL METRICS'))
        self.stdout.write(self.style.SUCCESS('=' * 80))
        
        self.stdout.write(f'\nThreshold: {metrics["threshold"]}')
        self.stdout.write(f'\nSubset Accuracy: {metrics["subset_accuracy"]:.4f}')
        self.stdout.write('  (Percentage of samples with all labels predicted correctly)')
        
        self.stdout.write(f'\nHamming Loss: {metrics["hamming_loss"]:.4f}')
        self.stdout.write('  (Fraction of labels that are incorrectly predicted)')
        
        # Micro-averaged
        self.stdout.write(self.style.SUCCESS('\n' + '-' * 80))
        self.stdout.write(self.style.SUCCESS(' MICRO-AVERAGED METRICS'))
        self.stdout.write(self.style.SUCCESS('-' * 80))
        self.stdout.write('\n(Aggregate contributions of all classes)')
        self.stdout.write(f'\nPrecision: {metrics["precision_micro"]:.4f}')
        self.stdout.write(f'Recall:    {metrics["recall_micro"]:.4f}')
        self.stdout.write(f'F1 Score:  {metrics["f1_micro"]:.4f}')
        
        # Macro-averaged
        self.stdout.write(self.style.SUCCESS('\n' + '-' * 80))
        self.stdout.write(self.style.SUCCESS(' MACRO-AVERAGED METRICS'))
        self.stdout.write(self.style.SUCCESS('-' * 80))
        self.stdout.write('\n(Unweighted mean of per-class metrics)')
        self.stdout.write(f'\nPrecision: {metrics["precision_macro"]:.4f}')
        self.stdout.write(f'Recall:    {metrics["recall_macro"]:.4f}')
        self.stdout.write(f'F1 Score:  {metrics["f1_macro"]:.4f}')
        
        # Weighted-averaged
        self.stdout.write(self.style.SUCCESS('\n' + '-' * 80))
        self.stdout.write(self.style.SUCCESS(' WEIGHTED-AVERAGED METRICS'))
        self.stdout.write(self.style.SUCCESS('-' * 80))
        self.stdout.write('\n(Weighted by support - number of true instances)')
        self.stdout.write(f'\nPrecision: {metrics["precision_weighted"]:.4f}')
        self.stdout.write(f'Recall:    {metrics["recall_weighted"]:.4f}')
        self.stdout.write(f'F1 Score:  {metrics["f1_weighted"]:.4f}')
        
        # Samples-averaged
        self.stdout.write(self.style.SUCCESS('\n' + '-' * 80))
        self.stdout.write(self.style.SUCCESS(' SAMPLES-AVERAGED METRICS'))
        self.stdout.write(self.style.SUCCESS('-' * 80))
        self.stdout.write('\n(Average of metrics computed for each sample)')
        self.stdout.write(f'\nPrecision: {metrics["precision_samples"]:.4f}')
        self.stdout.write(f'Recall:    {metrics["recall_samples"]:.4f}')
        self.stdout.write(f'F1 Score:  {metrics["f1_samples"]:.4f}')
        
        # Per-label analysis
        self.stdout.write(self.style.SUCCESS('\n' + '=' * 80))
        self.stdout.write(self.style.SUCCESS(' PER-LABEL ANALYSIS'))
        self.stdout.write(self.style.SUCCESS('=' * 80))
        
        for label, label_metrics in metrics['per_label'].items():
            self.stdout.write(f'\n{label}:')
            self.stdout.write(f'  Precision: {label_metrics["precision"]:.4f}')
            self.stdout.write(f'  Recall:    {label_metrics["recall"]:.4f}')
            self.stdout.write(f'  F1 Score:  {label_metrics["f1"]:.4f}')
            self.stdout.write(f'  Support:   {label_metrics["support"]} samples')
            
            # Performance assessment
            f1 = label_metrics["f1"]
            if f1 >= 0.8:
                assessment = self.style.SUCCESS('Excellent')
            elif f1 >= 0.6:
                assessment = self.style.WARNING('Good')
            else:
                assessment = self.style.ERROR('Needs Improvement')
            self.stdout.write(f'  Assessment: {assessment}')
        
        # Confusion matrices
        self.stdout.write(self.style.SUCCESS('\n' + '=' * 80))
        self.stdout.write(self.style.SUCCESS(' CONFUSION MATRICES'))
        self.stdout.write(self.style.SUCCESS('=' * 80))
        
        for label, cm in metrics['confusion_matrices'].items():
            self.stdout.write(f'\n{label}:')
            self.stdout.write('  [[TN  FP]')
            self.stdout.write('   [FN  TP]]')
            self.stdout.write(f'  {cm}')
            
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            self.stdout.write(f'  True Negatives:  {tn}')
            self.stdout.write(f'  False Positives: {fp}')
            self.stdout.write(f'  False Negatives: {fn}')
            self.stdout.write(f'  True Positives:  {tp}')
        
        # Feature importance
        self.stdout.write(self.style.SUCCESS('\n' + '=' * 80))
        self.stdout.write(self.style.SUCCESS(' TOP FEATURES PER LABEL'))
        self.stdout.write(self.style.SUCCESS('=' * 80))
        
        for label in classifier.label_names:
            self.stdout.write(f'\n{label}:')
            try:
                top_features = classifier.get_feature_importance(label, n=10)
                for i, (feature, importance) in enumerate(top_features, 1):
                    self.stdout.write(f'  {i:2d}. {feature:20s} ({importance:.4f})')
            except Exception as e:
                self.stdout.write(f'  Error: {str(e)}')
        
        # Save detailed report if requested
        if options['save_report']:
            report_path = os.path.join(
                settings.BASE_DIR, 'search_engine', 'data', 'metrics', 'evaluation_report.json'
            )
            
            # Create comprehensive report
            report = {
                'model_path': model_path,
                'metrics': metrics,
                'feature_importance': {}
            }
            
            for label in classifier.label_names:
                try:
                    top_features = classifier.get_feature_importance(label, n=20)
                    report['feature_importance'][label] = [
                        {'feature': f, 'importance': float(imp)}
                        for f, imp in top_features
                    ]
                except Exception:
                    report['feature_importance'][label] = []
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            self.stdout.write(self.style.SUCCESS(f'\n\nDetailed report saved to: {report_path}'))
        
        # Summary
        self.stdout.write(self.style.SUCCESS('\n' + '=' * 80))
        self.stdout.write(self.style.SUCCESS(' EVALUATION SUMMARY'))
        self.stdout.write(self.style.SUCCESS('=' * 80))
        
        f1_micro = metrics['f1_micro']
        if f1_micro >= 0.8:
            self.stdout.write(self.style.SUCCESS(f'\n[OK] EXCELLENT PERFORMANCE (F1-Micro: {f1_micro:.4f})'))
        elif f1_micro >= 0.6:
            self.stdout.write(self.style.WARNING(f'\n[!] GOOD PERFORMANCE (F1-Micro: {f1_micro:.4f})'))
        else:
            self.stdout.write(self.style.ERROR(f'\n[X] NEEDS IMPROVEMENT (F1-Micro: {f1_micro:.4f})'))
        
        self.stdout.write('\n')
