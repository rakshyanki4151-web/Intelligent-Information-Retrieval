import json
import os
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from search_engine.ml.classifier import DocumentClassifier
from search_engine.utils.visualizer import MetricsVisualizer
from search_engine.utils.report_generator import ReportGenerator

# Global classifier instance (or shared with other views)
_classifier = None

def get_classifier():
    global _classifier
    model_path = os.path.join(settings.DATA_DIR, 'classifier_model.pkl')
    
    if _classifier is None:
        _classifier = DocumentClassifier()
        if os.path.exists(model_path):
            _classifier.load_model(model_path)
    
    # Safety check: if model exists but instance isn't loaded (labels are missing), force a reload
    if os.path.exists(model_path) and (_classifier.label_names is None or not _classifier.label_names):
        try:
            _classifier.load_model(model_path)
        except Exception as e:
            print(f"Error reloading model: {e}")
            
    return _classifier

def metrics_dashboard(request):
    """Render the comprehensive metrics dashboard."""
    metrics_path = os.path.join(settings.METRICS_DIR, 'model_metrics.json')
    
    if not os.path.exists(metrics_path):
        return render(request, 'classifier/metrics.html', {'error': 'Metrics data not found. Please train the model first.'})
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    classifier = get_classifier()
    
    # Pre-calculate overall stats
    overall = {
        'accuracy': metrics.get('subset_accuracy', 0) * 100,
        'f1_micro': metrics.get('f1_micro', 0) * 100,
        'hamming_loss': metrics.get('hamming_loss', 0),
        'threshold': metrics.get('threshold', 0.3) * 100
    }
    
    # Prepare per-label data
    per_label_data = []
    for label, scores in metrics['per_label'].items():
        per_label_data.append({
            'label': label,
            'precision': round(scores['precision'], 4),
            'recall': round(scores['recall'], 4),
            'f1': round(scores['f1'], 4),
            'support': scores['support']
        })
    per_label_data.sort(key=lambda x: x['f1'], reverse=True)
    
    # Prepare chart data
    vis = MetricsVisualizer()
    heatmaps = vis.get_confusion_matrix_data(metrics)
    feature_importance = vis.get_feature_importance_data(classifier)
    
    # Context
    context = {
        'metrics': metrics,
        'overall': overall,
        'heatmaps': heatmaps,
        'feature_importance': feature_importance,
        'per_label_data': per_label_data
    }
    
    return render(request, 'classifier/metrics.html', context)

def export_metrics_csv(request):
    """Export per-label metrics to CSV."""
    metrics_path = os.path.join(settings.METRICS_DIR, 'model_metrics.json')
    if not os.path.exists(metrics_path):
        return HttpResponse("Metrics not found", status=404)
        
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    csv_data = ReportGenerator.generate_metrics_csv(metrics)
    response = HttpResponse(csv_data, content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="classifier_metrics.csv"'
    return response

def export_metrics_json(request):
    """Export full metrics to JSON."""
    metrics_path = os.path.join(settings.METRICS_DIR, 'model_metrics.json')
    if not os.path.exists(metrics_path):
        return HttpResponse("Metrics not found", status=404)
        
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    response = HttpResponse(json.dumps(metrics, indent=2), content_type='application/json')
    response['Content-Disposition'] = 'attachment; filename="classifier_metrics.json"'
    return response
