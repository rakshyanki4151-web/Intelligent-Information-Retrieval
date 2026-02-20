from django.http import HttpResponse, FileResponse, Http404
from django.shortcuts import render
from django.conf import settings
import os
import tempfile
from ..utils.report_generator import ReportGenerator
from ..utils.export_manager import ExportManager
from ..utils.data_analyzer import DataAnalyzer
from ..ml.classifier import DocumentClassifier

def export_center_view(request):
    """View for the Exports dashboard."""
    return render(request, 'classifier/exports.html')

def download_pdf_report(request):
    """Generate and download the full PDF report."""
    # Build data for the report
    json_path = settings.CLASSIFIER_DATA
    classifier = DocumentClassifier()
    metrics = classifier.train(json_path)
    
    analyzer = DataAnalyzer(json_path)
    stats = analyzer.get_basic_stats()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        report_gen = ReportGenerator(tmp.name)
        report_gen.generate_full_report(metrics, stats)
        
        response = FileResponse(open(tmp.name, 'rb'), content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="classification_report_2025.pdf"'
        return response

def download_dataset_csv(request):
    """Export dataset to CSV and download."""
    json_path = settings.CLASSIFIER_DATA
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        ExportManager.export_dataset_to_csv(json_path, tmp.name)
        
        response = FileResponse(open(tmp.name, 'rb'), content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="multilabel_dataset.csv"'
        return response

def download_model_bundle(request):
    """Placeholder for model bundle download (ZIP would be better but simple .pkl for now)."""
    # For simplicity, we just return the classifier pkl if it exists
    model_path = os.path.join(settings.DATA_DIR, 'classifier_model.pkl')
    if os.path.exists(model_path):
        return FileResponse(open(model_path, 'rb'), content_type='application/octet-stream')
    else:
        raise Http404("Model file not found. Please train the model first.")
