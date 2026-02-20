from django.shortcuts import render
from django.conf import settings
import os
import json

def classifier_home(request):
    """Landing page for the classifier app"""
    return render(request, 'classifier/home.html')

def doc_management(request):
    """Placeholder for document management CRUD"""
    json_path = settings.CLASSIFIER_DATA
    data = []
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
    return render(request, 'classifier/management.html', {'docs': data[:20]})

def model_comparison(request):
    """View comparing current model with alternatives"""
    return render(request, 'classifier/comparison.html')

def about_view(request):
    """Project about page"""
    return render(request, 'classifier/about.html')
