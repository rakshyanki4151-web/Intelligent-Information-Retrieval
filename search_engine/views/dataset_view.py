from django.shortcuts import render
from django.conf import settings
import os
import json
from ..utils.data_analyzer import DataAnalyzer

def dataset_stats_view(request):
    """
    View for displaying dataset statistics and quality metrics.
    """
    json_path = settings.CLASSIFIER_DATA
    analyzer = DataAnalyzer(json_path)
    
    # Gather all analysis data
    stats = analyzer.get_basic_stats()
    distribution = analyzer.get_category_distribution()
    cooccurrence = analyzer.get_cooccurrence_matrix()
    quality = analyzer.perform_quality_checks()
    vocab = analyzer.get_vocabulary_stats()
    sources = analyzer.get_source_distribution()
    
    # Prepare data for charts (JSON serialize for template)
    chart_data = {
        'distribution_labels': list(distribution['counts'].keys()),
        'distribution_data': list(distribution['counts'].values()),
        'cooccurrence': cooccurrence,
        'vocab_labels': [item[0] for item in vocab['most_common']],
        'vocab_data': [item[1] for item in vocab['most_common']],
    }
    
    # Handle sample documents with filtering
    category_filter = request.GET.get('category', 'all')
    docs = analyzer.data
    if category_filter != 'all':
        docs = [d for d in docs if category_filter in d.get('labels', [])]
    
    # Minimal pagination logic
    page = int(request.GET.get('page', 1))
    page_size = 10
    total_pages = (len(docs) + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_docs = docs[start_idx:end_idx]

    context = {
        'stats': stats,
        'distribution': distribution,
        'quality': quality,
        'vocab': vocab,
        'sources': sources,
        'chart_data': chart_data,
        'chart_data_json': json.dumps(chart_data),
        'sample_docs': paginated_docs,
        'current_page': page,
        'total_pages': total_pages,
        'category_filter': category_filter,
        'all_categories': stats.get('unique_labels', [])
    }
    
    return render(request, 'classifier/dataset.html', context)
