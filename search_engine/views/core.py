from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.http import require_http_methods
import threading
import os

from ..models import Publication, CrawlerLog
from ..utils.search_engine import search_index
from ..utils.crawler import BFSCrawler


# Global crawler status
CRAWLER_STATE = {
    'running': False,
    'logs': [],
    'progress': 0,
    'current_log': None
}


def index(request):
    """Homepage with search box"""
    if search_index.get_document_count() == 0:
        _load_index_from_db()
        
    stats = {
        'total_docs': Publication.objects.count(),
        'indexed_docs': search_index.get_document_count()
    }
    return render(request, 'search_engine/index.html', {'stats': stats})


def search_results(request):
    """Search results page"""
    query = request.GET.get('q', '')
    
    if not query:
        return render(request, 'search_engine/search_results.html', {
            'query': '',
            'results': [],
            'total_results': 0
        })
    
    # Load index if not loaded
    if search_index.get_document_count() == 0:
        _load_index_from_db()
    
    # Search using weighted inverted index (no category filter - Task 1 is independent)
    results = search_index.search(query, top_k=50)
    
    return render(request, 'search_engine/search_results.html', {
        'query': query,
        'results': results,
        'total_results': len(results)
    })


def crawler_page(request):
    """Crawler management page"""
    recent_logs = CrawlerLog.objects.all()[:10]
    return render(request, 'search_engine/crawler.html', {
        'status': CRAWLER_STATE,
        'recent_logs': recent_logs
    })


from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
@require_http_methods(["POST"])
def run_crawler(request):
    """API endpoint to run crawler"""
    global CRAWLER_STATE
    
    if CRAWLER_STATE['running']:
        return JsonResponse({
            'status': 'error',
            'message': 'Crawler already running'
        })
    
    CRAWLER_STATE['running'] = True
    CRAWLER_STATE['logs'] = []
    CRAWLER_STATE['progress'] = 0
    
    def crawl_task():
        """Background crawl task"""
        global CRAWLER_STATE
        
        from django.utils import timezone
        
        # Create log entry
        log = CrawlerLog.objects.create(
            status='running',
            start_time=timezone.now()
        )
        CRAWLER_STATE['current_log'] = log.id
        
        def log_callback(msg):
            CRAWLER_STATE['logs'].append(msg)
            if len(CRAWLER_STATE['logs']) > 100:
                CRAWLER_STATE['logs'] = CRAWLER_STATE['logs'][-100:]
        
        try:
            crawler = BFSCrawler(callback=log_callback)
            start_url = "https://pureportal.coventry.ac.uk/en/organisations/ics-research-centre-for-computational-science-and-mathematical-mo"
            
            publications = crawler.crawl_bfs_with_pagination(start_url, max_profiles=10, max_pubs=50)

            # Save to database and index (NO classification - Task 1 is independent)
            saved_count = 0
            for pub_data in publications:
                pub, created = Publication.objects.update_or_create(
                    publication_link=pub_data['publication_link'],
                    defaults={
                        'title': pub_data['title'],
                        'authors': ', '.join(pub_data['authors']) if isinstance(pub_data['authors'], list) else pub_data['authors'],
                        'year': pub_data['year'],
                        'abstract': pub_data.get('abstract', ''),
                        'keywords': ', '.join(pub_data.get('keywords', [])) if isinstance(pub_data.get('keywords', []), list) else '',
                        'profile_link': pub_data['profile_link']
                    }
                )
                
                # Add to search index (defer vector rebuild)
                search_index.add_document(pub_data, doc_id=f"pub_{pub.id}", rebuild=False)
                saved_count += 1
            
            # Rebuild vectors once and save index
            search_index._rebuild_vectors()
            search_index.save(settings.INDEX_FILE)
            
            # Update log
            log.status = 'completed'
            log.end_time = timezone.now()
            log.documents_added = len(publications)
            log.profiles_crawled = len(crawler.visited_urls)
            log.save()
            
            CRAWLER_STATE['logs'].append(f"Saved {saved_count} publications to database")
            
        except Exception as e:
            log.status = 'failed'
            log.end_time = timezone.now()
            log.error_message = str(e)
            log.save()
            
            CRAWLER_STATE['logs'].append(f"Error: {str(e)}")
        
        finally:
            CRAWLER_STATE['running'] = False
            CRAWLER_STATE['progress'] = 100
    
    # Start crawler in background thread
    thread = threading.Thread(target=crawl_task, daemon=True)
    thread.start()
    
    return JsonResponse({
        'status': 'success',
        'message': 'Crawler started'
    })


def crawler_status_api(request):
    """API endpoint for crawler status"""
    return JsonResponse(CRAWLER_STATE)


def stats(request):
    """Statistics dashboard"""
    total_pubs = Publication.objects.count()
    recent_pubs = Publication.objects.order_by('-created_at')[:10]
    recent_logs = CrawlerLog.objects.all()[:5]
    
    # Year distribution
    years = Publication.objects.values_list('year', flat=True)
    year_counts = {}
    for year in years:
        if year != 'N/A':
            year_counts[year] = year_counts.get(year, 0) + 1
    
    return render(request, 'search_engine/stats.html', {
        'total_pubs': total_pubs,
        'indexed_docs': search_index.get_document_count(),
        'recent_pubs': recent_pubs,
        'recent_logs': recent_logs,
        'year_counts': sorted(year_counts.items(), reverse=True)[:10]
    })


from ..ml.classifier import DocumentClassifier

# Global classifier instance
doc_classifier = DocumentClassifier()

# Load model if it exists
if os.path.exists(settings.CLASSIFIER_MODEL):
    try:
        doc_classifier.load_model(settings.CLASSIFIER_MODEL)
    except Exception as e:
        print(f"Error loading classifier: {e}")

def classification_view(request):
    """View for manual text classification"""
    result = None
    if request.method == 'POST':
        text = request.POST.get('text', '')
        if text:
            result = doc_classifier.predict(text)
    
    return render(request, 'search_engine/classifier.html', {'result': result})

def dashboard_view(request):
    """Scientific validation dashboard with Preprocessing Transparency"""
    from django.conf import settings
    data_path = settings.CLASSIFIER_DATA
    metrics_data = doc_classifier.train(data_path)
    
    # Category list for the template
    categories = doc_classifier.label_names
    
    # Prepare metrics summary
    metrics = {
        'accuracy': metrics_data.get('subset_accuracy', 0),
        'f1': metrics_data.get('f1_micro', 0),
        'precision': metrics_data.get('precision_micro', 0),
        'recall': metrics_data.get('recall_micro', 0),
    }
    
    # Get distribution from DB or metrics
    from ..models import ClassifiedDocument
    from django.db.models import Count
    dist_db = ClassifiedDocument.objects.values('category').annotate(count=Count('category'))
    
    # Convert per_label metrics for easy looping
    per_label_list = []
    for cat in categories:
        stats = metrics_data.get('per_label', {}).get(cat, {})
        per_label_list.append({
            'name': cat,
            'precision': stats.get('precision', 0),
            'recall': stats.get('recall', 0),
            'f1': stats.get('f1', 0),
            'support': stats.get('support', 0)
        })
    
    # Task 3: Transformation Transparency Logic
    from ..utils.preprocessor import TextPreprocessor
    tp = TextPreprocessor()
    sample_abstract = "This research article on deep learning and neural networks (available at http://example.com) examines the future of computing and computational models."
    transformation_steps = tp.get_preprocessing_steps(sample_abstract)
    
    return render(request, 'search_engine/dashboard.html', {
        'metrics': metrics,
        'distribution': list(dist_db),
        'per_label_list': per_label_list,
        'transformation_steps': transformation_steps,
        'sample_abstract': sample_abstract
    })

def _load_index_from_db():
    """Load search index from database"""
    # Try loading from file first
    if search_index.load(settings.INDEX_FILE):
        return
    
    # Otherwise, build from database
    publications = Publication.objects.all()
    for pub in publications:
        pub_data = {
            'title': pub.title,
            'authors': pub.authors.split(','),
            'year': pub.year,
            'abstract': pub.abstract,
            'keywords': pub.keywords.split(','),
            'publication_link': pub.publication_link,
            'profile_link': pub.profile_link
        }
        search_index.add_document(pub_data, doc_id=f"pub_{pub.id}", rebuild=False)
    
    # Rebuild and save
    if publications.exists():
        search_index._rebuild_vectors()
        search_index.save(settings.INDEX_FILE)
