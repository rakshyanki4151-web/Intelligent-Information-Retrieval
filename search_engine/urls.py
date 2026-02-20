from django.urls import path
from . import views
from .views.classifier_view import classify_view, classify_api
from .views.metrics_view import metrics_dashboard, export_metrics_csv, export_metrics_json
from .views.management_view import classifier_home, doc_management, model_comparison, about_view
from .views.export_view import export_center_view, download_pdf_report, download_dataset_csv, download_model_bundle
from .views.dataset_view import dataset_stats_view

app_name = 'search_engine'

urlpatterns = [
    path('', views.index, name='index'),
    path('search/', views.search_results, name='search'),
    path('crawler/', views.crawler_page, name='crawler'),
    path('api/run-crawler/', views.run_crawler, name='run_crawler'),
    path('api/crawler-status/', views.crawler_status_api, name='crawler_status'),
    path('stats/', views.stats, name='stats'),
    
    # Unified Classifier Portal
    path('classifier/', classifier_home, name='classifier_home'),
    path('classifier/predict/', classify_view, name='classify'),
    path('classifier/api/predict/', classify_api, name='classify_api'),
    path('classifier/metrics/', metrics_dashboard, name='metrics'),
    path('classifier/dataset/', dataset_stats_view, name='dataset_stats'),
    path('classifier/robustness/', views.robustness_view.robustness_dashboard, name='robustness'),
    path('classifier/robustness/run/', views.robustness_view.run_robustness_api, name='run_robustness_api'),
    path('classifier/management/', doc_management, name='doc_management'),
    path('classifier/comparison/', model_comparison, name='model_comparison'),
    path('classifier/exports/', export_center_view, name='export_center'),
    path('classifier/about/', about_view, name='about'),
    
    # Export Controllers
    path('export/pdf/', download_pdf_report, name='download_pdf'),
    path('export/csv-data/', download_dataset_csv, name='download_csv'),
    path('export/model/', download_model_bundle, name='download_model'),
    
    
    # Legacy / Internal
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('export/csv/', export_metrics_csv, name='export_csv_metrics'),
    path('export/json/', export_metrics_json, name='export_json_metrics'),
]


