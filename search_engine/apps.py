from django.apps import AppConfig


class SearchEngineConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'search_engine'
    
    def ready(self):
        """Initialize search engine components when Django starts"""
        import os
        if os.environ.get('RUN_MAIN') == 'true':
            # Only run in main process, not reloader
            from .utils.scheduler import start_scheduler
            start_scheduler()
