import os
from django.core.management.base import BaseCommand
from django.conf import settings
from search_engine.models import Publication, CrawlerLog
from search_engine.utils.crawler import BFSCrawler
from search_engine.utils.search_engine import search_index
from django.utils import timezone

class Command(BaseCommand):
    help = 'Run the BFS crawler to fetch publications'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting crawl...'))
        
        # Create log entry
        log = CrawlerLog.objects.create(
            status='running',
            start_time=timezone.now()
        )
        
        def log_callback(msg):
            self.stdout.write(msg)
        
        try:
            crawler = BFSCrawler(callback=log_callback)
            start_url = "https://pureportal.coventry.ac.uk/en/organisations/ics-research-centre-for-computational-science-and-mathematical-mo"
            
            publications = crawler.crawl_bfs_with_pagination(start_url, max_profiles=5, max_pubs=20)
            
            # Save to database and index
            saved_count = 0
            for pub_data in publications:
                # Incremental update logic using unique publication_link
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
                
                # Add to search index
                search_index.add_document(pub_data, doc_id=f"pub_{pub.id}")
                saved_count += 1
            
            # Save index
            search_index.save(settings.INDEX_FILE)
            
            # Update log
            log.status = 'completed'
            log.end_time = timezone.now()
            log.documents_added = len(publications)
            log.profiles_crawled = len(crawler.visited_urls)
            log.save()
            
            self.stdout.write(self.style.SUCCESS(f'Successfully crawled {saved_count} publications.'))
            
        except Exception as e:
            log.status = 'failed'
            log.end_time = timezone.now()
            log.error_message = str(e)
            log.save()
            self.stdout.write(self.style.ERROR(f'Crawl failed: {str(e)}'))
