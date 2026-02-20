from django.db import models
from django.utils import timezone


class Publication(models.Model):
    """Model for storing publication data"""
    title = models.CharField(max_length=500)
    authors = models.TextField()  # Stored as comma-separated
    year = models.CharField(max_length=10, default='N/A')
    abstract = models.TextField(blank=True)
    keywords = models.TextField(blank=True)  # Stored as comma-separated
    publication_link = models.URLField(max_length=1000, unique=True)
    profile_link = models.URLField(max_length=1000)
    fingerprints = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['title']),
            models.Index(fields=['year']),
        ]
    
    def __str__(self):
        return f"{self.title} ({self.year})"


class ClassifiedDocument(models.Model):
    """Model for Task 2: Document Classification"""
    CATEGORY_CHOICES = [
        ('business', 'Business'),
        ('entertainment', 'Entertainment'),
        ('health', 'Health'),
    ]
    
    text = models.TextField()
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.category}: {self.text[:50]}..."


class CrawlerLog(models.Model):
    """3. Scientific Logging (Task 3 Validation)"""
    STATUS_CHOICES = [
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='running')
    start_time = models.DateTimeField(default=timezone.now)
    end_time = models.DateTimeField(null=True, blank=True)
    documents_added = models.IntegerField(default=0)  # Re-labeled as requested
    profiles_crawled = models.IntegerField(default=0)
    error_message = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-start_time']
    
    def __str__(self):
        return f"Crawl {self.start_time.strftime('%Y-%m-%d %H:%M')} - {self.status}"
    
    def duration(self):
        """Calculate duration of crawl"""
        if self.end_time:
            delta = self.end_time - self.start_time
            return f"{delta.total_seconds():.0f} seconds"
        return "In progress"
