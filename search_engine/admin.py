from django.contrib import admin
from .models import Publication, CrawlerLog


@admin.register(Publication)
class PublicationAdmin(admin.ModelAdmin):
    list_display = ['title', 'year', 'created_at']
    list_filter = ['year', 'created_at']
    search_fields = ['title', 'authors', 'abstract']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(CrawlerLog)
class CrawlerLogAdmin(admin.ModelAdmin):
    list_display = ['start_time', 'status', 'documents_added', 'profiles_crawled', 'duration']
    list_filter = ['status', 'start_time']
    readonly_fields = ['start_time', 'end_time']
