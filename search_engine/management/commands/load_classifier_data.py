import json
import os
from django.core.management.base import BaseCommand
from django.conf import settings
from search_engine.models import ClassifiedDocument

class Command(BaseCommand):
    help = 'Load classification documents from classification.json'

    def handle(self, *args, **options):
        json_path = os.path.join(settings.DATA_DIR, 'raw', 'classification.json')
        
        if not os.path.exists(json_path):
            self.stdout.write(self.style.ERROR(f'File not found: {json_path}'))
            return

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Clear existing data
        ClassifiedDocument.objects.all().delete()
        
        count = 0
        for category, texts in data.items():
            for text in texts:
                ClassifiedDocument.objects.create(
                    text=text,
                    category=category
                )
                count += 1
        
        self.stdout.write(self.style.SUCCESS(f'Successfully loaded {count} documents.'))
