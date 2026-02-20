"""
Text Preprocessor for Document Classification & Search Engine consistency.
Standard preprocessor for all text cleaning.
"""
import re
import string
import nltk
from typing import List, Dict, Optional
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
for res in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        if res == 'punkt':
            nltk.data.find('tokenizers/punkt')
        elif res == 'punkt_tab':
            nltk.data.find('tokenizers/punkt_tab')
        else:
            nltk.data.find(f'corpora/{res}')
    except LookupError:
        nltk.download(res, quiet=True)

class TextPreprocessor:
    """
    Unified preprocessor for consistency.
    Configuration: lowercase, remove_urls, remove_emails, remove_stopwords, lemmatization.
    """
    
    DOMAIN_STOPWORDS = {
        'said', 'says', 'according', 'report', 'reports', 'news', 'article',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
    }
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_stopwords: bool = True,
        use_lemmatization: bool = True
    ):
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_stopwords = remove_stopwords
        self.use_lemmatization = use_lemmatization
        
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.stop_words = set(stopwords.words('english')).union(self.DOMAIN_STOPWORDS)
        self.stopwords = self.stop_words  # Alias for pickling compatibility
        
        # Regex patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    def process_text(self, text: str) -> List[str]:
        """The unified cleaning pipeline used across the entire project."""
        if not text or not isinstance(text, str):
            return []
            
        # 1. Lowercase
        if self.lowercase:
            text = text.lower()
            
        # 2. URL and Email Removal
        if self.remove_urls:
            text = self.url_pattern.sub('[URL]', text)
        if self.remove_emails:
            text = self.email_pattern.sub('[EMAIL]', text)
            
        # 3. Special Characters
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        
        # 4. Tokenization
        tokens = word_tokenize(text)
        
        # 5. Stopwords and Lemmatization
        processed = []
        for token in tokens:
            if self.remove_stopwords and token in self.stop_words:
                continue
            if len(token) < 2:
                continue
            
            if self.use_lemmatization and self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)
            
            processed.append(token)
            
        return processed

    def get_preprocessing_steps(self, text: str) -> List[Dict[str, str]]:
        """Provides transparency for Task 3: Shows exactly how tokens were derived."""
        steps = [{'step': 'Original Text', 'result': text}]
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
            steps.append({'step': 'Lowercase Conversion', 'result': text})
            
        # Entities
        if self.remove_urls or self.remove_emails:
            text = self.url_pattern.sub('[URL]', text)
            text = self.email_pattern.sub('[EMAIL]', text)
            steps.append({'step': 'Entity Masking', 'result': text})
            
        # Tokenization
        clean_text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        tokens = word_tokenize(clean_text)
        steps.append({'step': 'Tokenization', 'result': " | ".join(tokens)})
        
        # Final
        final = self.process_text(text)
        steps.append({'step': 'Lemmatization & Stopword Removal', 'result': " | ".join(final)})
        
        return steps

    def get_config(self) -> Dict:
        return {
            'lowercase': self.lowercase,
            'remove_urls': self.remove_urls,
            'remove_emails': self.remove_emails,
            'remove_stopwords': self.remove_stopwords,
            'use_lemmatization': self.use_lemmatization
        }
