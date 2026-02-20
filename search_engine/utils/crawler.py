"""
Requests-based BFS Crawler with Pagination for Coventry University Publications
"""
import re
import time
import requests
import urllib.robotparser
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

class BFSCrawler:
    """
    BFS Crawler:
    - Level 1: Discover author profiles
    - Level 2: Visit profiles + handle pagination
    - robots.txt compliance
    - 2-second delay
    """
    
    def __init__(self, callback=None):
        self.callback = callback
        self.visited_urls = set()
        self.publications = []
        self.base_domain = 'pureportal.coventry.ac.uk'
        self.robot_parser = urllib.robotparser.RobotFileParser()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def log(self, msg):
        """Log a message with Unicode safety for Windows terminals."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        try:
            log_msg = f"[{timestamp}] {msg}"
            if self.callback:
                self.callback(log_msg)
            print(log_msg)
        except UnicodeEncodeError:
            # Fallback for Windows terminals failing on special hyphens/chars
            clean_msg = str(msg).encode('ascii', 'ignore').decode('ascii')
            log_msg = f"[{timestamp}] {clean_msg}"
            if self.callback:
                self.callback(log_msg)
            print(log_msg)
    
    def check_robots_txt(self, base_url):
        """Check if crawling is allowed by robots.txt"""
        try:
            parsed = urlparse(base_url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            self.log(f"Checking robots.txt: {robots_url}")
            
            # Use requests to get robots.txt content as robotparser's read() can fail on SSL/headers
            response = self.session.get(robots_url, timeout=10)
            if response.status_code == 200:
                self.robot_parser.parse(response.text.splitlines())
                can_fetch = self.robot_parser.can_fetch("*", base_url)
                self.log(f"{'OK' if can_fetch else 'ERR'} Robots.txt: {'Allowed' if can_fetch else 'Disallowed'}")
                return can_fetch
            else:
                self.log(f"WARN: robots.txt not found (Status {response.status_code}), assuming allowed")
                return True
        except Exception as e:
            self.log(f"WARN: robots.txt check failed: {str(e)}, assuming allowed")
            return True
    
    def get_page(self, url):
        """Fetch a page using requests"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            self.log(f"ERR: Failed to fetch {url}: {str(e)}")
            return None

    def crawl_bfs_with_pagination(self, start_url, max_profiles=10, max_pubs=30):
        """
        3-LEVEL BFS CRAWLER:
        - Level 1: Discover Author profiles
        - Level 2: Visit profiles + handle .nextLink pagination
        - Level 3: Deep-scrape individual publications (Abstract/Tags)
        - Incremental: Skip if exists in DB
        - Ethics: robots.txt + 2s delay
        """
        if not self.check_robots_txt(start_url):
            self.log("ERR: Crawling blocked by robots.txt")
            return []
        
        self.log("="*60)
        self.log("STARTING ETHICAL BFS CRAWLER (LEVEL 1-3)")
        self.log("="*60)
        
        try:
            # LEVEL 1: Discover profiles
            self.log("\n[LEVEL 1] Discovering author profiles...")
            soup = self.get_page(start_url)
            if not soup:
                return []
                
            profile_links = self._extract_profile_links(soup, start_url)
            profile_queue = profile_links[:max_profiles]
            
            self.log(f"OK: Found {len(profile_queue)} profiles")
            
            # Imports for incremental check
            from search_engine.models import Publication
            
            # LEVEL 2: Process with PAGINATION
            for idx, profile_url in enumerate(profile_queue, 1):
                if profile_url in self.visited_urls:
                    continue
                
                self.log(f"\n[{idx}/{len(profile_queue)}] AUTHOR: {profile_url}")
                self.visited_urls.add(profile_url)
                
                time.sleep(2)  # Ethical delay
                
                soup = self.get_page(profile_url)
                if not soup:
                    continue
                    
                author_name = self._extract_author_name(soup)
                
                # PAGINATION LOOP
                page_num = 1
                profile_pubs_metadata = []
                current_soup = soup
                
                while True:
                    self.log(f"  Page {page_num}")
                    pubs = self._extract_publications_metadata(current_soup, profile_url, author_name)
                    profile_pubs_metadata.extend(pubs)
                    
                    if len(profile_pubs_metadata) >= max_pubs:
                        break
                        
                    # CHECK FOR .nextLink
                    next_link = current_soup.select_one('.nextLink')
                    if next_link and next_link.get('href'):
                        next_url = urljoin(profile_url, next_link['href'])
                        self.log(f"  -> Next page: {next_url}")
                        
                        time.sleep(2)
                        current_soup = self.get_page(next_url)
                        if not current_soup:
                            break
                        page_num += 1
                    else:
                        break
                
                # LEVEL 3: Deep Scrape for each publication
                for pub_meta in profile_pubs_metadata:
                    pub_url = pub_meta['publication_link']
                    
                    # Incremental Logic: Preserve server resources
                    if Publication.objects.filter(publication_link=pub_url).exists():
                        self.log(f"    [SKIP] Already indexed: {pub_meta['title'][:50]}...")
                        # We still add it to self.publications if we want to return the full set, 
                        # but we don't fetch it again.
                        continue

                    self.log(f"    [LEVEL 3] Scraping: {pub_meta['title'][:50]}...")
                    time.sleep(2) # Strict 2s delay
                    
                    details = self._scrape_publication_details(pub_url)
                    pub_meta.update(details)
                    self.publications.append(pub_meta)
                    
                    if len(self.publications) >= (max_profiles * max_pubs):
                        break

            self.log(f"\n{'=' * 60}")
            self.log(f"OK: Crawl complete! {len(self.publications)} NEW publications")
            self.log(f"{'=' * 60}")
            
            return self.publications
        
        except Exception as e:
            self.log(f"CRITICAL ERR: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return self.publications
    
    def _extract_profile_links(self, soup, base_url):
        """Extract profile links from the page"""
        profile_links = set()
        patterns = [r'/en/persons/[\w-]+', r'/persons/[\w-]+']
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            for pattern in patterns:
                if re.search(pattern, href):
                    full_url = urljoin(base_url, href)
                    if self.base_domain in full_url:
                        profile_links.add(full_url)
        return sorted(list(profile_links))
    
    def _extract_author_name(self, soup):
        header = soup.select_one('.header h1, .header h2, h1')
        if header:
            name = header.get_text(strip=True)
            if name: return name
        return 'Unknown Author'
    
    def _extract_publications_metadata(self, soup, profile_url, author_name):
        """Extract basic info from listing page"""
        publications = []
        # Updated selectors for modern Pure Portal structure
        containers = soup.select('.list-results .result-container, .list-results .result-item, .rendering_researchoutput, article.publication')
        
        for container in containers:
            try:
                # Title & Link
                title_elem = container.select_one('.title, .title a')
                if not title_elem: continue
                title = title_elem.get_text(strip=True)
                
                link_elem = container.select_one('a.link, .title a')
                pub_link = urljoin(profile_url, link_elem['href']) if link_elem else profile_url
                
                # Year
                year = 'N/A'
                year_elem = container.select_one('.date, .year')
                if year_elem:
                    year_match = re.search(r'(19|20)\d{2}', year_elem.get_text())
                    if year_match: year = year_match.group()
                
                # Authors
                authors = [author_name]
                authors_elem = container.select_one('.authors')
                if authors_elem:
                    authors_text = authors_elem.get_text(strip=True)
                    authors = [a.strip() for a in re.split(r'[,&;]', authors_text) if a.strip()]
                
                publications.append({
                    'title': title,
                    'authors': authors,
                    'year': year,
                    'publication_link': pub_link,
                    'profile_link': profile_url,
                })
            except:
                continue
        return publications

    def _scrape_publication_details(self, url):
        """Level 3: Exact selector scraping"""
        details = {'abstract': '', 'keywords': []}
        soup = self.get_page(url)
        if not soup:
            return details
            
        # Abstract Selector: Multiple fallbacks for robustness
        abstract_elem = soup.select_one('.rendering_researchoutput_abstractportal .textblock, .abstract .textblock, .rendering_researchoutput .textblock')
        if abstract_elem:
            details['abstract'] = abstract_elem.get_text(strip=True)
            
        # Keywords Section (Task 3 Symmetry)
        # Look for headers containing "Keywords"
        kw_section = soup.find(['h2', 'h3'], string=re.compile(r'Keywords', re.I))
        if kw_section:
            next_ul = kw_section.find_next('ul')
            if next_ul:
                details['keywords'] = [li.get_text(strip=True) for li in next_ul.find_all('li')]
        
        # Fingerprint Tags (Additional metadata)
        tags = soup.select('.fingerprint-tag')
        for tag in tags:
            tag_text = tag.get_text(strip=True)
            if tag_text and tag_text not in details['keywords']:
                details['keywords'].append(tag_text)
        
        return details

# Global crawler instance
crawler = BFSCrawler()

