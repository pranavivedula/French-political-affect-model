"""Party-specific scraper implementations."""

from typing import List
from bs4 import BeautifulSoup
from loguru import logger
from urllib.parse import urljoin

from src.scraping.base_scraper import BaseScraper


class LREMScraper(BaseScraper):
    """Scraper for LREM/Renaissance party website."""

    def __init__(self):
        super().__init__(
            party_code='LREM',
            base_url='https://www.renaissance-en-marche.fr',
            news_url='https://www.renaissance-en-marche.fr/actualites'
        )

    def get_article_urls(self, max_pages: int = 10) -> List[str]:
        """
        Get article URLs from LREM news section.

        Note: This implementation may need adjustment based on actual website structure.
        """
        article_urls = []

        try:
            # Fetch the news index page
            html = self.fetch_page(self.news_url)
            if not html:
                logger.warning(f"[{self.party_code}] Could not fetch news page")
                return article_urls

            soup = BeautifulSoup(html, 'html.parser')

            # Strategy 1: Look for article links in common containers
            # Adjust selectors based on actual website structure
            link_selectors = [
                'article a[href]',  # Articles with links
                '.article-item a[href]',  # Common class name
                '.news-item a[href]',
                'a.post-link[href]',
                '.entry-title a[href]'
            ]

            for selector in link_selectors:
                links = soup.select(selector)
                if links:
                    logger.debug(f"[{self.party_code}] Found {len(links)} links using selector: {selector}")
                    break

            # If no links found with selectors, try all links in main content
            if not links:
                main_content = soup.find('main') or soup.find('div', {'class': ['content', 'main', 'articles']})
                if main_content:
                    links = main_content.find_all('a', href=True)

            # Extract and normalize URLs
            for link in links:
                href = link.get('href')
                if not href:
                    continue

                # Create absolute URL
                full_url = urljoin(self.base_url, href)

                # Filter out navigation and non-article links
                if self._is_article_url(full_url):
                    if full_url not in article_urls:
                        article_urls.append(full_url)

            logger.info(f"[{self.party_code}] Found {len(article_urls)} article URLs")

        except Exception as e:
            logger.error(f"[{self.party_code}] Error getting article URLs: {e}")

        return article_urls

    def _is_article_url(self, url: str) -> bool:
        """
        Check if URL is likely an article URL.

        Filters out navigation, social media, etc.
        """
        # Must be from the same domain
        if not url.startswith(self.base_url):
            return False

        # Filter out common non-article patterns
        exclude_patterns = [
            '/contact', '/mentions-legales', '/politique-confidentialite',
            '/a-propos', '/about', '/twitter', '/facebook', '/instagram',
            '/rejoindre', '/don', '/adhesion', '#', 'mailto:', 'tel:',
            '.pdf', '.jpg', '.png', '.gif'
        ]

        url_lower = url.lower()
        for pattern in exclude_patterns:
            if pattern in url_lower:
                return False

        # Include if it looks like an article
        include_patterns = ['/actualites/', '/communique/', '/article/', '/news/', '/post/']
        for pattern in include_patterns:
            if pattern in url_lower:
                return True

        # Default: include if it's a subpage
        return len(url) > len(self.base_url) + 10


class RNScraper(BaseScraper):
    """Scraper for Rassemblement National party website."""

    def __init__(self):
        super().__init__(
            party_code='RN',
            base_url='https://rassemblementnational.fr',
            news_url='https://rassemblementnational.fr/actualites'
        )

    def get_article_urls(self, max_pages: int = 10) -> List[str]:
        """Get article URLs from RN news section."""
        article_urls = []

        try:
            html = self.fetch_page(self.news_url)
            if not html:
                return article_urls

            soup = BeautifulSoup(html, 'html.parser')

            # Adjust selectors based on RN website structure
            links = soup.select('article a[href], .post a[href], .news-item a[href]')

            for link in links:
                href = link.get('href')
                if href:
                    full_url = urljoin(self.base_url, href)
                    if self._is_article_url(full_url) and full_url not in article_urls:
                        article_urls.append(full_url)

            logger.info(f"[{self.party_code}] Found {len(article_urls)} article URLs")

        except Exception as e:
            logger.error(f"[{self.party_code}] Error getting article URLs: {e}")

        return article_urls

    def _is_article_url(self, url: str) -> bool:
        """Check if URL is likely an article URL."""
        if not url.startswith(self.base_url):
            return False

        exclude = ['/contact', '/mentions', '#', 'mailto:', '.pdf', '.jpg']
        return not any(pattern in url.lower() for pattern in exclude)


class LFIScraper(BaseScraper):
    """Scraper for La France Insoumise party website."""

    def __init__(self):
        super().__init__(
            party_code='LFI',
            base_url='https://lafranceinsoumise.fr',
            news_url='https://lafranceinsoumise.fr/actualites'
        )

    def get_article_urls(self, max_pages: int = 10) -> List[str]:
        """Get article URLs from LFI news section."""
        article_urls = []

        try:
            html = self.fetch_page(self.news_url)
            if not html:
                return article_urls

            soup = BeautifulSoup(html, 'html.parser')
            links = soup.select('article a[href], .article-link[href]')

            for link in links:
                href = link.get('href')
                if href:
                    full_url = urljoin(self.base_url, href)
                    if self._is_article_url(full_url) and full_url not in article_urls:
                        article_urls.append(full_url)

            logger.info(f"[{self.party_code}] Found {len(article_urls)} article URLs")

        except Exception as e:
            logger.error(f"[{self.party_code}] Error getting article URLs: {e}")

        return article_urls

    def _is_article_url(self, url: str) -> bool:
        """Check if URL is likely an article URL."""
        if not url.startswith(self.base_url):
            return False
        exclude = ['/contact', '/mentions', '#', 'mailto:', '.pdf']
        return not any(pattern in url.lower() for pattern in exclude)


class PSScraper(BaseScraper):
    """Scraper for Parti Socialiste party website."""

    def __init__(self):
        super().__init__(
            party_code='PS',
            base_url='https://parti-socialiste.fr',
            news_url='https://parti-socialiste.fr/actualites'
        )

    def get_article_urls(self, max_pages: int = 10) -> List[str]:
        """Get article URLs from PS news section."""
        article_urls = []

        try:
            html = self.fetch_page(self.news_url)
            if not html:
                return article_urls

            soup = BeautifulSoup(html, 'html.parser')
            links = soup.select('article a[href], .article a[href]')

            for link in links:
                href = link.get('href')
                if href:
                    full_url = urljoin(self.base_url, href)
                    if self._is_article_url(full_url) and full_url not in article_urls:
                        article_urls.append(full_url)

            logger.info(f"[{self.party_code}] Found {len(article_urls)} article URLs")

        except Exception as e:
            logger.error(f"[{self.party_code}] Error getting article URLs: {e}")

        return article_urls

    def _is_article_url(self, url: str) -> bool:
        """Check if URL is likely an article URL."""
        if not url.startswith(self.base_url):
            return False
        exclude = ['/contact', '/mentions', '#', 'mailto:', '.pdf']
        return not any(pattern in url.lower() for pattern in exclude)


class EELVScraper(BaseScraper):
    """Scraper for Europe Écologie Les Verts party website."""

    def __init__(self):
        super().__init__(
            party_code='EELV',
            base_url='https://www.eelv.fr',
            news_url='https://www.eelv.fr/actualites'
        )

    def get_article_urls(self, max_pages: int = 10) -> List[str]:
        """Get article URLs from EELV news section."""
        article_urls = []

        try:
            html = self.fetch_page(self.news_url)
            if not html:
                return article_urls

            soup = BeautifulSoup(html, 'html.parser')
            links = soup.select('article a[href], .news a[href]')

            for link in links:
                href = link.get('href')
                if href:
                    full_url = urljoin(self.base_url, href)
                    if self._is_article_url(full_url) and full_url not in article_urls:
                        article_urls.append(full_url)

            logger.info(f"[{self.party_code}] Found {len(article_urls)} article URLs")

        except Exception as e:
            logger.error(f"[{self.party_code}] Error getting article URLs: {e}")

        return article_urls

    def _is_article_url(self, url: str) -> bool:
        """Check if URL is likely an article URL."""
        if not url.startswith(self.base_url):
            return False
        exclude = ['/contact', '/mentions', '#', 'mailto:', '.pdf']
        return not any(pattern in url.lower() for pattern in exclude)


class LRScraper(BaseScraper):
    """Scraper for Les Républicains party website."""

    def __init__(self):
        super().__init__(
            party_code='LR',
            base_url='https://www.republicains.fr',
            news_url='https://www.republicains.fr/actualites'
        )

    def get_article_urls(self, max_pages: int = 10) -> List[str]:
        """Get article URLs from LR news section."""
        article_urls = []

        try:
            html = self.fetch_page(self.news_url)
            if not html:
                return article_urls

            soup = BeautifulSoup(html, 'html.parser')
            links = soup.select('article a[href], .article-item a[href]')

            for link in links:
                href = link.get('href')
                if href:
                    full_url = urljoin(self.base_url, href)
                    if self._is_article_url(full_url) and full_url not in article_urls:
                        article_urls.append(full_url)

            logger.info(f"[{self.party_code}] Found {len(article_urls)} article URLs")

        except Exception as e:
            logger.error(f"[{self.party_code}] Error getting article URLs: {e}")

        return article_urls

    def _is_article_url(self, url: str) -> bool:
        """Check if URL is likely an article URL."""
        if not url.startswith(self.base_url):
            return False
        exclude = ['/contact', '/mentions', '#', 'mailto:', '.pdf']
        return not any(pattern in url.lower() for pattern in exclude)


# Factory function to get scraper by party code
def get_scraper(party_code: str) -> BaseScraper:
    """
    Get scraper instance for a party code.

    Args:
        party_code: Party code (e.g., 'LREM', 'RN')

    Returns:
        Scraper instance for the party

    Raises:
        ValueError: If party code is not recognized
    """
    scrapers = {
        'LREM': LREMScraper,
        'RN': RNScraper,
        'LFI': LFIScraper,
        'PS': PSScraper,
        'EELV': EELVScraper,
        'LR': LRScraper
    }

    scraper_class = scrapers.get(party_code)
    if not scraper_class:
        raise ValueError(f"Unknown party code: {party_code}")

    return scraper_class()


def get_all_scrapers() -> List[BaseScraper]:
    """
    Get all party scrapers.

    Returns:
        List of all scraper instances
    """
    return [
        LREMScraper(),
        RNScraper(),
        LFIScraper(),
        PSScraper(),
        EELVScraper(),
        LRScraper()
    ]
