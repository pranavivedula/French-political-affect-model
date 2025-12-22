"""Base scraper class with rate limiting and robots.txt compliance."""

import time
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
from datetime import datetime, timedelta
from loguru import logger
import trafilatura
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import USER_AGENT, RATE_LIMIT_SECONDS, MAX_RETRIES, REQUEST_TIMEOUT


class BaseScraper(ABC):
    """Abstract base class for party website scrapers."""

    def __init__(self, party_code: str, base_url: str, news_url: str = None):
        """
        Initialize scraper.

        Args:
            party_code: Party code (e.g., 'LREM', 'RN')
            base_url: Base URL of party website
            news_url: URL of news/articles section
        """
        self.party_code = party_code
        self.base_url = base_url
        self.news_url = news_url or base_url
        self.session = self._create_session()
        self.robot_parser = self._get_robot_parser()
        self.last_request_time = 0

    def _create_session(self) -> requests.Session:
        """Create requests session with proper headers."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        return session

    def _get_robot_parser(self) -> RobotFileParser:
        """Get and parse robots.txt for the website."""
        robot_parser = RobotFileParser()
        robots_url = urljoin(self.base_url, '/robots.txt')

        try:
            robot_parser.set_url(robots_url)
            robot_parser.read()
            logger.info(f"[{self.party_code}] Successfully parsed robots.txt")

            # Check if we can fetch anything
            if not robot_parser.can_fetch(USER_AGENT, self.base_url):
                logger.warning(f"[{self.party_code}] robots.txt disallows scraping with our user agent")

        except Exception as e:
            logger.warning(f"[{self.party_code}] Could not fetch robots.txt: {e}")

        return robot_parser

    def _respect_rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_SECONDS:
            sleep_time = RATE_LIMIT_SECONDS - elapsed
            logger.debug(f"[{self.party_code}] Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        return self.robot_parser.can_fetch(USER_AGENT, url)

    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
    def fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch page content with rate limiting and retries.

        Args:
            url: URL to fetch

        Returns:
            Page HTML content or None if failed
        """
        # Check robots.txt
        if not self.can_fetch(url):
            logger.warning(f"[{self.party_code}] robots.txt disallows fetching: {url}")
            return None

        # Respect rate limit
        self._respect_rate_limit()

        try:
            logger.debug(f"[{self.party_code}] Fetching: {url}")
            response = self.session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            return response.text

        except requests.exceptions.HTTPError as e:
            logger.error(f"[{self.party_code}] HTTP error fetching {url}: {e}")
            raise

        except requests.exceptions.Timeout:
            logger.error(f"[{self.party_code}] Timeout fetching {url}")
            raise

        except requests.exceptions.RequestException as e:
            logger.error(f"[{self.party_code}] Error fetching {url}: {e}")
            raise

    def extract_content(self, html: str, url: str) -> Optional[Dict[str, str]]:
        """
        Extract clean text content from HTML using trafilatura.

        Args:
            html: HTML content
            url: URL of the page (for metadata)

        Returns:
            Dictionary with extracted content or None
        """
        try:
            # Extract main content
            content = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
                url=url
            )

            if not content or len(content.strip()) < 100:
                logger.warning(f"[{self.party_code}] Extracted content too short or empty for: {url}")
                return None

            # Extract metadata
            metadata = trafilatura.extract_metadata(html)

            # Build result
            result = {
                'content': content.strip(),
                'word_count': len(content.split()),
                'url': url
            }

            # Add metadata if available
            if metadata:
                result['title'] = metadata.title or ''
                result['date'] = metadata.date or None

            return result

        except Exception as e:
            logger.error(f"[{self.party_code}] Error extracting content from {url}: {e}")
            return None

    @abstractmethod
    def get_article_urls(self, max_pages: int = 10) -> List[str]:
        """
        Get list of article URLs from the news section.

        This must be implemented by subclasses as each party website
        has different structure.

        Args:
            max_pages: Maximum number of index pages to scrape

        Returns:
            List of article URLs
        """
        pass

    def scrape_article(self, url: str) -> Optional[Dict[str, any]]:
        """
        Scrape a single article.

        Args:
            url: Article URL

        Returns:
            Dictionary with article data or None if failed
        """
        try:
            html = self.fetch_page(url)
            if not html:
                return None

            content_data = self.extract_content(html, url)
            if not content_data:
                return None

            # Add party code and scraping timestamp
            content_data['party_code'] = self.party_code
            content_data['scraped_at'] = datetime.utcnow()

            logger.success(f"[{self.party_code}] Successfully scraped: {url}")
            return content_data

        except Exception as e:
            logger.error(f"[{self.party_code}] Failed to scrape article {url}: {e}")
            return None

    def scrape_recent(self, days_back: int = 30, max_articles: int = 100) -> List[Dict[str, any]]:
        """
        Scrape recent articles from the party website.

        Args:
            days_back: Number of days to look back
            max_articles: Maximum number of articles to scrape

        Returns:
            List of scraped article data
        """
        logger.info(f"[{self.party_code}] Starting scrape for articles from last {days_back} days")

        # Get article URLs
        article_urls = self.get_article_urls(max_pages=5)
        logger.info(f"[{self.party_code}] Found {len(article_urls)} article URLs")

        # Scrape articles
        articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)

        for url in article_urls[:max_articles]:
            article_data = self.scrape_article(url)

            if article_data:
                # Check if article is recent enough (if date available)
                if article_data.get('date'):
                    try:
                        article_date = datetime.fromisoformat(article_data['date'].replace('Z', '+00:00'))
                        if article_date < cutoff_date:
                            logger.debug(f"[{self.party_code}] Article too old, skipping: {url}")
                            continue
                    except (ValueError, AttributeError):
                        # If date parsing fails, include the article anyway
                        pass

                articles.append(article_data)

        logger.success(f"[{self.party_code}] Scraped {len(articles)} articles")
        return articles

    def test_scraper(self, num_articles: int = 3) -> bool:
        """
        Test the scraper by scraping a few articles.

        Args:
            num_articles: Number of articles to test with

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"[{self.party_code}] Testing scraper with {num_articles} articles")

        try:
            articles = self.scrape_recent(days_back=90, max_articles=num_articles)

            if len(articles) > 0:
                logger.success(f"[{self.party_code}] Scraper test successful! Scraped {len(articles)} articles")

                # Print sample
                for i, article in enumerate(articles[:2], 1):
                    logger.info(f"\nSample Article {i}:")
                    logger.info(f"  Title: {article.get('title', 'N/A')}")
                    logger.info(f"  URL: {article.get('url', 'N/A')}")
                    logger.info(f"  Date: {article.get('date', 'N/A')}")
                    logger.info(f"  Word Count: {article.get('word_count', 0)}")
                    logger.info(f"  Content Preview: {article.get('content', '')[:200]}...")

                return True
            else:
                logger.error(f"[{self.party_code}] Scraper test failed - no articles scraped")
                return False

        except Exception as e:
            logger.error(f"[{self.party_code}] Scraper test failed with error: {e}")
            return False
