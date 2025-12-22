"""Script to run web scrapers for French political parties."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from loguru import logger
from src.scraping.party_scrapers import get_scraper, get_all_scrapers
from src.database.repository import DatabaseRepository, PartyRepository, DocumentRepository, ScrapingLogRepository


def test_scraper(party_code: str):
    """Test scraper for a specific party."""
    logger.info(f"Testing scraper for party: {party_code}")

    try:
        scraper = get_scraper(party_code)
        success = scraper.test_scraper(num_articles=3)

        if success:
            logger.success(f"Scraper test successful for {party_code}")
            return True
        else:
            logger.error(f"Scraper test failed for {party_code}")
            return False

    except Exception as e:
        logger.error(f"Error testing scraper for {party_code}: {e}")
        return False


def scrape_party(party_code: str, days_back: int = 30, save_to_db: bool = True):
    """
    Scrape documents for a specific party.

    Args:
        party_code: Party code (e.g., 'LREM')
        days_back: Number of days to look back
        save_to_db: Whether to save to database
    """
    logger.info(f"Scraping party: {party_code} (last {days_back} days)")

    try:
        # Initialize repositories
        db = DatabaseRepository()
        party_repo = PartyRepository(db)
        doc_repo = DocumentRepository(db)
        log_repo = ScrapingLogRepository(db)

        # Get party from database
        party = party_repo.get_party_by_code(party_code)
        if not party:
            logger.error(f"Party {party_code} not found in database")
            return

        # Get scraper
        scraper = get_scraper(party_code)

        # Scrape documents
        start_time = datetime.utcnow()
        articles = scraper.scrape_recent(days_back=days_back)
        duration = (datetime.utcnow() - start_time).total_seconds()

        logger.info(f"Scraped {len(articles)} articles for {party_code}")

        if not save_to_db:
            logger.info("Skipping database save (save_to_db=False)")
            return

        # Save to database
        documents_new = 0
        documents_updated = 0

        for article in articles:
            try:
                # Check if document already exists
                existing = doc_repo.get_document_by_url(article['url'])

                if existing:
                    logger.debug(f"Document already exists: {article['url']}")
                    documents_updated += 1
                    continue

                # Parse date
                date_published = None
                if article.get('date'):
                    try:
                        date_published = datetime.fromisoformat(article['date'].replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        logger.warning(f"Could not parse date: {article.get('date')}")

                # Create document
                doc_data = {
                    'party_id': party.id,
                    'url': article['url'],
                    'title': article.get('title', '')[:500],  # Limit title length
                    'date_published': date_published,
                    'content': article['content'],
                    'word_count': article.get('word_count', 0),
                    'document_type': 'article'  # Default type
                }

                doc_repo.create_document(doc_data)
                documents_new += 1
                logger.success(f"Saved new document: {article['url']}")

            except Exception as e:
                logger.error(f"Error saving document {article.get('url')}: {e}")

        # Log scraping operation
        log_data = {
            'party_id': party.id,
            'status': 'success' if len(articles) > 0 else 'failed',
            'documents_found': len(articles),
            'documents_new': documents_new,
            'documents_updated': documents_updated,
            'duration_seconds': duration
        }
        log_repo.create_log(log_data)

        logger.success(f"Scraping complete for {party_code}: {documents_new} new, {documents_updated} existing")

    except Exception as e:
        logger.error(f"Error scraping party {party_code}: {e}")

        # Log failure
        try:
            party = party_repo.get_party_by_code(party_code)
            if party:
                log_data = {
                    'party_id': party.id,
                    'status': 'failed',
                    'documents_found': 0,
                    'documents_new': 0,
                    'error_message': str(e)
                }
                log_repo.create_log(log_data)
        except:
            pass


def scrape_all_parties(days_back: int = 30):
    """Scrape all parties."""
    logger.info(f"Scraping all parties (last {days_back} days)")

    scrapers = get_all_scrapers()

    for scraper in scrapers:
        try:
            scrape_party(scraper.party_code, days_back=days_back, save_to_db=True)
        except Exception as e:
            logger.error(f"Error scraping {scraper.party_code}: {e}")

    logger.success("Finished scraping all parties")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run web scrapers for French political parties")

    parser.add_argument(
        '--party',
        type=str,
        help='Party code to scrape (LREM, RN, LFI, PS, EELV, LR)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Scrape all parties'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Test the scraper without saving to database'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to look back (default: 30)'
    )

    args = parser.parse_args()

    # Configure logging
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

    if args.test and args.party:
        test_scraper(args.party)

    elif args.all:
        scrape_all_parties(days_back=args.days)

    elif args.party:
        scrape_party(args.party, days_back=args.days, save_to_db=True)

    else:
        parser.print_help()
        logger.error("Please specify --party or --all")


if __name__ == "__main__":
    main()
