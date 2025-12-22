"""Script to initialize the database and populate with party metadata."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from config.settings import load_party_config
from src.database.repository import DatabaseRepository, PartyRepository


def setup_database():
    """Initialize database and populate with party data."""
    logger.info("Starting database setup...")

    # Initialize database repository
    db = DatabaseRepository()

    # Create all tables
    logger.info("Creating database tables...")
    db.create_tables()
    logger.success("Database tables created successfully")

    # Initialize party repository
    party_repo = PartyRepository(db)

    # Load party configuration
    logger.info("Loading party configuration from party_config.yaml...")
    parties_config = load_party_config()

    # Insert party data
    logger.info(f"Inserting {len(parties_config)} parties into database...")
    for party_data in parties_config:
        # Check if party already exists
        existing_party = party_repo.get_party_by_code(party_data['code'])

        if existing_party:
            logger.info(f"Party {party_data['code']} already exists, skipping...")
            continue

        # Create new party
        try:
            party = party_repo.create_party(party_data)
            logger.success(f"Created party: {party.code} - {party.name}")
        except Exception as e:
            logger.error(f"Failed to create party {party_data['code']}: {e}")

    # Verify parties were created
    all_parties = party_repo.get_all_parties()
    logger.success(f"Database setup complete! Total parties: {len(all_parties)}")

    # Display party summary
    logger.info("\nParty Summary:")
    logger.info("-" * 80)
    for party in all_parties:
        logger.info(f"  {party.code:6} | {party.name:30} | {party.political_position:15}")
    logger.info("-" * 80)


def reset_database():
    """Drop all tables and recreate (WARNING: This deletes all data!)."""
    logger.warning("RESETTING DATABASE - ALL DATA WILL BE LOST!")

    response = input("Are you sure you want to reset the database? Type 'yes' to confirm: ")

    if response.lower() != 'yes':
        logger.info("Database reset cancelled")
        return

    db = DatabaseRepository()

    logger.info("Dropping all tables...")
    db.drop_tables()
    logger.success("All tables dropped")

    # Now run normal setup
    setup_database()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Setup database for French Political Affect Analysis")
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset database (WARNING: Deletes all data!)'
    )

    args = parser.parse_args()

    if args.reset:
        reset_database()
    else:
        setup_database()
