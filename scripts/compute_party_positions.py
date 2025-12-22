"""Script to compute party positions on the Circumplex Model."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from loguru import logger
from tabulate import tabulate

from src.database.repository import (
    DatabaseRepository,
    PartyRepository,
    DocumentRepository,
    TemporalSnapshotRepository
)
from src.analysis.aggregator import PartyAffectAggregator
from src.analysis.circumplex_mapper import CircumplexMapper
from config.settings import load_party_config


def compute_party_position(
    party_code: str,
    start_date: datetime = None,
    end_date: datetime = None,
    method: str = 'weighted'
) -> dict:
    """
    Compute current position for a single party.

    Args:
        party_code: Party code (e.g., 'LREM')
        start_date: Start date for documents (optional)
        end_date: End date for documents (optional)
        method: Aggregation method

    Returns:
        Dictionary with party position data
    """
    logger.info(f"Computing position for party: {party_code}")

    # Initialize repositories
    db = DatabaseRepository()
    party_repo = PartyRepository(db)
    doc_repo = DocumentRepository(db)

    # Get party
    party = party_repo.get_party_by_code(party_code)
    if not party:
        logger.error(f"Party {party_code} not found")
        return None

    # Get analyzed documents
    documents = doc_repo.get_documents_by_party(
        party.id,
        start_date=start_date,
        end_date=end_date,
        analyzed_only=True
    )

    if not documents:
        logger.warning(f"No analyzed documents found for {party_code}")
        return None

    logger.info(f"Found {len(documents)} analyzed documents for {party_code}")

    # Convert to format expected by aggregator
    doc_data = [
        {
            'valence': doc.valence,
            'arousal': doc.arousal,
            'date_published': doc.date_published,
            'word_count': doc.word_count
        }
        for doc in documents
    ]

    # Aggregate scores
    aggregator = PartyAffectAggregator()
    agg_result = aggregator.aggregate_documents(doc_data, method=method)

    # Map to circumplex
    mapper = CircumplexMapper()
    circumplex_info = mapper.map_to_circumplex(
        agg_result['valence'],
        agg_result['arousal']
    )

    # Combine results
    result = {
        'party_code': party_code,
        'party_name': party.name,
        'party_full_name': party.full_name,
        'political_position': party.political_position,
        'color': party.color,
        **agg_result,
        'circumplex': circumplex_info,
        'date_range': {
            'start': min(doc.date_published for doc in documents if doc.date_published),
            'end': max(doc.date_published for doc in documents if doc.date_published)
        }
    }

    return result


def compute_all_parties(
    start_date: datetime = None,
    end_date: datetime = None,
    method: str = 'weighted'
) -> list:
    """
    Compute positions for all parties.

    Args:
        start_date: Start date filter
        end_date: End date filter
        method: Aggregation method

    Returns:
        List of party position dictionaries
    """
    logger.info("Computing positions for all parties...")

    parties_config = load_party_config()
    results = []

    for party_config in parties_config:
        party_code = party_config['code']

        result = compute_party_position(
            party_code,
            start_date=start_date,
            end_date=end_date,
            method=method
        )

        if result:
            results.append(result)

    return results


def save_temporal_snapshots(
    party_code: str,
    period: str = 'monthly',
    method: str = 'weighted'
):
    """
    Compute and save temporal snapshots for a party.

    Args:
        party_code: Party code
        period: Period type ('monthly', 'weekly')
        method: Aggregation method
    """
    logger.info(f"Computing temporal snapshots for {party_code} ({period})")

    # Initialize repositories
    db = DatabaseRepository()
    party_repo = PartyRepository(db)
    doc_repo = DocumentRepository(db)
    snapshot_repo = TemporalSnapshotRepository(db)

    # Get party
    party = party_repo.get_party_by_code(party_code)
    if not party:
        logger.error(f"Party {party_code} not found")
        return

    # Get all analyzed documents
    documents = doc_repo.get_documents_by_party(party.id, analyzed_only=True)

    if not documents:
        logger.warning(f"No documents for {party_code}")
        return

    # Convert to format for aggregator
    doc_data = [
        {
            'valence': doc.valence,
            'arousal': doc.arousal,
            'date_published': doc.date_published,
            'word_count': doc.word_count
        }
        for doc in documents
    ]

    # Compute temporal aggregations
    aggregator = PartyAffectAggregator()

    if period == 'monthly':
        snapshots = aggregator.aggregate_monthly(doc_data, method=method)
    else:
        logger.error(f"Unsupported period: {period}")
        return

    # Save snapshots to database
    for snapshot in snapshots:
        snapshot_data = {
            'party_id': party.id,
            'snapshot_date': snapshot['snapshot_date'],
            'period_start': snapshot['period_start'],
            'period_end': snapshot['period_end'],
            'valence': snapshot['valence'],
            'arousal': snapshot['arousal'],
            'valence_std': snapshot['valence_std'],
            'arousal_std': snapshot['arousal_std'],
            'valence_ci': snapshot['valence_ci'],
            'arousal_ci': snapshot['arousal_ci'],
            'num_documents': snapshot['num_documents'],
            'aggregation_method': snapshot['aggregation_method']
        }

        snapshot_repo.create_snapshot(snapshot_data)
        logger.success(f"Saved snapshot for {snapshot['snapshot_date'].strftime('%Y-%m')}")

    logger.success(f"Saved {len(snapshots)} snapshots for {party_code}")


def display_results(results: list):
    """
    Display party positions in a formatted table.

    Args:
        results: List of party position dictionaries
    """
    if not results:
        logger.warning("No results to display")
        return

    logger.info("\n" + "="*100)
    logger.info("PARTY POSITIONS ON CIRCUMPLEX MODEL")
    logger.info("="*100)

    # Create table data
    table_data = []
    for r in results:
        table_data.append([
            r['party_code'],
            r['party_name'],
            f"{r['valence']:.3f}",
            f"{r['arousal']:.3f}",
            r['circumplex']['quadrant_label'],
            r['circumplex']['nearest_emotion'],
            r['num_documents']
        ])

    headers = [
        'Code', 'Party', 'Valence', 'Arousal',
        'Quadrant', 'Emotion', 'Docs'
    ]

    print("\n" + tabulate(table_data, headers=headers, tablefmt='grid'))

    # Display detailed interpretations
    logger.info("\nDETAILED INTERPRETATIONS:")
    logger.info("-"*100)

    mapper = CircumplexMapper()
    for r in results:
        logger.info(f"\n{r['party_name']} ({r['party_code']}):")
        logger.info(f"  Political Position: {r['political_position']}")
        logger.info(f"  Valence: {r['valence']:.3f} ± {r['valence_ci']:.3f} (95% CI)")
        logger.info(f"  Arousal: {r['arousal']:.3f} ± {r['arousal_ci']:.3f} (95% CI)")
        logger.info(f"  Documents: {r['num_documents']}")
        logger.info(f"  Date Range: {r['date_range']['start'].strftime('%Y-%m-%d')} to "
                   f"{r['date_range']['end'].strftime('%Y-%m-%d')}")

        interpretation = mapper.get_regional_interpretation(
            r['valence'],
            r['arousal']
        )
        logger.info(f"  Interpretation: {interpretation}")

    logger.info("="*100 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute party positions on the Circumplex Model"
    )

    parser.add_argument(
        '--party',
        type=str,
        help='Compute position for specific party'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Compute positions for all parties'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='weighted',
        choices=['weighted', 'mean', 'median'],
        help='Aggregation method (default: weighted)'
    )

    parser.add_argument(
        '--days',
        type=int,
        help='Only include documents from last N days'
    )

    parser.add_argument(
        '--save-snapshots',
        action='store_true',
        help='Save temporal snapshots to database'
    )

    parser.add_argument(
        '--period',
        type=str,
        default='monthly',
        choices=['monthly', 'weekly'],
        help='Period for temporal snapshots'
    )

    args = parser.parse_args()

    # Configure logging
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

    # Calculate date range if --days specified
    end_date = None
    start_date = None
    if args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        logger.info(f"Filtering to documents from last {args.days} days")

    # Execute command
    if args.save_snapshots:
        if args.party:
            save_temporal_snapshots(args.party, period=args.period, method=args.method)
        elif args.all:
            parties_config = load_party_config()
            for party in parties_config:
                save_temporal_snapshots(party['code'], period=args.period, method=args.method)
        else:
            logger.error("Specify --party or --all for saving snapshots")

    elif args.all:
        results = compute_all_parties(
            start_date=start_date,
            end_date=end_date,
            method=args.method
        )
        display_results(results)

    elif args.party:
        result = compute_party_position(
            args.party,
            start_date=start_date,
            end_date=end_date,
            method=args.method
        )
        if result:
            display_results([result])

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
