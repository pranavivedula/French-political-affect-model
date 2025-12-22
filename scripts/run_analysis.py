"""Script to run NLP analysis on scraped documents."""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from loguru import logger
from tqdm import tqdm

from src.nlp.valence_detector import ValenceDetector
from src.nlp.arousal_detector import ArousalDetector
from src.database.repository import (
    DatabaseRepository,
    DocumentRepository,
    SentenceRepository
)


def analyze_document(doc, valence_detector, arousal_detector, save_sentences=True):
    """
    Analyze a single document for valence and arousal.

    Args:
        doc: Document object from database
        valence_detector: ValenceDetector instance
        arousal_detector: ArousalDetector instance
        save_sentences: Whether to save sentence-level scores

    Returns:
        Tuple of (valence, arousal, sentence_data)
    """
    try:
        # Analyze valence
        valence_results = valence_detector.analyze_document(
            doc.content,
            return_sentences=save_sentences
        )

        # Analyze arousal
        arousal_results = arousal_detector.analyze_document(
            doc.content,
            return_sentences=save_sentences
        )

        document_valence = valence_results['document_valence']
        document_arousal = arousal_results['document_arousal']

        # Combine sentence-level data if requested
        sentence_data = []
        if save_sentences and valence_results.get('sentences'):
            for i, (v_sent, a_sent) in enumerate(zip(
                valence_results['sentences'],
                arousal_results['sentences']
            )):
                sentence_data.append({
                    'document_id': doc.id,
                    'text': v_sent['text'],
                    'position': i,
                    'word_count': v_sent['word_count'],
                    'valence': v_sent['valence'],
                    'arousal': a_sent['arousal']
                })

        logger.success(
            f"Analyzed document {doc.id}: valence={document_valence:.3f}, "
            f"arousal={document_arousal:.3f}"
        )

        return document_valence, document_arousal, sentence_data

    except Exception as e:
        logger.error(f"Error analyzing document {doc.id}: {e}")
        return None, None, []


def run_analysis(
    limit: int = None,
    party_code: str = None,
    save_sentences: bool = True,
    batch_size: int = 10
):
    """
    Run NLP analysis on unanalyzed documents.

    Args:
        limit: Maximum number of documents to analyze
        party_code: Only analyze documents from this party
        save_sentences: Whether to save sentence-level scores
        batch_size: Number of documents to process before committing
    """
    logger.info("Starting NLP analysis pipeline...")

    # Initialize repositories
    db = DatabaseRepository()
    doc_repo = DocumentRepository(db)
    sent_repo = SentenceRepository(db)

    # Get unanalyzed documents
    logger.info("Fetching unanalyzed documents...")
    unanalyzed_docs = doc_repo.get_unanalyzed_documents(limit=limit)

    if not unanalyzed_docs:
        logger.info("No unanalyzed documents found")
        return

    # Filter by party if specified
    if party_code:
        unanalyzed_docs = [
            doc for doc in unanalyzed_docs
            if doc.party.code == party_code
        ]
        logger.info(f"Filtered to {len(unanalyzed_docs)} documents from party {party_code}")

    logger.info(f"Found {len(unanalyzed_docs)} documents to analyze")

    # Initialize NLP models
    logger.info("Loading NLP models...")
    valence_detector = ValenceDetector()
    arousal_detector = ArousalDetector()
    logger.success("Models loaded successfully")

    # Process documents
    analyzed_count = 0
    failed_count = 0

    with tqdm(total=len(unanalyzed_docs), desc="Analyzing documents") as pbar:
        for doc in unanalyzed_docs:
            try:
                # Analyze document
                valence, arousal, sentence_data = analyze_document(
                    doc,
                    valence_detector,
                    arousal_detector,
                    save_sentences=save_sentences
                )

                if valence is not None and arousal is not None:
                    # Update document with scores
                    success = doc_repo.update_document_scores(
                        doc.id,
                        valence=valence,
                        arousal=arousal
                    )

                    if success:
                        # Save sentence-level scores if requested
                        if save_sentences and sentence_data:
                            sent_repo.create_sentences_bulk(sentence_data)

                        analyzed_count += 1
                    else:
                        failed_count += 1
                        logger.error(f"Failed to update document {doc.id}")
                else:
                    failed_count += 1

            except Exception as e:
                logger.error(f"Error processing document {doc.id}: {e}")
                failed_count += 1

            pbar.update(1)
            pbar.set_postfix({
                'analyzed': analyzed_count,
                'failed': failed_count
            })

    # Final summary
    logger.success(
        f"Analysis complete: {analyzed_count} documents analyzed, "
        f"{failed_count} failed"
    )


def test_analysis(text: str = None):
    """
    Test the NLP analysis pipeline with sample text.

    Args:
        text: Optional text to analyze (uses default French political sample if not provided)
    """
    if not text:
        text = """
        Le gouvernement annonce de nouvelles mesures pour lutter contre le changement climatique.
        Cette initiative représente un tournant majeur dans notre politique environnementale.
        Les citoyens doivent être mobilisés pour faire face à cette crise urgente!
        Ensemble, nous pouvons construire un avenir durable pour nos enfants.
        """

    logger.info("Testing NLP analysis pipeline...")
    logger.info(f"Sample text:\n{text}\n")

    # Initialize detectors
    logger.info("Loading models...")
    valence_detector = ValenceDetector()
    arousal_detector = ArousalDetector()

    # Analyze text
    logger.info("Analyzing valence...")
    valence_results = valence_detector.analyze_document(text, return_sentences=True)

    logger.info("Analyzing arousal...")
    arousal_results = arousal_detector.analyze_document(text, return_sentences=True)

    # Display results
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS RESULTS")
    logger.info("="*80)
    logger.info(f"Document Valence: {valence_results['document_valence']:.3f} "
                f"(range: -1 to +1)")
    logger.info(f"Document Arousal: {arousal_results['document_arousal']:.3f} "
                f"(range: 0 to 1)")
    logger.info(f"Number of sentences: {valence_results['num_sentences']}")

    logger.info("\nSentence-level scores:")
    logger.info("-"*80)
    for i, (v_sent, a_sent) in enumerate(zip(
        valence_results['sentences'],
        arousal_results['sentences']
    ), 1):
        logger.info(f"\nSentence {i}:")
        logger.info(f"  Text: {v_sent['text'][:80]}...")
        logger.info(f"  Valence: {v_sent['valence']:.3f}")
        logger.info(f"  Arousal: {a_sent['arousal']:.3f}")

    logger.info("="*80)
    logger.success("Test complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run NLP analysis on political documents")

    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of documents to analyze'
    )

    parser.add_argument(
        '--party',
        type=str,
        help='Only analyze documents from this party (e.g., LREM, RN)'
    )

    parser.add_argument(
        '--no-sentences',
        action='store_true',
        help='Skip saving sentence-level scores (faster)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test analysis with sample text'
    )

    parser.add_argument(
        '--test-text',
        type=str,
        help='Custom text for test analysis'
    )

    args = parser.parse_args()

    # Configure logging
    logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

    if args.test:
        test_analysis(text=args.test_text)
    else:
        run_analysis(
            limit=args.limit,
            party_code=args.party,
            save_sentences=not args.no_sentences
        )


if __name__ == "__main__":
    main()
