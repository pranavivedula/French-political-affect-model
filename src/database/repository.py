"""Data access layer for database operations."""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, and_, or_, desc
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from config.settings import DATABASE_URL
from src.database.models import Base, Party, Document, Sentence, TemporalSnapshot, ScrapingLog


class DatabaseRepository:
    """Repository for database operations."""

    def __init__(self, database_url: str = None):
        """Initialize database repository."""
        self.database_url = database_url or DATABASE_URL
        self.engine = create_engine(self.database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """Drop all tables in the database."""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Session:
        """Get a database session context manager."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


class PartyRepository:
    """Repository for Party operations."""

    def __init__(self, db: DatabaseRepository):
        self.db = db

    def create_party(self, party_data: Dict[str, Any]) -> Party:
        """Create a new party."""
        with self.db.get_session() as session:
            party = Party(**party_data)
            session.add(party)
            session.flush()
            session.refresh(party)
            return party

    def get_party_by_code(self, code: str) -> Optional[Party]:
        """Get party by code."""
        with self.db.get_session() as session:
            return session.query(Party).filter(Party.code == code).first()

    def get_all_parties(self) -> List[Party]:
        """Get all parties."""
        with self.db.get_session() as session:
            return session.query(Party).all()

    def get_party_by_id(self, party_id: int) -> Optional[Party]:
        """Get party by ID."""
        with self.db.get_session() as session:
            return session.query(Party).filter(Party.id == party_id).first()


class DocumentRepository:
    """Repository for Document operations."""

    def __init__(self, db: DatabaseRepository):
        self.db = db

    def create_document(self, document_data: Dict[str, Any]) -> Document:
        """Create a new document."""
        with self.db.get_session() as session:
            document = Document(**document_data)
            session.add(document)
            session.flush()
            session.refresh(document)
            return document

    def document_exists(self, url: str) -> bool:
        """Check if document with URL already exists."""
        with self.db.get_session() as session:
            return session.query(Document).filter(Document.url == url).first() is not None

    def get_document_by_url(self, url: str) -> Optional[Document]:
        """Get document by URL."""
        with self.db.get_session() as session:
            return session.query(Document).filter(Document.url == url).first()

    def get_unanalyzed_documents(self, limit: int = None) -> List[Document]:
        """Get documents that haven't been analyzed yet."""
        with self.db.get_session() as session:
            query = session.query(Document).filter(Document.valence.is_(None))
            if limit:
                query = query.limit(limit)
            return query.all()

    def get_documents_by_party(
        self,
        party_id: int,
        start_date: datetime = None,
        end_date: datetime = None,
        analyzed_only: bool = True
    ) -> List[Document]:
        """Get documents for a party within date range."""
        with self.db.get_session() as session:
            query = session.query(Document).filter(Document.party_id == party_id)

            if analyzed_only:
                query = query.filter(Document.valence.isnot(None))

            if start_date:
                query = query.filter(Document.date_published >= start_date)

            if end_date:
                query = query.filter(Document.date_published <= end_date)

            return query.order_by(desc(Document.date_published)).all()

    def update_document_scores(
        self,
        document_id: int,
        valence: float,
        arousal: float
    ) -> bool:
        """Update document with NLP scores."""
        with self.db.get_session() as session:
            document = session.query(Document).filter(Document.id == document_id).first()
            if document:
                document.valence = valence
                document.arousal = arousal
                document.analyzed_at = datetime.utcnow()
                session.commit()
                return True
            return False

    def get_recent_documents(self, limit: int = 20) -> List[Document]:
        """Get most recent documents across all parties."""
        with self.db.get_session() as session:
            return (
                session.query(Document)
                .filter(Document.valence.isnot(None))
                .order_by(desc(Document.date_published))
                .limit(limit)
                .all()
            )

    def get_extreme_documents(
        self,
        party_id: int,
        metric: str = 'valence',
        top_n: int = 5,
        ascending: bool = True
    ) -> List[Document]:
        """Get documents with extreme valence or arousal scores."""
        with self.db.get_session() as session:
            query = session.query(Document).filter(
                and_(
                    Document.party_id == party_id,
                    Document.valence.isnot(None)
                )
            )

            if metric == 'valence':
                query = query.order_by(Document.valence if ascending else desc(Document.valence))
            else:  # arousal
                query = query.order_by(Document.arousal if ascending else desc(Document.arousal))

            return query.limit(top_n).all()


class SentenceRepository:
    """Repository for Sentence operations."""

    def __init__(self, db: DatabaseRepository):
        self.db = db

    def create_sentence(self, sentence_data: Dict[str, Any]) -> Sentence:
        """Create a new sentence."""
        with self.db.get_session() as session:
            sentence = Sentence(**sentence_data)
            session.add(sentence)
            session.flush()
            session.refresh(sentence)
            return sentence

    def create_sentences_bulk(self, sentences_data: List[Dict[str, Any]]):
        """Create multiple sentences in bulk."""
        with self.db.get_session() as session:
            sentences = [Sentence(**data) for data in sentences_data]
            session.bulk_save_objects(sentences)
            session.commit()

    def get_sentences_by_document(self, document_id: int) -> List[Sentence]:
        """Get all sentences for a document."""
        with self.db.get_session() as session:
            return (
                session.query(Sentence)
                .filter(Sentence.document_id == document_id)
                .order_by(Sentence.position)
                .all()
            )


class TemporalSnapshotRepository:
    """Repository for TemporalSnapshot operations."""

    def __init__(self, db: DatabaseRepository):
        self.db = db

    def create_snapshot(self, snapshot_data: Dict[str, Any]) -> TemporalSnapshot:
        """Create a new temporal snapshot."""
        with self.db.get_session() as session:
            snapshot = TemporalSnapshot(**snapshot_data)
            session.add(snapshot)
            session.flush()
            session.refresh(snapshot)
            return snapshot

    def get_snapshots_by_party(
        self,
        party_id: int,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> List[TemporalSnapshot]:
        """Get temporal snapshots for a party within date range."""
        with self.db.get_session() as session:
            query = session.query(TemporalSnapshot).filter(
                TemporalSnapshot.party_id == party_id
            )

            if start_date:
                query = query.filter(TemporalSnapshot.snapshot_date >= start_date)

            if end_date:
                query = query.filter(TemporalSnapshot.snapshot_date <= end_date)

            return query.order_by(TemporalSnapshot.snapshot_date).all()

    def get_latest_snapshot(self, party_id: int) -> Optional[TemporalSnapshot]:
        """Get the most recent snapshot for a party."""
        with self.db.get_session() as session:
            return (
                session.query(TemporalSnapshot)
                .filter(TemporalSnapshot.party_id == party_id)
                .order_by(desc(TemporalSnapshot.snapshot_date))
                .first()
            )


class ScrapingLogRepository:
    """Repository for ScrapingLog operations."""

    def __init__(self, db: DatabaseRepository):
        self.db = db

    def create_log(self, log_data: Dict[str, Any]) -> ScrapingLog:
        """Create a new scraping log entry."""
        with self.db.get_session() as session:
            log = ScrapingLog(**log_data)
            session.add(log)
            session.flush()
            session.refresh(log)
            return log

    def get_recent_logs(self, party_id: int = None, limit: int = 10) -> List[ScrapingLog]:
        """Get recent scraping logs."""
        with self.db.get_session() as session:
            query = session.query(ScrapingLog)

            if party_id:
                query = query.filter(ScrapingLog.party_id == party_id)

            return query.order_by(desc(ScrapingLog.timestamp)).limit(limit).all()

    def get_last_successful_scrape(self, party_id: int) -> Optional[ScrapingLog]:
        """Get the last successful scrape for a party."""
        with self.db.get_session() as session:
            return (
                session.query(ScrapingLog)
                .filter(
                    and_(
                        ScrapingLog.party_id == party_id,
                        ScrapingLog.status == 'success'
                    )
                )
                .order_by(desc(ScrapingLog.timestamp))
                .first()
            )
