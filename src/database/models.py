"""SQLAlchemy database models for French Political Affect Analysis."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Party(Base):
    """Political party model."""

    __tablename__ = 'parties'

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(10), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    full_name = Column(String(200))
    website = Column(String(500))
    news_url = Column(String(500))
    color = Column(String(7))  # Hex color for visualization
    political_position = Column(String(50))
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    documents = relationship("Document", back_populates="party", cascade="all, delete-orphan")
    temporal_snapshots = relationship("TemporalSnapshot", back_populates="party", cascade="all, delete-orphan")
    scraping_logs = relationship("ScrapingLog", back_populates="party", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Party(code='{self.code}', name='{self.name}')>"


class Document(Base):
    """Political document model."""

    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True, autoincrement=True)
    party_id = Column(Integer, ForeignKey('parties.id'), nullable=False, index=True)

    url = Column(String(1000), unique=True, nullable=False, index=True)
    title = Column(String(500))
    date_published = Column(DateTime, index=True)
    date_scraped = Column(DateTime, default=datetime.utcnow)

    content = Column(Text, nullable=False)
    word_count = Column(Integer)

    # NLP scores (nullable until analysis is run)
    valence = Column(Float)
    arousal = Column(Float)
    analyzed_at = Column(DateTime)

    # Metadata
    document_type = Column(String(50))  # 'press_release', 'manifesto', 'article', etc.
    language_detected = Column(String(10))

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    party = relationship("Party", back_populates="documents")
    sentences = relationship("Sentence", back_populates="document", cascade="all, delete-orphan")

    # Indexes for common queries
    __table_args__ = (
        Index('idx_party_date', 'party_id', 'date_published'),
        Index('idx_analyzed', 'analyzed_at'),
    )

    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title[:50]}...', party={self.party.code if self.party else None})>"


class Sentence(Base):
    """Individual sentence model for fine-grained analysis."""

    __tablename__ = 'sentences'

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False, index=True)

    text = Column(Text, nullable=False)
    position = Column(Integer)  # Sentence order in document (0-indexed)
    word_count = Column(Integer)

    # NLP scores
    valence = Column(Float)
    arousal = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="sentences")

    def __repr__(self):
        return f"<Sentence(id={self.id}, text='{self.text[:30]}...', valence={self.valence}, arousal={self.arousal})>"


class TemporalSnapshot(Base):
    """Temporal snapshot of party affect scores."""

    __tablename__ = 'temporal_snapshots'

    id = Column(Integer, primary_key=True, autoincrement=True)
    party_id = Column(Integer, ForeignKey('parties.id'), nullable=False, index=True)

    snapshot_date = Column(DateTime, nullable=False, index=True)  # End of period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)

    # Aggregated scores
    valence = Column(Float, nullable=False)
    arousal = Column(Float, nullable=False)

    # Statistical measures
    valence_std = Column(Float)
    arousal_std = Column(Float)
    valence_ci = Column(Float)  # 95% confidence interval
    arousal_ci = Column(Float)

    # Metadata
    num_documents = Column(Integer, nullable=False)
    aggregation_method = Column(String(50))  # 'weighted', 'mean', 'median'

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    party = relationship("Party", back_populates="temporal_snapshots")

    # Indexes for temporal queries
    __table_args__ = (
        Index('idx_party_snapshot', 'party_id', 'snapshot_date'),
    )

    def __repr__(self):
        return f"<TemporalSnapshot(party={self.party.code if self.party else None}, date={self.snapshot_date}, valence={self.valence:.2f}, arousal={self.arousal:.2f})>"


class ScrapingLog(Base):
    """Log of scraping operations."""

    __tablename__ = 'scraping_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    party_id = Column(Integer, ForeignKey('parties.id'), nullable=False, index=True)

    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    status = Column(String(20), nullable=False)  # 'success', 'failed', 'partial'

    documents_found = Column(Integer, default=0)
    documents_new = Column(Integer, default=0)
    documents_updated = Column(Integer, default=0)

    error_message = Column(Text)
    duration_seconds = Column(Float)

    # Relationships
    party = relationship("Party", back_populates="scraping_logs")

    def __repr__(self):
        return f"<ScrapingLog(party={self.party.code if self.party else None}, status='{self.status}', timestamp={self.timestamp})>"
