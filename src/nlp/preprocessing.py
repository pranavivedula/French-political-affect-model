"""Text preprocessing utilities for French political texts."""

import re
from typing import List, Optional
import nltk
from loguru import logger

from config.settings import MIN_TEXT_LENGTH, SENTENCE_MIN_WORDS, MAX_SEQUENCE_LENGTH

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)


class TextPreprocessor:
    """Preprocessor for French political texts."""

    def __init__(self):
        """Initialize the preprocessor."""
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')

    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted elements.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove Twitter handles and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)

        # Normalize whitespace (but preserve paragraph breaks)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences.

        Args:
            text: Text to segment

        Returns:
            List of sentences
        """
        if not text:
            return []

        # Use NLTK's French sentence tokenizer
        sentences = self.sentence_tokenizer.tokenize(text)

        # Filter out very short sentences
        filtered_sentences = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) >= SENTENCE_MIN_WORDS:
                filtered_sentences.append(sentence.strip())

        return filtered_sentences

    def chunk_long_text(self, text: str, max_length: int = MAX_SEQUENCE_LENGTH, overlap: int = 50) -> List[str]:
        """
        Split long text into chunks with overlap for transformer models.

        Args:
            text: Text to chunk
            max_length: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks

        Returns:
            List of text chunks
        """
        # Simple word-based chunking (approximates tokens)
        words = text.split()

        if len(words) <= max_length:
            return [text]

        chunks = []
        start = 0

        while start < len(words):
            end = start + max_length
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))

            # Move start position with overlap
            start = end - overlap

        return chunks

    def preprocess_document(self, text: str) -> Optional[dict]:
        """
        Preprocess a complete document.

        Args:
            text: Raw document text

        Returns:
            Dictionary with preprocessed data or None if text too short
        """
        # Clean text
        cleaned = self.clean_text(text)

        # Check minimum length
        if len(cleaned) < MIN_TEXT_LENGTH:
            logger.warning(f"Text too short after cleaning: {len(cleaned)} chars")
            return None

        # Segment into sentences
        sentences = self.segment_sentences(cleaned)

        if not sentences:
            logger.warning("No valid sentences found after segmentation")
            return None

        # Calculate statistics
        word_count = len(cleaned.split())

        return {
            'cleaned_text': cleaned,
            'sentences': sentences,
            'num_sentences': len(sentences),
            'word_count': word_count,
            'char_count': len(cleaned)
        }

    def prepare_for_model(self, text: str, max_length: int = MAX_SEQUENCE_LENGTH) -> List[str]:
        """
        Prepare text for input to transformer model.

        Handles long texts by chunking if necessary.

        Args:
            text: Text to prepare
            max_length: Maximum sequence length

        Returns:
            List of text chunks ready for model input
        """
        # Clean text first
        cleaned = self.clean_text(text)

        # Check if chunking is needed (approximate with word count)
        words = cleaned.split()
        if len(words) <= max_length:
            return [cleaned]

        # Chunk long text
        return self.chunk_long_text(cleaned, max_length=max_length)


def preprocess_text(text: str) -> Optional[dict]:
    """
    Convenience function to preprocess text.

    Args:
        text: Raw text

    Returns:
        Preprocessed data dictionary or None
    """
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_document(text)


def clean_and_segment(text: str) -> List[str]:
    """
    Convenience function to clean text and segment into sentences.

    Args:
        text: Raw text

    Returns:
        List of cleaned sentences
    """
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.clean_text(text)
    return preprocessor.segment_sentences(cleaned)
