"""
NLP Text Analyzer Module
Advanced Natural Language Processing for technical documentation analysis.

Features:
- Named Entity Recognition (NER) for extracting part numbers, specs
- Text similarity matching for finding related issues
- Topic modeling to identify common problem areas
- Sentiment analysis for community feedback
- Keyword extraction from specifications
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
from collections import Counter
import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline


@dataclass
class Entity:
    """Represents an extracted named entity."""
    text: str
    entity_type: str
    start_idx: int
    end_idx: int
    confidence: float


@dataclass
class SimilarityMatch:
    """Represents a document similarity match."""
    doc_a: str
    doc_b: str
    similarity_score: float
    common_topics: List[str]


class NamedEntityRecognizer:
    """
    Named Entity Recognition for semiconductor documents.
    
    Extracts:
    - Part numbers (STM32F407VG)
    - Package types (LQFP144, BGA)
    - Frequencies (168 MHz)
    - Temperature ranges (-40°C to 85°C)
    - Email addresses
    - Version information
    """
    
    # Entity patterns
    PATTERNS = {
        "part_number": r'\b(?:STM|LPC|NXP|ARM|TM4)[A-Z0-9]{6,10}\b',
        "package_type": r'\b(?:BGA|LQFP|TFBGA|QFN|DIP|SOIC|SO8|SO16)\d*\b',
        "frequency": r'(\d+(?:\.\d+)?)\s*(?:MHz|kHz|GHz)',
        "temperature": r'(?:[-−]?\d+)\s*°?C(?:\s*to\s*(?:[-−]?\d+)\s*°?C)?',
        "voltage": r'(\d+(?:\.\d+)?)\s*[Vv](?:\s*to\s*\d+(?:\.\d+)?\s*[Vv])?',
        "pin_count": r'(\d+)\s*(?:-?pin|pin)',
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "version": r'[Vv](?:\d+\.\d+(?:\.\d+)?|errev\s*[A-Z]\d)',
        "date": r'(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})',
    }
    
    def __init__(self):
        """Initialize the NER."""
        self.compiled_patterns = {
            entity_type: re.compile(pattern, re.IGNORECASE)
            for entity_type, pattern in self.PATTERNS.items()
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for entity_type, pattern in self.compiled_patterns.items():
            for match in pattern.finditer(text):
                entity = Entity(
                    text=match.group(0),
                    entity_type=entity_type,
                    start_idx=match.start(),
                    end_idx=match.end(),
                    confidence=0.95  # Pattern-based confidence
                )
                entities.append(entity)
        
        # Sort by position
        entities.sort(key=lambda e: e.start_idx)
        
        # Remove duplicates (overlapping entities)
        unique_entities = []
        for entity in entities:
            if not any(self._overlaps(entity, e) for e in unique_entities):
                unique_entities.append(entity)
        
        return unique_entities
    
    def _overlaps(self, e1: Entity, e2: Entity) -> bool:
        """Check if two entities overlap."""
        return not (e1.end_idx <= e2.start_idx or e2.end_idx <= e1.start_idx)
    
    def extract_by_type(self, text: str, entity_type: str) -> List[str]:
        """
        Extract entities of a specific type.
        
        Args:
            text: Input text
            entity_type: Type of entity to extract
            
        Returns:
            List of extracted entity texts
        """
        entities = self.extract_entities(text)
        return [e.text for e in entities if e.entity_type == entity_type]


class TextSimilarityMatcher:
    """
    Finds similar documents and identifies common themes.
    
    Uses multiple similarity metrics: TF-IDF, semantic, word overlap, and character-level similarity.
    """
    
    def __init__(self, ngram_range: Tuple[int, int] = (1, 3)):
        """Initialize the Text Similarity Matcher."""
        self.vectorizer = TfidfVectorizer(
            max_features=200,
            ngram_range=ngram_range,
            min_df=1,
            max_df=0.95,
            stop_words=None  # Keep stop words for better domain matching
        )
        self.vectorizer_no_stop = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            stop_words='english'
        )
        self.tfidf_matrix = None
        self.tfidf_matrix_no_stop = None
        self.documents = None
    
    def fit(self, documents: List[str]):
        """
        Fit the similarity matcher on a corpus.
        
        Args:
            documents: List of documents to build index
        """
        self.documents = documents
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        self.tfidf_matrix_no_stop = self.vectorizer_no_stop.fit_transform(documents)
    
    def _word_overlap_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity based on word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using TF-IDF with all words (including stop words)."""
        from sklearn.feature_extraction.text import TfidfVectorizer as TV
        from sklearn.metrics.pairwise import cosine_similarity
        
        try:
            vec = TV(max_features=100, ngram_range=(1, 2), min_df=1)
            vectors = vec.fit_transform([text1, text2])
            return float(cosine_similarity(vectors[0], vectors[1])[0][0])
        except:
            return 0.0
    
    def _character_ngram_similarity(self, text1: str, text2: str) -> float:
        """Calculate character n-gram similarity for catching semantic variants."""
        from difflib import SequenceMatcher
        # Use sequence matcher for character-level similarity
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def compute_combined_similarity(self, text1: str, text2: str) -> dict:
        """
        Compute multiple similarity metrics and return combined score.
        
        Returns:
            Dictionary with individual and combined similarity scores
        """
        # Compute multiple similarity metrics
        word_overlap = self._word_overlap_similarity(text1, text2)
        semantic = self._semantic_similarity(text1, text2)
        char_ngram = self._character_ngram_similarity(text1, text2)
        
        # TF-IDF similarity (handles short domain-specific text better with stop words)
        tfidf_vec = TfidfVectorizer(max_features=150, ngram_range=(1, 3), min_df=1)
        try:
            vectors = tfidf_vec.fit_transform([text1, text2])
            tfidf_sim = float(cosine_similarity(vectors[0], vectors[1])[0][0])
        except:
            tfidf_sim = 0.0
        
        # Weighted combination: prioritize semantic and character similarity for short technical text
        combined = (
            word_overlap * 0.20 +      # 20% - exact word matches
            semantic * 0.25 +          # 25% - semantic/TF-IDF meaning
            char_ngram * 0.30 +        # 30% - character patterns (catches "UART"/"Serial", "fails"/"drops")
            tfidf_sim * 0.25            # 25% - TF-IDF with domain context
        )
        
        return {
            'combined': combined,
            'word_overlap': word_overlap,
            'semantic': semantic,
            'character_ngram': char_ngram,
            'tfidf': tfidf_sim
        }
    
    def find_similar(self, query: str, top_k: int = 5) -> List[Tuple[int, str, float]]:
        """
        Find similar documents to a query.
        
        Args:
            query: Query document
            top_k: Number of results to return
            
        Returns:
            List of (index, document, similarity_score) tuples
        """
        if self.tfidf_matrix is None:
            raise ValueError("Matcher must be fitted before querying")
        
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero similarities
                results.append((idx, self.documents[idx], float(similarities[idx])))
        
        return results
    
    def find_related_pairs(self, threshold: float = 0.5) -> List[SimilarityMatch]:
        """
        Find all pairs of similar documents.
        
        Args:
            threshold: Minimum similarity score
            
        Returns:
            List of similar document pairs
        """
        if self.tfidf_matrix is None:
            raise ValueError("Matcher must be fitted before finding pairs")
        
        # Compute pairwise similarity
        similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        matches = []
        n_docs = len(self.documents)
        
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                score = similarity_matrix[i, j]
                if score >= threshold:
                    # Extract common topics
                    topics = self._extract_common_topics(
                        self.documents[i],
                        self.documents[j]
                    )
                    
                    match = SimilarityMatch(
                        doc_a=self.documents[i][:100],
                        doc_b=self.documents[j][:100],
                        similarity_score=float(score),
                        common_topics=topics
                    )
                    matches.append(match)
        
        # Sort by similarity
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches
    
    def _extract_common_topics(self, doc1: str, doc2: str) -> List[str]:
        """Extract common keywords between documents."""
        # Simple implementation: find common words
        words1 = set(doc1.lower().split())
        words2 = set(doc2.lower().split())
        common = words1 & words2
        
        # Filter out common English words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are', 'was', 'were'}
        common = [w for w in common if w not in stopwords and len(w) > 2]
        
        return list(common)[:5]


class TopicModeler:
    """
    Identifies main topics in a document collection.
    
    Uses Latent Dirichlet Allocation (LDA) to discover hidden topic structure.
    """
    
    def __init__(self, n_topics: int = 5):
        """Initialize the Topic Modeler."""
        self.n_topics = n_topics
        self.vectorizer = None
        self.lda_model = None
        self.feature_names = None
    
    def fit(self, documents: List[str]) -> Dict[int, List[str]]:
        """
        Fit LDA model on documents.
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary mapping topic ID to top words
        """
        # Create document-term matrix
        self.vectorizer = CountVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words='english'
        )
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Fit LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=50
        )
        self.lda_model.fit(doc_term_matrix)
        
        return self._get_topics()
    
    def _get_topics(self, n_words: int = 10) -> Dict[int, List[str]]:
        """Extract top words for each topic."""
        topics = {}
        
        for topic_id, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [self.feature_names[i] for i in top_indices]
            topics[topic_id] = top_words
        
        return topics
    
    def predict_topics(self, document: str) -> List[Tuple[int, float]]:
        """
        Predict topics for a document.
        
        Args:
            document: Input document
            
        Returns:
            List of (topic_id, probability) tuples
        """
        if self.lda_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        doc_vector = self.vectorizer.transform([document])
        topic_dist = self.lda_model.transform(doc_vector)[0]
        
        # Return topics with non-zero probability
        return [(i, prob) for i, prob in enumerate(topic_dist) if prob > 0.1]


class KeywordExtractor:
    """
    Extracts important keywords and phrases from technical documents.
    """
    
    def __init__(self):
        """Initialize the Keyword Extractor."""
        self.vectorizer = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 3),
            stop_words='english'
        )
    
    def extract_keywords(self, document: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Extract top keywords from a document.
        
        Args:
            document: Input document
            top_k: Number of keywords to extract
            
        Returns:
            List of (keyword, score) tuples
        """
        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform([document])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get TF-IDF scores
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Get top-k
        top_indices = np.argsort(tfidf_scores)[-top_k:][::-1]
        keywords = [(feature_names[i], tfidf_scores[i]) for i in top_indices if tfidf_scores[i] > 0]
        
        return keywords
    
    @staticmethod
    def extract_technical_terms(text: str) -> List[str]:
        """
        Extract technical jargon specific to semiconductors.
        
        Args:
            text: Input text
            
        Returns:
            List of technical terms found
        """
        technical_terms = {
            'peripheral': r'\b(?:UART|SPI|I2C|ADC|DAC|DMA|GPIO|PWM|USB|CAN|RTC|WDT|TIMER)\b',
            'architecture': r'\b(?:ARM|RISC|MIPS|x86|AVR|Cortex-M|NEON)\b',
            'specification': r'\b(?:MHz|GHz|mA|mW|ns|µs|°C|V|bit|byte)\b',
            'process': r'\b(?:nm|µm|process|node|technology|etch)\b',
        }
        
        terms = []
        for term_type, pattern in technical_terms.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms.extend(matches)
        
        return list(set(terms))


class SentimentAnalyzer:
    """
    Analyzes sentiment in issue reports and community feedback.
    """
    
    # Sentiment keywords (simple lexicon-based approach)
    POSITIVE_WORDS = {
        'excellent', 'great', 'good', 'fantastic', 'amazing', 'works', 'perfect',
        'solved', 'fixed', 'resolved', 'helpful', 'thank', 'appreciate'
    }
    
    NEGATIVE_WORDS = {
        'broken', 'crash', 'error', 'fail', 'problem', 'issue', 'bug', 'hang',
        'freeze', 'terrible', 'bad', 'worst', 'frustrating', 'doesn\'t work'
    }
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (sentiment, confidence)
        """
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.POSITIVE_WORDS if word in text_lower)
        negative_count = sum(1 for word in self.NEGATIVE_WORDS if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = positive_count / (positive_count + negative_count + 1)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = negative_count / (positive_count + negative_count + 1)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return sentiment, float(confidence)


class NLPAnalyzer:
    """
    Complete NLP analysis pipeline.
    """
    
    def __init__(self):
        """Initialize the NLP Analyzer."""
        self.ner = NamedEntityRecognizer()
        self.similarity_matcher = TextSimilarityMatcher()
        self.topic_modeler = TopicModeler(n_topics=5)
        self.keyword_extractor = KeywordExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    def analyze_document(self, text: str) -> Dict[str, any]:
        """
        Perform complete NLP analysis on a document.
        
        Args:
            text: Input document
            
        Returns:
            Dictionary with analysis results
        """
        return {
            "entities": self.ner.extract_entities(text),
            "keywords": self.keyword_extractor.extract_keywords(text),
            "technical_terms": KeywordExtractor.extract_technical_terms(text),
            "sentiment": self.sentiment_analyzer.analyze_sentiment(text),
        }


# Import for topic modeling
from sklearn.feature_extraction.text import CountVectorizer


def main():
    """Demonstrate NLP capabilities."""
    print("\n" + "=" * 70)
    print("NLP TEXT ANALYZER - SEMIINTEL")
    print("=" * 70)
    
    # Sample technical documents
    documents = [
        "STM32F407VG UART transmission drops characters at 115200 baud rate",
        "I2C clock stretching causes system hang requiring hard reset",
        "DMA memory corruption in certain access patterns at 168 MHz",
        "USB enumeration fails intermittently with external hub",
        "ADC sampling produces incorrect values in low-power mode",
    ]
    
    analyzer = NLPAnalyzer()
    
    # 1. Named Entity Recognition
    print("\n1. NAMED ENTITY RECOGNITION")
    print("-" * 70)
    
    test_text = "STM32F407VG in LQFP144 package operating at 168 MHz from 2.0V to 3.6V"
    entities = analyzer.ner.extract_entities(test_text)
    
    print(f"\nText: {test_text}")
    print("\nExtracted Entities:")
    for entity in entities:
        print(f"  [{entity.entity_type:12}] {entity.text:15} @ {entity.start_idx}-{entity.end_idx}")
    
    # 2. Text Similarity
    print("\n\n2. TEXT SIMILARITY MATCHING")
    print("-" * 70)
    
    analyzer.similarity_matcher.fit(documents)
    
    query = "UART communication problem at high baud rate"
    similar = analyzer.similarity_matcher.find_similar(query, top_k=2)
    
    print(f"\nQuery: {query}")
    print("\nMost Similar Documents:")
    for idx, doc, score in similar:
        print(f"  [{score:.2%}] {doc[:70]}...")
    
    # 3. Keyword Extraction
    print("\n\n3. KEYWORD EXTRACTION")
    print("-" * 70)
    
    keywords = analyzer.keyword_extractor.extract_keywords(documents[0], top_k=5)
    print(f"\nDocument: {documents[0]}")
    print("\nTop Keywords:")
    for keyword, score in keywords:
        print(f"  {keyword:20} (TF-IDF: {score:.4f})")
    
    # 4. Technical Terms
    print("\n\n4. TECHNICAL TERM EXTRACTION")
    print("-" * 70)
    
    for doc in documents:
        terms = KeywordExtractor.extract_technical_terms(doc)
        if terms:
            print(f"\n{doc[:50]}...")
            print(f"  Technical terms: {', '.join(terms)}")
    
    # 5. Sentiment Analysis
    print("\n\n5. SENTIMENT ANALYSIS")
    print("-" * 70)
    
    test_sentiments = [
        "This chip is amazing and works perfectly!",
        "The datasheet is confusing and the chip doesn't work as advertised.",
        "The documentation could be improved.",
    ]
    
    for text in test_sentiments:
        sentiment, confidence = analyzer.sentiment_analyzer.analyze_sentiment(text)
        print(f"\n\"{text}\"")
        print(f"  Sentiment: {sentiment} ({confidence:.2%} confidence)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
