"""
SEMIINTEL Modules Package
Contains all intelligence gathering and analysis modules.
"""

from .dorking_engine import DorkingEngine, STM_CHIP_MODELS
from .pdf_parser import PDFMetadataExtractor, EmailIntelligenceExtractor
from .github_scanner import (
    GitHubScanner,
    StackOverflowScanner,
    VerificationAnalyzer,
    IssueType,
    CommunityIssue,
)
from .ml_analyzer import (
    MLPipeline,
    SeverityClassifier,
    IssueClusterer,
    PerformancePredictor,
    AnomalyDetector,
)
from .dataset_loader import (
    DatasetManager,
    KaggleDatasetRegistry,
    SyntheticDataGenerator,
)
from .nlp_analyzer import (
    NLPAnalyzer,
    NamedEntityRecognizer,
    TextSimilarityMatcher,
    TopicModeler,
    KeywordExtractor,
    SentimentAnalyzer,
)

__version__ = "2.0.0"
__author__ = "Your Name"
__description__ = "Semiconductor Intelligence Automation Tool with ML/NLP"

__all__ = [
    "DorkingEngine",
    "STM_CHIP_MODELS",
    "PDFMetadataExtractor",
    "EmailIntelligenceExtractor",
    "GitHubScanner",
    "StackOverflowScanner",
    "VerificationAnalyzer",
    "IssueType",
    "CommunityIssue",
    "MLPipeline",
    "SeverityClassifier",
    "IssueClusterer",
    "PerformancePredictor",
    "AnomalyDetector",
    "DatasetManager",
    "KaggleDatasetRegistry",
    "SyntheticDataGenerator",
    "NLPAnalyzer",
    "NamedEntityRecognizer",
    "TextSimilarityMatcher",
    "TopicModeler",
    "KeywordExtractor",
    "SentimentAnalyzer",
]
