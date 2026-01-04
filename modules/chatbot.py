"""
Lightweight retrieval-based chatbot used inside the Streamlit app.
It does not call any paid APIs; all responses come from local TF-IDF
similarity over curated semiconductor datasets and platform summaries.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ChatTurn:
    """Represents a single interaction between the user and the bot."""
    user: str
    bot: str


class ResponseFilter:
    """Content filtering to ensure safe, appropriate responses."""
    
    # Patterns that indicate potentially problematic content
    OFFENSIVE_PATTERNS = [
        r'\b(nigga|nigger|retard|stupid|idiot|dumb)\b',
        r'(?:you|ur|u)\s+(?:are\s+)?(?:stupid|dumb|retard|idiot)',
    ]
    
    @staticmethod
    def is_safe(text: str) -> bool:
        """Check if response contains appropriate content."""
        text_lower = text.lower()
        for pattern in ResponseFilter.OFFENSIVE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False
        return True
    
    @staticmethod
    def sanitize(text: str) -> str:
        """Sanitize response to remove problematic patterns."""
        for pattern in ResponseFilter.OFFENSIVE_PATTERNS:
            text = re.sub(pattern, '[filtered]', text, flags=re.IGNORECASE)
        return text


class ConversationalRetrievalBot:
    """Simple retrieval chatbot that runs entirely locally.

    It embeds knowledge snippets using TF-IDF and serves the best-matching
    snippet as a concise answer, keeping a short conversational memory.
    """

    def __init__(
        self,
        knowledge_snippets: List[Dict[str, str]],
        max_history: int = 6,
    ) -> None:
        if not knowledge_snippets:
            raise ValueError("knowledge_snippets cannot be empty")

        self.knowledge = knowledge_snippets
        # TfidfVectorizer with bigrams to better match queries
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 2),
        )
        self.doc_matrix = self.vectorizer.fit_transform(
            [doc["content"] for doc in self.knowledge]
        )
        self.history: Deque[ChatTurn] = deque(maxlen=max_history)
        self.filter = ResponseFilter()

    def _build_context_query(self, query: str) -> str:
        """Optionally blend short history into the query for continuity."""
        history_text = " ".join(turn.user + " " + turn.bot for turn in self.history)
        return (history_text + " " + query).strip()
    
    def _extract_key_intent(self, query: str) -> Optional[Tuple[str, float]]:
        """Extract main intent from query with confidence score."""
        q = query.lower().strip()
        
        # Remove common question words
        q_normalized = re.sub(r'^(what|how|why|when|where|which|who|can|do|does|is|are)\s+', '', q)
        q_normalized = re.sub(r'\?+$', '', q_normalized).strip()
        
        return (q_normalized, 0.8) if q_normalized else None

    def _intent_response(self, query: str) -> Optional[Dict[str, Optional[str]]]:
        """Enhanced rule-based intents for common questions with more flexibility."""
        q = query.lower()
        tokens = set(q.replace("-", " ").replace("?", "").replace("!", "").split())
        
        # Filter out common stop words for better matching
        stop_words = {"is", "are", "the", "a", "an", "do", "does", "can", "how", "what", "why", "where", "when", "you", "your", "i", "me", "my", "im"}
        meaningful_tokens = tokens - stop_words
        
        # Check for off-topic/nonsensical queries
        if not meaningful_tokens or len(meaningful_tokens) < 1:
            return {
                "answer": (
                    "I'm here to help with:\n"
                    "📊 **Datasets**: Available data sources and descriptions\n"
                    "🤖 **Methods**: TF-IDF, clustering, classification, anomaly detection\n"
                    "🔧 **Platform**: SEMIINTEL features and tools\n\n"
                    "Try asking something specific like 'What datasets do you have?' or 'How does TF-IDF work?'"
                ),
                "source": None,
                "score": 0.8,
                "link": None,
            }

        # Dataset intent - broader matching
        dataset_keywords = {"dataset", "datasets", "data", "sources", "source", "kaggle", "available"}
        if meaningful_tokens & dataset_keywords:
            dataset_titles = [k["title"] for k in self.knowledge if k.get("link")][:6]
            dataset_count = sum(1 for k in self.knowledge if k.get("link"))
            
            # Varied responses based on query specifics
            if "available" in meaningful_tokens or "have" in tokens:
                answer = (
                    f"We have {dataset_count} curated datasets covering GitHub issues, Stack Overflow, "
                    f"semiconductor manufacturing, wafer defects, IoT failures, and more.\n\n"
                    f"**Top picks:** {', '.join(dataset_titles)}. \n\n"
                    "Want details on any dataset? Just ask!"
                )
            else:
                answer = (
                    f"Yes, we have {dataset_count} datasets including:\n"
                    f"• {dataset_titles[0] if dataset_titles else 'GitHub Issues'}\n"
                    f"• {dataset_titles[1] if len(dataset_titles) > 1 else 'Stack Overflow'}\n"
                    f"• {dataset_titles[2] if len(dataset_titles) > 2 else 'Semiconductor Data'}\n"
                    f"• And {max(0, dataset_count - 3)} more...\n\n"
                    "Ask about specific ones or what you want to analyze!"
                )
            
            self.history.append(ChatTurn(user=query, bot=answer))
            return {
                "answer": answer,
                "source": "Dataset Registry",
                "score": 0.95,
                "link": None,
            }

        # Quick intent lookup with more variations
        keyword_map = {
            "tfidf": ("TF-IDF Text Analysis", 0.85),
            "tf-idf": ("TF-IDF Text Analysis", 0.85),
            "vectorizer": ("TF-IDF Text Analysis", 0.80),
            "vectorization": ("TF-IDF Text Analysis", 0.80),
            "cluster": ("Clustering & Pattern Discovery", 0.85),
            "clustering": ("Clustering & Pattern Discovery", 0.85),
            "anomaly": ("Clustering & Pattern Discovery", 0.85),
            "unsupervised": ("Clustering & Pattern Discovery", 0.80),
            "classify": ("Classification & Severity Prediction", 0.85),
            "classification": ("Classification & Severity Prediction", 0.85),
            "severity": ("Classification & Severity Prediction", 0.85),
            "predict": ("Classification & Severity Prediction", 0.80),
            "feature": ("SEMIINTEL Features", 0.80),
            "platform": ("SEMIINTEL Features", 0.85),
            "features": ("SEMIINTEL Features", 0.80),
            "tool": ("SEMIINTEL Features", 0.75),
            "osint": ("How to Use OSINT Tools", 0.85),
            "github": ("How to Use OSINT Tools", 0.80),
            "stackoverflow": ("How to Use OSINT Tools", 0.80),
            "entity": ("Entity Recognition and NLP", 0.85),
            "sentiment": ("Entity Recognition and NLP", 0.80),
            "nlp": ("Entity Recognition and NLP", 0.85),
        }

        # Try to find intent match
        for key, (title, confidence) in keyword_map.items():
            if key in q:
                doc = next((d for d in self.knowledge if d.get("title") == title), None)
                if doc:
                    answer = doc["summary"]
                    if doc.get("cta"):
                        answer = f"{answer} {doc['cta']}"
                    self.history.append(ChatTurn(user=query, bot=answer))
                    return {
                        "answer": answer,
                        "source": title,
                        "score": confidence,
                        "link": doc.get("link"),
                    }

        return None

    def ask(self, query: str, min_confidence: float = 0.50) -> Dict[str, Optional[str]]:
        """
        Ask the chatbot a question with improved response handling.
        
        Args:
            query: User's question
            min_confidence: Minimum similarity score (0-1) to provide an answer.
                           If below this, returns a fallback response.
                           Default 0.50 (50%) prevents low-confidence matches.
        
        Returns:
            Dictionary with answer, source, score, and link
        """
        if not query or not query.strip():
            return {
                "answer": "I'm ready to help! Ask me about:\n• 📊 Datasets and data sources\n• 🤖 ML techniques (TF-IDF, clustering, classification)\n• 🔧 Platform features and capabilities\n• 📚 Analysis methods and use cases",
                "source": None,
                "score": 0.0,
                "link": None,
            }

        # Try rule-based intents first for natural language queries
        intent_answer = self._intent_response(query)
        if intent_answer:
            return intent_answer

        context_query = self._build_context_query(query)
        query_vec = self.vectorizer.transform([context_query])
        scores = cosine_similarity(query_vec, self.doc_matrix).ravel()
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx]) if scores.size else 0.0
        
        # Get top 3 matches to check for guidance vs dataset
        top_3_indices = scores.argsort()[-3:][::-1]
        top_3_scores = scores[top_3_indices]

        # Check confidence threshold - increased to 0.50 to avoid weak matches
        if best_score < min_confidence:
            fallback = (
                "I'm not entirely sure about that, but I can help with:\n\n"
                "📊 **Datasets**: Ask about available data sources\n"
                "🤖 **Methods**: Learn about TF-IDF, clustering, anomaly detection\n"
                "🔧 **Platform**: Explore SEMIINTEL features\n\n"
                "Try asking: 'What datasets do you have?' or 'How does clustering work?'"
            )
            return {
                "answer": fallback,
                "source": None,
                "score": best_score,
                "link": None,
            }

        best_doc = self.knowledge[best_idx]
        
        # Check if best match is from guidance (not dataset) when confidence is moderate
        # This prevents returning dataset summaries for off-topic queries
        if best_score < 0.65:
            is_guidance = best_doc.get("title") not in [
                k.get("name") for k in self.knowledge 
                if k.get("source_url") or k.get("kaggle_id")
            ]
            if not is_guidance:
                # Best match is a dataset with moderate confidence, use fallback instead
                fallback = (
                    "I'm not entirely sure about that, but I can help with:\n\n"
                    "📊 **Datasets**: Ask about available data sources\n"
                    "🤖 **Methods**: Learn about TF-IDF, clustering, anomaly detection\n"
                    "🔧 **Platform**: Explore SEMIINTEL features\n\n"
                    "Try asking: 'What datasets do you have?' or 'How does clustering work?'"
                )
                return {
                    "answer": fallback,
                    "source": None,
                    "score": best_score,
                    "link": None,
                }
        
        answer = best_doc["summary"]
        if best_doc.get("cta"):
            answer = f"{answer}\n\n{best_doc['cta']}"
        
        # Filter answer for safety
        if not self.filter.is_safe(answer):
            answer = self.filter.sanitize(answer)

        self.history.append(ChatTurn(user=query, bot=answer))

        return {
            "answer": answer,
            "source": best_doc.get("title"),
            "score": best_score,
            "link": best_doc.get("link"),
        }

    def clear_history(self) -> None:
        self.history.clear()
    
    def get_follow_up_suggestions(self) -> List[str]:
        """Return contextual follow-up question suggestions."""
        suggestions = [
            "Which dataset would you like details on?",
            "How do you want to analyze this data?",
            "Are you looking for classification or clustering?",
            "Want to learn about a specific analysis method?",
            "Need help with SEMIINTEL features?",
        ]
        return suggestions[:3]  # Return top 3


def build_default_knowledge(datasets: List[Dict]) -> List[Dict[str, str]]:
    """Create knowledge snippets from dataset metadata and platform guidance."""
    knowledge: List[Dict[str, str]] = []

    for ds in datasets:
        title = ds.get("name", "Dataset")
        size_gb = ds.get("size_mb", 0)
        rows = ds.get("rows", 0)
        source = ds.get("source", "")
        primary_use = ds.get("primary_use", "Use case not specified")
        description = ds.get("description", "")
        license_hint = ds.get("license", "See source terms")
        link = ds.get("source_url") or ds.get("kaggle_id")

        # Build targeted content for TF-IDF matching
        # Keep dataset content focused and specific to avoid over-matching
        content_parts = [
            title,
            description,
            primary_use,
            f"dataset {title.lower()}",
        ]
        content = " ".join(content_parts).lower()

        summary = (
            f"{title}: {primary_use}. Rows: {rows:,} | Size: {size_gb/1024:.3f} GB. "
            f"Source: {source or 'Dataset provider'} | License: {license_hint}."
        )

        knowledge.append(
            {
                "title": title,
                "content": content,
                "summary": summary,
                "link": link,
                "cta": f"Access at: {link}" if link else None,
            }
        )

    # Enhanced guidance entries with better coverage
    guidance_entries = [
        {
            "title": "Available Datasets",
            "content": (
                "datasets available data sources kaggle UCI NASA github stack overflow "
                "semiconductor manufacturing wafer defect issue bug review specification "
                "benchmark machine learning training testing data analysis"
            ),
            "summary": (
                "13 curated datasets available: GitHub issues, Stack Overflow, IC performance, "
                "semiconductor manufacturing, IoT failures, hardware bugs, technical docs, "
                "electronics reviews, MCU specs, community bugs, wafer maps, SECOM, and NASA bearing data. "
                "Each dataset comes with documentation and use case examples."
            ),
            "link": None,
            "cta": None,
        },
        {
            "title": "TF-IDF Text Analysis",
            "content": (
                "TF-IDF term frequency inverse document frequency text vectorization similarity "
                "nlp natural language processing word importance ranking text mining document comparison "
                "keyword extraction feature extraction scoring"
            ),
            "summary": (
                "TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical vectors "
                "based on word importance. It's perfect for:\n"
                "• Document similarity search\n"
                "• Text classification\n"
                "• Keyword extraction\n"
                "• Issue categorization\n\n"
                "Use it to analyze GitHub issues, specification documents, or technical discussions."
            ),
            "link": None,
            "cta": None,
        },
        {
            "title": "Clustering & Pattern Discovery",
            "content": (
                "clustering KMeans DBSCAN grouping unsupervised learning pattern discovery "
                "anomaly detection outlier density-based hierarchical agglomerative clustering "
                "elbow method silhouette score group similar records"
            ),
            "summary": (
                "Clustering groups similar records together without labels:\n"
                "• **KMeans**: Creates balanced groups, good for issue categorization\n"
                "• **DBSCAN**: Finds density-based clusters, great for outlier detection\n"
                "• **Hierarchical**: Reveals relationships between records\n\n"
                "Use clustering to discover issue patterns, identify device failure modes, "
                "and find anomalies in manufacturing data."
            ),
            "link": None,
            "cta": None,
        },
        {
            "title": "Classification & Severity Prediction",
            "content": (
                "classification supervised learning severity prediction label training "
                "random forest gradient boosting classifier categorical labels "
                "train test split cross validation metrics accuracy precision recall"
            ),
            "summary": (
                "Classification assigns predefined labels (high/medium/low severity) using labeled data:\n"
                "• Train the model on historical labeled data\n"
                "• Evaluate performance with test data\n"
                "• Predict labels for new records\n\n"
                "Use classification for predicting issue severity, defect categories, "
                "hardware failure risk levels, and device health status."
            ),
            "link": None,
            "cta": None,
        },
        {
            "title": "SEMIINTEL Features",
            "content": (
                "SEMIINTEL osint github scanner stackoverflow verification ml pipeline "
                "nlp ner sentiment entity recognition anomaly detection dashboard analytics "
                "platform features tools capabilities retrieval extraction analysis"
            ),
            "summary": (
                "SEMIINTEL is an integrated semiconductor intelligence platform with:\n"
                "1. **OSINT Tools**: GitHub/StackOverflow scanners for real-time data\n"
                "2. **ML Pipeline**: Severity classifier, clustering, anomaly detection\n"
                "3. **NLP Analysis**: Named entity recognition, sentiment analysis, keywords\n"
                "4. **Curated Datasets**: 13 specialized datasets for training\n"
                "5. **Analytics**: Interactive dashboard for exploration\n\n"
                "All local processing—no API calls or cloud dependencies!"
            ),
            "link": None,
            "cta": None,
        },
        {
            "title": "How to Use OSINT Tools",
            "content": (
                "OSINT github scanner stack overflow web scraping verification keywords "
                "search pattern matching data collection real-time updates feeds monitoring"
            ),
            "summary": (
                "The OSINT tools help you gather intelligence from public sources:\n\n"
                "**GitHub Scanner**:\n"
                "• Search for repositories, issues, and discussions\n"
                "• Extract keywords and trends\n"
                "• Analyze issue patterns\n\n"
                "**Stack Overflow Search**:\n"
                "• Find common questions and issues\n"
                "• Identify problem patterns\n"
                "• Learn from community solutions\n\n"
                "All results are processed locally for privacy!"
            ),
            "link": None,
            "cta": None,
        },
        {
            "title": "Data Analysis Best Practices",
            "content": (
                "data preprocessing cleaning normalization feature engineering exploratory analysis "
                "visualization plotting statistical analysis correlation distribution validation"
            ),
            "summary": (
                "Before analyzing data, follow these best practices:\n\n"
                "1. **Exploration**: Load and inspect your data\n"
                "2. **Cleaning**: Handle missing values and outliers\n"
                "3. **Preprocessing**: Normalize and standardize features\n"
                "4. **Visualization**: Plot distributions and relationships\n"
                "5. **Feature Engineering**: Create meaningful features\n"
                "6. **Validation**: Test on held-out data\n\n"
                "SEMIINTEL's ML pipeline handles many of these steps automatically!"
            ),
            "link": None,
            "cta": None,
        },
        {
            "title": "Entity Recognition and NLP",
            "content": (
                "named entity recognition NER sentiment analysis text classification "
                "part of speech tagging dependency parsing entity extraction "
                "keywords phrases topics language processing"
            ),
            "summary": (
                "NLP features help extract meaning from text:\n\n"
                "**Named Entity Recognition (NER)**:\n"
                "• Extract person, organization, location names\n"
                "• Identify technical terms and components\n\n"
                "**Sentiment Analysis**:\n"
                "• Detect positive/negative sentiment\n"
                "• Gauge user satisfaction\n\n"
                "**Keyword Extraction**:\n"
                "• Find important terms\n"
                "• Understand main topics\n\n"
                "Use these for issue analysis, feedback understanding, and topic discovery!"
            ),
            "link": None,
            "cta": None,
        },
    ]

    knowledge.extend(guidance_entries)
    return knowledge
