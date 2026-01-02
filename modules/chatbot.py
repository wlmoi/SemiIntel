"""
Lightweight retrieval-based chatbot used inside the Streamlit app.
It does not call any paid APIs; all responses come from local TF-IDF
similarity over curated semiconductor datasets and platform summaries.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ChatTurn:
    """Represents a single interaction between the user and the bot."""
    user: str
    bot: str


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

    def _build_context_query(self, query: str) -> str:
        """Optionally blend short history into the query for continuity."""
        history_text = " ".join(turn.user + " " + turn.bot for turn in self.history)
        return (history_text + " " + query).strip()

    def ask(self, query: str, min_confidence: float = 0.25) -> Dict[str, Optional[str]]:
        """
        Ask the chatbot a question.
        
        Args:
            query: User's question
            min_confidence: Minimum similarity score (0-1) to provide an answer.
                           If below this, returns a fallback response.
        
        Returns:
            Dictionary with answer, source, score, and link
        """
        if not query or not query.strip():
            return {
                "answer": "Please provide a question about datasets, ML, or OSINT.",
                "source": None,
                "score": 0.0,
                "link": None,
            }

        context_query = self._build_context_query(query)
        query_vec = self.vectorizer.transform([context_query])
        scores = cosine_similarity(query_vec, self.doc_matrix).ravel()
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx]) if scores.size else 0.0

        # Check confidence threshold
        if best_score < min_confidence:
            fallback = (
                f"I'm not sure how to answer that. I can help with:\n"
                f"• Available datasets and their descriptions\n"
                f"• Data analysis methods (TF-IDF, clustering, anomaly detection)\n"
                f"• Platform features and usage\n"
                f"Try asking: 'What datasets are available?' or 'How do I use TF-IDF?'"
            )
            return {
                "answer": fallback,
                "source": None,
                "score": best_score,
                "link": None,
            }

        best_doc = self.knowledge[best_idx]
        answer = best_doc["summary"]
        if best_doc.get("cta"):
            answer = f"{answer} {best_doc['cta']}"

        self.history.append(ChatTurn(user=query, bot=answer))

        return {
            "answer": answer,
            "source": best_doc.get("title"),
            "score": best_score,
            "link": best_doc.get("link"),
        }

    def clear_history(self) -> None:
        self.history.clear()


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

        # Build rich content for TF-IDF matching
        content_parts = [
            title,
            description,
            primary_use,
            f"rows data {rows}",
            f"size {size_gb}",
            source or "kaggle",
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

    # Add more specific guidance entries with expanded vocabulary
    guidance_entries = [
        {
            "title": "Available Datasets",
            "content": "datasets available data sources kaggle UCI NASA github stack overflow semiconductor manufacturing wafer defect issue bug review specification benchmark machine learning training",
            "summary": (
                "13 curated datasets available: GitHub issues, Stack Overflow, IC performance, "
                "semiconductor manufacturing, IoT failures, hardware bugs, technical docs, electronics reviews, "
                "MCU specs, community bugs, wafer maps, SECOM, and NASA bearing data."
            ),
            "link": None,
            "cta": None,
        },
        {
            "title": "TF-IDF Text Analysis",
            "content": "TF-IDF term frequency inverse document frequency text vectorization similarity nlp natural language processing",
            "summary": (
                "TF-IDF converts text to numerical vectors based on word importance. Use for document similarity, "
                "text classification, and keyword extraction. Great for analyzing issues and specifications."
            ),
            "link": None,
            "cta": None,
        },
        {
            "title": "Clustering & Pattern Discovery",
            "content": "clustering KMeans DBSCAN grouping unsupervised learning pattern discovery anomaly detection outlier",
            "summary": (
                "Clustering groups similar records. KMeans creates balanced groups; DBSCAN finds density-based clusters. "
                "Use for discovering issue patterns, device failure modes, and anomalies in manufacturing data."
            ),
            "link": None,
            "cta": None,
        },
        {
            "title": "Classification & Severity Prediction",
            "content": "classification supervised learning severity prediction label training random forest gradient boosting classifier categorical",
            "summary": (
                "Classification assigns labels (high/medium/low severity). Train on labeled data. "
                "Useful for predicting issue severity, defect classes, and device health status."
            ),
            "link": None,
            "cta": None,
        },
        {
            "title": "SEMIINTEL Features",
            "content": "SEMIINTEL osint github scanner stackoverflow verification ml pipeline nlp ner sentiment entity recognition anomaly detection dashboard",
            "summary": (
                "SEMIINTEL includes: (1) OSINT tools (GitHub/StackOverflow scanners), "
                "(2) ML pipeline (severity classifier, clustering, anomaly detection), "
                "(3) NLP analysis (NER, sentiment, keywords), (4) 13 curated datasets, (5) Analytics dashboard."
            ),
            "link": None,
            "cta": None,
        },
    ]

    knowledge.extend(guidance_entries)
    return knowledge
