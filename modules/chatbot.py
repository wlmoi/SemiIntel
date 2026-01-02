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
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(
            [doc["content"] for doc in self.knowledge]
        )
        self.history: Deque[ChatTurn] = deque(maxlen=max_history)

    def _build_context_query(self, query: str) -> str:
        """Optionally blend short history into the query for continuity."""
        history_text = " ".join(turn.user + " " + turn.bot for turn in self.history)
        return (history_text + " " + query).strip()

    def ask(self, query: str) -> Dict[str, Optional[str]]:
        if not query or not query.strip():
            return {
                "answer": "Please provide a question about datasets, ML, or OSINT.",
                "source": None,
                "score": 0.0,
            }

        context_query = self._build_context_query(query)
        query_vec = self.vectorizer.transform([context_query])
        scores = cosine_similarity(query_vec, self.doc_matrix).ravel()
        best_idx = int(scores.argmax())
        best_doc = self.knowledge[best_idx]
        best_score = float(scores[best_idx]) if scores.size else 0.0

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
        license_hint = ds.get("license", "See source terms")
        link = ds.get("source_url") or ds.get("kaggle_id")

        summary = (
            f"{title}: {primary_use}. Rows: {rows:,} | Size: {size_gb/1024:.3f} GB. "
            f"Source: {source or 'Dataset provider'} | License: {license_hint}."
        )

        knowledge.append(
            {
                "title": title,
                "content": f"{title} {primary_use} {rows} rows {size_gb:.2f} GB {source}",
                "summary": summary,
                "link": link,
                "cta": f"Download/source: {link}" if link else None,
            }
        )

    knowledge.append(
        {
            "title": "Platform Guidance",
            "content": "Use TF-IDF, clustering, anomaly detection, and OSINT scanners",
            "summary": (
                "For quick analysis, start with TF-IDF similarity, run clustering for patterns, "
                "and use the OSINT tools for GitHub/StackOverflow enrichment."
            ),
            "link": None,
            "cta": None,
        }
    )

    return knowledge
