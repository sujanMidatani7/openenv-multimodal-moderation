from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")

BOOST_TERMS = {
    "kill": {"violence"},
    "murder": {"violence"},
    "weapon": {"violence"},
    "nude": {"nudity"},
    "nudity": {"nudity"},
    "adult_pose": {"nudity"},
    "slur": {"harassment"},
    "dogpile": {"harassment"},
    "harassment": {"harassment"},
    "medical": {"misinfo"},
    "doctor": {"misinfo"},
    "stroke": {"misinfo"},
    "gore": {"graphic"},
}


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    title: str
    text: str
    score: float


class PolicyRetriever:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks
        self._prepared = [self._prepare_chunk(chunk) for chunk in chunks]

    def _prepare_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        combined = f"{chunk['title']} {chunk['text']}".lower()
        tokens = set(TOKEN_PATTERN.findall(combined))
        return {
            "chunk": chunk,
            "tokens": tokens,
        }

    def _tokenize(self, text: str) -> set[str]:
        return set(TOKEN_PATTERN.findall(text.lower()))

    def _boost_score(self, query_tokens: set[str], chunk_tokens: set[str]) -> float:
        boost = 0.0
        for token in query_tokens:
            categories = BOOST_TERMS.get(token)
            if not categories:
                continue
            if any(category in chunk_tokens for category in categories):
                boost += 1.5
        return boost

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        query_tokens = self._tokenize(query)
        scored: list[RetrievedChunk] = []
        for item in self._prepared:
            overlap = len(query_tokens & item["tokens"])
            boost = self._boost_score(query_tokens, item["tokens"])
            score = float(overlap) + boost
            if score <= 0:
                continue
            chunk = item["chunk"]
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    title=chunk["title"],
                    text=chunk["text"],
                    score=score,
                )
            )

        scored.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
        return scored[:top_k]
