from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    title: str
    text: str
    score: float


class PolicyRetriever:
    def __init__(
        self,
        chunks: list[dict[str, Any]],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_folder: str | None = None,
    ) -> None:
        self._chunks = chunks
        resolved_model = self._resolve_model_path(model_name)
        self._model = SentenceTransformer(
            resolved_model,
            cache_folder=cache_folder or os.getenv("SENTENCE_TRANSFORMERS_HOME"),
            local_files_only=True,
        )
        embeddings = self._embed([chunk["text"] for chunk in chunks])
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def _resolve_model_path(self, model_name: str) -> str:
        home = os.path.expanduser("~")
        snapshot_root = os.path.join(
            home,
            ".cache",
            "huggingface",
            "hub",
            "models--sentence-transformers--all-MiniLM-L6-v2",
            "snapshots",
        )
        if os.path.isdir(snapshot_root):
            snapshots = sorted(os.listdir(snapshot_root))
            if snapshots:
                return os.path.join(snapshot_root, snapshots[-1])
        return model_name

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32")

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        query_vec = self._embed([query])
        scores, indices = self._index.search(query_vec, top_k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    title=chunk["title"],
                    text=chunk["text"],
                    score=float(score),
                )
            )
        return results
