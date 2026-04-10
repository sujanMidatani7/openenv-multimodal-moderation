from __future__ import annotations

import contextlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
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
        self._cache_folder = cache_folder or os.getenv("SENTENCE_TRANSFORMERS_HOME")
        resolved_model = self._resolve_model_path(model_name)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self._model = SentenceTransformer(
                    resolved_model,
                    cache_folder=self._cache_folder,
                    local_files_only=True,
                )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load sentence-transformers/all-MiniLM-L6-v2 from local cache. "
                "Ensure the model is preloaded in the image or available under "
                "SENTENCE_TRANSFORMERS_HOME / HF cache before runtime."
            ) from exc
        embeddings = self._embed([chunk["text"] for chunk in chunks])
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    def _resolve_model_path(self, model_name: str) -> str:
        for root in self._candidate_cache_roots():
            snapshot_root = root / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
            if snapshot_root.is_dir():
                snapshots = sorted(path for path in snapshot_root.iterdir() if path.is_dir())
                if snapshots:
                    return str(snapshots[-1])
        return model_name

    def _candidate_cache_roots(self) -> list[Path]:
        roots: list[Path] = []
        env_candidates = [
            self._cache_folder,
            os.getenv("HF_HUB_CACHE"),
            os.getenv("HUGGINGFACE_HUB_CACHE"),
            os.getenv("HF_HOME"),
        ]
        for candidate in env_candidates:
            if not candidate:
                continue
            candidate_path = Path(candidate)
            if candidate_path.name == "hub":
                roots.append(candidate_path)
            else:
                roots.append(candidate_path / "hub")
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")

        deduped: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            key = str(root.resolve()) if root.exists() else str(root)
            if key not in seen:
                seen.add(key)
                deduped.append(root)
        return deduped

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
