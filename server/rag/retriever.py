from __future__ import annotations

import json
import os
from typing import Any, Dict, List


_POLICIES_PATH = os.path.join(os.path.dirname(__file__), "policies.json")

_policies_cache: List[Dict[str, Any]] = []


def _load_policies() -> List[Dict[str, Any]]:
    global _policies_cache
    if _policies_cache:
        return _policies_cache
    if os.path.exists(_POLICIES_PATH):
        with open(_POLICIES_PATH, "r", encoding="utf-8") as f:
            _policies_cache = json.load(f)
    else:
        # Fallback built-in policies when policies.json is absent
        _policies_cache = [
  {
    "chunk_id": "violence-01",
    "title": "Violence and Credible Threats",
    "text": "Content that includes direct threats, instructions to kill, or praise for murder must be removed immediately. Escalate only when context is ambiguous or safety risk cannot be resolved with the available evidence."
  },
  {
    "chunk_id": "violence-02",
    "title": "Violence Context Review",
    "text": "Fictional, quoted, or documentary references to violence may remain allowed when the speaker is clearly not endorsing harm. When intent is uncertain, flag or escalate instead of allowing outright."
  },
  {
    "chunk_id": "nudity-01",
    "title": "Adult Nudity",
    "text": "Mentions or metadata indicating nudity should be flagged for human review when the material may be adult sexual content. Remove only if the content is explicit or paired with exploitative cues."
  },
  {
    "chunk_id": "nudity-02",
    "title": "Suggestive Imagery Metadata",
    "text": "Image tags such as exposed body parts, provocative posing, or synthetic nude prompts increase risk and should bias the system toward flagging rather than allowing."
  },
  {
    "chunk_id": "self-harm-01",
    "title": "Self-Harm and Crisis",
    "text": "Content asking for methods of self-harm or celebrating imminent self-harm should be escalated or removed depending on immediacy. Support-seeking or recovery discussion should generally remain allowed."
  },
  {
    "chunk_id": "harassment-01",
    "title": "Harassment and Abuse",
    "text": "Targeted abuse, slurs, and repeated demeaning language should be flagged or removed based on severity. Benign disagreement or criticism alone is not enough for removal."
  },
  {
    "chunk_id": "spam-01",
    "title": "Spam and Low-Quality Promotion",
    "text": "Commercial spam, scam bait, or repetitive promotional text should usually be flagged. Remove only when the content is clearly malicious or part of coordinated abuse."
  },
  {
    "chunk_id": "misinfo-01",
    "title": "Misinformation Handling",
    "text": "Potential misinformation without direct physical harm usually requires flagging or escalation, not automatic removal. High-risk medical or civic deception should lean toward escalation."
  },
  {
    "chunk_id": "graphic-01",
    "title": "Graphic Media",
    "text": "Graphic injury descriptions or metadata indicating gore should be removed or escalated when public safety and age exposure concerns are high."
  },
  {
    "chunk_id": "default-01",
    "title": "Default Safe Handling",
    "text": "When evidence is weak and no policy trigger is present, prefer allowing the content. Use escalation only when confidence is low or policy signals conflict."
  }
]
    return _policies_cache


def _keyword_score(text: str, policy_text: str) -> float:
    """Simple overlap score: fraction of content words found in policy text."""
    content_words = set(text.lower().split())
    policy_words = set(policy_text.lower().split())
    if not content_words:
        return 0.0
    return len(content_words & policy_words) / len(content_words)


def retrieve_policy_chunks(query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Return top_k policy chunks most relevant to query_text."""
    policies = _load_policies()
    scored = [
        {
            "policy_id": p["chunk_id"],
            "text": p["text"],
            "score": _keyword_score(query_text, p["text"]),
        }
        for p in policies
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]