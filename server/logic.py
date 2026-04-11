from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Built-in moderation cases
# Each case maps to one of the five tasks declared in openenv.yaml
# ---------------------------------------------------------------------------

MODERATION_CASES: Dict[str, Dict[str, Any]] = {
    "violence-remove": {
        "id": "violence-remove",
        "content": {
            "text": "Graphic depiction of someone being murdered with explicit details about how to kill another person.",
            "image_description": "Violent scene showing physical assault",
        },
        "expected_action": "remove",
        "reviewer_recommendation": "remove",
        "risk_keywords": ["kill", "murder"],
    },
    "nudity-flag": {
        "id": "nudity-flag",
        "content": {
            "text": "This post contains nude imagery that may violate platform policy.",
            "image_description": "Image containing nudity",
        },
        "expected_action": "flag",
        "reviewer_recommendation": "flag",
        "risk_keywords": ["nude"],
    },
    "harassment-flag": {
        "id": "harassment-flag",
        "content": {
            "text": "Repeated targeted messages insulting and demeaning a specific user.",
            "image_description": "Screenshot of harassment campaign",
        },
        "expected_action": "flag",
        "reviewer_recommendation": "flag",
        "risk_keywords": [],
    },
    "misinfo-escalate": {
        "id": "misinfo-escalate",
        "content": {
            "text": "False health claims stating that vaccines cause autism, shared widely.",
            "image_description": "Infographic with misleading statistics",
        },
        "expected_action": "escalate",
        "reviewer_recommendation": "escalate",
        "risk_keywords": [],
    },
    "benign-allow": {
        "id": "benign-allow",
        "content": {
            "text": "A photo of a sunset at the beach with a caption about enjoying the weekend.",
            "image_description": "Landscape photo, beach sunset",
        },
        "expected_action": "allow",
        "reviewer_recommendation": "allow",
        "risk_keywords": [],
    },
}

CASE_IDS: List[str] = list(MODERATION_CASES.keys())

# ---------------------------------------------------------------------------
# Reward constants (match openenv.yaml rewards section)
# ---------------------------------------------------------------------------
REWARD_ANALYSIS_STEP = 0.2
REWARD_RETRIEVAL_STEP = 0.2
REWARD_CORRECT_DECISION = 1.0
REWARD_REVIEWER_AGREEMENT = 0.2
PENALTY_UNSAFE_ALLOW = -0.6


def get_case(case_id: str) -> Optional[Dict[str, Any]]:
    return MODERATION_CASES.get(case_id)


def get_expected_action(case: Dict[str, Any]) -> str:
    """Apply rule overrides then return expected action."""
    text = case["content"].get("text", "").lower()
    if any(kw in text for kw in ["kill", "murder"]):
        return "remove"
    if "nude" in text:
        return "flag"
    return case["expected_action"]


def is_risky_content(case: Dict[str, Any]) -> bool:
    text = case["content"].get("text", "").lower()
    return any(kw in text for kw in ["kill", "murder", "nude", "harass", "violence", "assault"])


def compute_step_reward(
    step_name: str,
    action_type: Optional[str],
    case: Dict[str, Any],
) -> float:
    """Return the reward delta for a single step."""
    if step_name == "analyze":
        return REWARD_ANALYSIS_STEP
    if step_name == "retrieve_policy":
        return REWARD_RETRIEVAL_STEP
    if step_name == "decide":
        expected = get_expected_action(case)
        if action_type == expected:
            return REWARD_CORRECT_DECISION
        if action_type == "allow" and is_risky_content(case):
            return PENALTY_UNSAFE_ALLOW
        return 0.0
    if step_name == "review":
        reviewer_rec = case.get("reviewer_recommendation", "")
        if action_type == reviewer_rec:
            return REWARD_REVIEWER_AGREEMENT
        return 0.0
    # finalize — no additional reward
    return 0.0