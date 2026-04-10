from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from server.models import ActionType, PolicyChunk, StepType
from server.rag import POLICY_CHUNKS, PolicyRetriever


STEP_SEQUENCE = [
    StepType.ANALYZE,
    StepType.RETRIEVE_POLICY,
    StepType.DECIDE,
    StepType.REVIEW,
    StepType.FINALIZE,
]

RULE_REMOVE_TERMS = ("kill", "murder")
RULE_FLAG_TERMS = ("nude",)


@dataclass(frozen=True)
class ModerationCase:
    case_id: str
    content_text: str
    image_metadata: dict[str, Any]
    expected_action: ActionType
    reviewer_action: ActionType
    policy_hint: str
    severity: str


@dataclass
class EpisodeContext:
    episode_id: str
    case: ModerationCase
    current_step: StepType = StepType.ANALYZE
    step_count: int = 0
    done: bool = False
    total_reward: float = 0.0
    state_data: dict[str, Any] = field(default_factory=dict)


CASES: list[ModerationCase] = [
    ModerationCase(
        case_id="violence-remove",
        content_text=(
            "The post says: 'Find him tonight and murder him before he talks.' "
            "The attached image metadata includes weapon_focus_score=0.91."
        ),
        image_metadata={"weapon_focus_score": 0.91, "nudity_score": 0.01, "scene": "alley"},
        expected_action=ActionType.REMOVE,
        reviewer_action=ActionType.REMOVE,
        policy_hint="credible threat and murder instruction",
        severity="high",
    ),
    ModerationCase(
        case_id="nudity-flag",
        content_text=(
            "Promo text asks viewers to unlock a nude behind-the-scenes gallery. "
            "Metadata tags show adult_pose=True and synthetic_prompt='studio nude portrait'."
        ),
        image_metadata={"adult_pose": True, "synthetic_prompt": "studio nude portrait", "nudity_score": 0.82},
        expected_action=ActionType.FLAG,
        reviewer_action=ActionType.FLAG,
        policy_hint="adult nudity",
        severity="medium",
    ),
    ModerationCase(
        case_id="harassment-flag",
        content_text=(
            "A thread targets a creator with repeated insults and slurs, telling followers to dogpile them."
        ),
        image_metadata={"harassment_cluster_score": 0.76, "nudity_score": 0.0},
        expected_action=ActionType.FLAG,
        reviewer_action=ActionType.FLAG,
        policy_hint="targeted abuse",
        severity="medium",
    ),
    ModerationCase(
        case_id="misinfo-escalate",
        content_text=(
            "The post claims a homemade tonic cures stroke in minutes and tells users to ignore doctors."
        ),
        image_metadata={"medical_claim_score": 0.88, "citation_present": False},
        expected_action=ActionType.ESCALATE,
        reviewer_action=ActionType.ESCALATE,
        policy_hint="high-risk medical misinformation",
        severity="high",
    ),
    ModerationCase(
        case_id="benign-allow",
        content_text=(
            "A travel blogger shares sunrise photos and says the trail was quiet and beautiful."
        ),
        image_metadata={"camera_make": "Canon", "nudity_score": 0.0, "violence_score": 0.0},
        expected_action=ActionType.ALLOW,
        reviewer_action=ActionType.ALLOW,
        policy_hint="default safe handling",
        severity="low",
    ),
]

DEFAULT_CASE_IDS = [case.case_id for case in CASES]
MAX_EPISODE_REWARD = 1.6

TASKS = {
    case.case_id: {
        "task_name": case.case_id,
        "benchmark": "openenv-multimodal-moderation",
        "policy_hint": case.policy_hint,
        "severity": case.severity,
        "expected_action": case.expected_action.value,
    }
    for case in CASES
}


def build_retriever() -> PolicyRetriever:
    return PolicyRetriever(POLICY_CHUNKS)


def select_case(seed: int | None = None, case_id: str | None = None) -> ModerationCase:
    if case_id:
        for case in CASES:
            if case.case_id == case_id:
                return case
        raise ValueError(f"Unknown case_id '{case_id}'. Available: {DEFAULT_CASE_IDS}")
    rng = random.Random(seed)
    return rng.choice(CASES)


def new_episode(seed: int | None = None, episode_id: str | None = None, case_id: str | None = None) -> EpisodeContext:
    selected = select_case(seed=seed, case_id=case_id)
    return EpisodeContext(episode_id=episode_id or str(uuid4()), case=selected)


def infer_rule_override(text: str) -> ActionType | None:
    lowered = text.lower()
    if any(token in lowered for token in RULE_REMOVE_TERMS):
        return ActionType.REMOVE
    if any(token in lowered for token in RULE_FLAG_TERMS):
        return ActionType.FLAG
    return None


def build_query(case: ModerationCase, analysis_reason: str | None = None) -> str:
    parts = [case.content_text, case.policy_hint]
    if analysis_reason:
        parts.append(analysis_reason)
    if case.image_metadata:
        metadata_bits = [f"{key}={value}" for key, value in case.image_metadata.items()]
        parts.append(" ".join(metadata_bits))
    return "\n".join(parts)


def convert_chunks(chunks: list[Any]) -> list[PolicyChunk]:
    return [
        PolicyChunk(chunk_id=chunk.chunk_id, title=chunk.title, text=chunk.text, score=chunk.score)
        for chunk in chunks
    ]


def reviewer_recommendation(case: ModerationCase, moderator_action: ActionType) -> tuple[ActionType, bool, str]:
    recommended = case.reviewer_action
    agrees = moderator_action == recommended
    message = (
        f"Reviewer {'agrees' if agrees else 'disagrees'} with moderator. "
        f"Recommended action: {recommended.value}."
    )
    return recommended, agrees, message


def contains_policy_reference(reason: str, policy: list[PolicyChunk]) -> bool:
    lowered = reason.lower()
    for chunk in policy:
        title_tokens = re.findall(r"[a-z]+", chunk.title.lower())
        if any(token in lowered for token in title_tokens):
            return True
    return False


def compute_step_reward(
    step_type: StepType,
    action_type: ActionType,
    reason: str,
    case: ModerationCase,
    policy: list[PolicyChunk],
    reviewer_action_value: ActionType | None = None,
) -> tuple[float, dict[str, float | str], str]:
    expected = infer_rule_override(case.content_text) or case.expected_action
    unsafe_penalty = 0.0
    if expected in {ActionType.REMOVE, ActionType.FLAG, ActionType.ESCALATE} and action_type == ActionType.ALLOW:
        unsafe_penalty -= 0.6

    if step_type == StepType.ANALYZE:
        base_reward = 0.2 if len(reason.strip()) >= 20 else 0.05
        reward = base_reward + unsafe_penalty
        return reward, {
            "analysis_step_reward": base_reward,
            "unsafe_penalty": unsafe_penalty,
            "expected_action": expected.value,
        }, "Analysis quality scored."

    if step_type == StepType.RETRIEVE_POLICY:
        base_reward = 0.2 if contains_policy_reference(reason, policy) else 0.1
        reward = base_reward + unsafe_penalty
        return reward, {
            "retrieval_step_reward": base_reward,
            "unsafe_penalty": unsafe_penalty,
            "expected_action": expected.value,
        }, "Policy grounding scored."

    if step_type == StepType.DECIDE:
        decision_reward = 1.0 if action_type == expected else 0.0
        reward = decision_reward + unsafe_penalty
        return reward, {
            "correct_decision_reward": decision_reward,
            "unsafe_penalty": unsafe_penalty,
            "expected_action": expected.value,
        }, "Decision accuracy scored."

    if step_type == StepType.REVIEW:
        reviewer_bonus = 0.2 if reviewer_action_value is not None and action_type == reviewer_action_value else 0.0
        reward = reviewer_bonus + unsafe_penalty
        return reward, {
            "reviewer_agreement_reward": reviewer_bonus,
            "unsafe_penalty": unsafe_penalty,
            "expected_action": expected.value,
        }, "Reviewer alignment scored."

    final_bonus = 0.1 if action_type == expected else 0.0
    reward = final_bonus + unsafe_penalty
    return reward, {
        "finalization_reward": final_bonus,
        "unsafe_penalty": unsafe_penalty,
        "expected_action": expected.value,
    }, "Finalization scored."


def next_step(step_type: StepType) -> StepType | None:
    idx = STEP_SEQUENCE.index(step_type)
    if idx + 1 >= len(STEP_SEQUENCE):
        return None
    return STEP_SEQUENCE[idx + 1]


def enumerate_tasks() -> list[dict[str, str]]:
    return [TASKS[task_id] for task_id in DEFAULT_CASE_IDS]


def grade_episode_summary(summary: dict[str, Any]) -> float:
    total_reward = float(summary.get("total_reward", 0.0))
    score = total_reward / MAX_EPISODE_REWARD if MAX_EPISODE_REWARD > 0 else 0.0
    return min(max(score, 0.0), 1.0)


def run_grader_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for case in CASES:
        ideal_total = MAX_EPISODE_REWARD
        ideal_score = grade_episode_summary({"total_reward": ideal_total})
        low_score = grade_episode_summary({"total_reward": -1.0})
        checks.append(
            {
                "task": case.case_id,
                "reward_range_ok": 0.0 <= ideal_total,
                "ideal_score": ideal_score,
                "low_score": low_score,
                "score_range_ok": 0.0 <= ideal_score <= 1.0 and 0.0 <= low_score <= 1.0,
            }
        )
    return checks
