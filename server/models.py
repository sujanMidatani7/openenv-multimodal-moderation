from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    allow = "allow"
    flag = "flag"
    remove = "remove"
    escalate = "escalate"


class StepType(str, Enum):
    analyze = "analyze"
    retrieve_policy = "retrieve_policy"
    decide = "decide"
    review = "review"
    finalize = "finalize"


class Content(BaseModel):
    text: str = ""
    image_url: Optional[str] = None
    image_description: Optional[str] = None


class PolicyChunk(BaseModel):
    policy_id: str = ""
    text: str = ""
    score: float = 0.0


class Action(BaseModel):
    action_type: ActionType
    reason: str = ""


class Observation(BaseModel):
    content: Optional[Content] = None
    policy: List[PolicyChunk] = Field(default_factory=list)
    step_type: StepType = StepType.analyze
    step_count: int = 0
    message: str = ""
    reward: float = 0.0
    done: bool = False


class State(BaseModel):
    episode_id: str = ""
    step_count: int = 0
    done: bool = False
    selected_case_id: Optional[str] = None
    reward_breakdown: Dict[str, float] = Field(
        default_factory=lambda: {
            "analysis_step": 0.0,
            "retrieval_step": 0.0,
            "correct_decision": 0.0,
            "reviewer_agreement": 0.0,
            "unsafe_penalty": 0.0,
        }
    )
    final_action: Optional[str] = None
    reviewer_note: Optional[str] = None
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_policy_chunks: List[PolicyChunk] = Field(default_factory=list)