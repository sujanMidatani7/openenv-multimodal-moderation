from __future__ import annotations

from enum import Enum
from typing import Any

from openenv.core.env_server.types import Action as OpenEnvAction
from openenv.core.env_server.types import Observation as OpenEnvObservation
from openenv.core.env_server.types import State as OpenEnvState
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    ALLOW = "allow"
    FLAG = "flag"
    REMOVE = "remove"
    ESCALATE = "escalate"


class StepType(str, Enum):
    ANALYZE = "analyze"
    RETRIEVE_POLICY = "retrieve_policy"
    DECIDE = "decide"
    REVIEW = "review"
    FINALIZE = "finalize"


class Content(BaseModel):
    text: str = Field(..., description="User-facing text content under review.")
    image_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Image metadata and derived signals available to the agent.",
    )


class PolicyChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier.")
    title: str = Field(..., description="Policy section title.")
    text: str = Field(..., description="Retrieved policy passage.")
    score: float = Field(..., description="Similarity score for this chunk.")


class Action(OpenEnvAction):
    action_type: ActionType = Field(..., description="Moderation action to take.")
    reason: str = Field(..., min_length=8, description="Reason for the action.")


class Observation(OpenEnvObservation):
    content: Content = Field(..., description="Content payload available at this step.")
    policy: list[PolicyChunk] = Field(
        default_factory=list,
        description="Retrieved policy chunks relevant to the current case.",
    )
    step_type: StepType = Field(..., description="Current workflow step.")
    step_count: int = Field(..., ge=0, description="Number of actions taken so far.")


class State(OpenEnvState):
    done: bool = Field(default=False, description="Whether the episode has finished.")
    current_step: StepType = Field(
        default=StepType.ANALYZE,
        description="Current step in the moderation workflow.",
    )
    state_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Persistent state accumulated across the episode.",
    )
    total_reward: float = Field(default=0.0, description="Cumulative episode reward.")
