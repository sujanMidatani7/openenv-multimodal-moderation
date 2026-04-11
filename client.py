from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import Action, Observation, State
except ImportError:
    from models import Action, Observation, State


class ModerationEnv(EnvClient[Action, Observation, State]):

    def _step_payload(self, action: Action) -> dict[str, Any]:
        return action.model_dump(mode="json")

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[Observation]:
        observation_payload = payload.get("observation", {})
        return StepResult(
            observation=Observation(**observation_payload),
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        return State(**payload)


ModerationEnvAction = Action
ModerationEnvObservation = Observation
ModerationEnvState = State

__all__ = [
    "ModerationEnv",
    "ModerationEnvAction",
    "ModerationEnvObservation",
    "ModerationEnvState",
]