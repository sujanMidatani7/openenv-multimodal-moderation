from __future__ import annotations

from dataclasses import dataclass

import httpx

from models import Action, Observation, State


@dataclass
class StepResult:
    observation: Observation
    reward: float | None
    done: bool


class ModerationEnvClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reset(self, **kwargs) -> StepResult:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            response = client.post("/reset", json=kwargs)
            response.raise_for_status()
            payload = response.json()
        return StepResult(
            observation=Observation(**payload["observation"]),
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def step(self, action: Action) -> StepResult:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            response = client.post("/step", json={"action": action.model_dump(exclude_none=True)})
            response.raise_for_status()
            payload = response.json()
        return StepResult(
            observation=Observation(**payload["observation"]),
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def state(self) -> State:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            response = client.get("/state")
            response.raise_for_status()
            payload = response.json()
        return State(**payload)
