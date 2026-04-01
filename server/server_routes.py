from __future__ import annotations

from fastapi import FastAPI

from server.logic import DEFAULT_CASE_IDS
from server.env import ContentModerationEnv


def attach_server_routes(app: FastAPI, env: ContentModerationEnv) -> FastAPI:
    @app.get("/state_full")
    def state_full() -> dict:
        return env.state.model_dump()

    @app.get("/episode_summary")
    def episode_summary() -> dict:
        state = env.state
        data = state.state_data
        return {
            "episode_id": state.episode_id,
            "step_count": state.step_count,
            "done": state.done,
            "total_reward": state.total_reward,
            "current_step": state.current_step.value,
            "case_id": data.get("case_id"),
            "expected_action": data.get("expected_action"),
            "final_action": data.get("final_action"),
            "reviewer_note": data.get("reviewer_note"),
            "reward_breakdown": data.get("reward_breakdown", []),
        }

    @app.get("/cases")
    def list_cases() -> dict:
        return {"case_ids": DEFAULT_CASE_IDS}

    return app
