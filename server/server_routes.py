from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


def register_routes(app, env) -> None:
    """Call this from app.py after creating the FastAPI app."""

    @app.get("/episode_summary")
    async def episode_summary() -> JSONResponse:
        state = env.state
        breakdown = state.reward_breakdown or {}
        total_reward = max(0.01, min(0.99, float(sum(breakdown.values()))))
        return JSONResponse({
            "episode_id": state.episode_id,
            "step_count": state.step_count,
            "done": state.done,
            "total_reward": total_reward,
            "reward_breakdown": breakdown,
            "final_action": state.final_action,
            "reviewer_note": state.reviewer_note,
        })

    @app.get("/cases")
    async def list_cases() -> JSONResponse:
        try:
            from .logic import CASE_IDS
        except ImportError:
            from logic import CASE_IDS
        return JSONResponse({"cases": CASE_IDS})

    @app.get("/state_full")
    async def state_full() -> JSONResponse:
        return JSONResponse(env.state.model_dump(mode="json"))