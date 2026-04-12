from __future__ import annotations

import traceback
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from .models import Action, Observation, State
    from .env import ModerationEnvironment
    from .logic import CASE_IDS
except ImportError:
    from models import Action, Observation, State
    from env import ModerationEnvironment
    from logic import CASE_IDS


# ---------------------------------------------------------------------------
# Single persistent environment — shared across ALL HTTP requests
# ---------------------------------------------------------------------------
_env = ModerationEnvironment()

app = FastAPI(
    title="OpenEnv Multimodal Moderation",
    description="Multimodal content moderation RL environment",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ResetOptions(BaseModel):
    case_id: Optional[str] = None
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class ResetRequest(BaseModel):
    options: Optional[ResetOptions] = None


class StepRequest(BaseModel):
    action: Action


# ---------------------------------------------------------------------------
# Core OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
async def reset(req: Optional[ResetRequest] = None) -> JSONResponse:
    try:
        opts = (req.options if req and req.options else None) or ResetOptions()
        obs: Observation = _env.reset(
            seed=opts.seed,
            episode_id=opts.episode_id,
            case_id=opts.case_id or "",
        )
        return JSONResponse({
            "observation": obs.model_dump(mode="json"),
            "reward": 0.0,
            "done": False,
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(req: StepRequest) -> JSONResponse:
    try:
        obs: Observation = _env.step(req.action)
        return JSONResponse({
            "observation": obs.model_dump(mode="json"),
            "reward": obs.reward,
            "done": obs.done,
        })
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state() -> JSONResponse:
    return JSONResponse(_env.state.model_dump(mode="json"))


@app.get("/schema")
async def schema() -> JSONResponse:
    return JSONResponse({
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": State.model_json_schema(),
    })


# ---------------------------------------------------------------------------
# /episode_summary — read by the reward_threshold graders in openenv.yaml
# ---------------------------------------------------------------------------

@app.get("/episode_summary")
async def episode_summary() -> JSONResponse:
    state = _env.state
    breakdown = dict(state.reward_breakdown or {})
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


# ---------------------------------------------------------------------------
# Helper endpoints
# ---------------------------------------------------------------------------

@app.get("/cases")
async def list_cases() -> JSONResponse:
    return JSONResponse({"cases": CASE_IDS})


@app.get("/state_full")
async def state_full() -> JSONResponse:
    return JSONResponse(_env.state.model_dump(mode="json"))


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})

def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
