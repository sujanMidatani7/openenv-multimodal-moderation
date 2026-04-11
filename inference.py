from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict
from typing import Any, Dict, List, Optional
import requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://sujanmidatani-openenv-multimodal-moderation.hf.space").rstrip("/")
TASK_NAME    = os.environ.get("TASK_NAME", "violence-remove")
BENCHMARK    = "openenv-multimodal-moderation"
 
VALID_ACTIONS = ["allow", "flag", "remove", "escalate"]
EPISODE_STEPS = ["analyze", "retrieve_policy", "decide", "review", "finalize"]

MAX_EPISODE_REWARD = 1.6
 

_extra: Dict[str, str] = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
 
client = OpenAI(
    base_url=API_BASE_URL or None,
    api_key=HF_TOKEN or "placeholder",
    default_headers=_extra or None,
)
 
SYSTEM_PROMPT = (
    "You are an expert content moderation AI. "
    "At each step you receive the current content and must respond with a JSON object: "
    '{"action_type": "<allow|flag|remove|escalate>", "reason": "<your reasoning>"}. '
    "Steps: analyze → retrieve_policy → decide → review → finalize."
)
 
 
# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
 
def call_reset(case_id: str = "") -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    if case_id:
        body["options"] = {"case_id": case_id}
    resp = requests.post(f"{ENV_BASE_URL}/reset", json=body, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    return raw.get("observation", raw)
 
 
def call_step(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    if "observation" in raw:
        flat = dict(raw["observation"])
        flat["reward"] = raw.get("reward", 0.0)
        flat["done"]   = raw.get("done", False)
        return flat
    return raw
 
 
def call_get(path: str) -> Dict[str, Any]:
    resp = requests.get(f"{ENV_BASE_URL}{path}", timeout=30)
    resp.raise_for_status()
    return resp.json()
 
 
# ---------------------------------------------------------------------------
# Model helper
# ---------------------------------------------------------------------------
 
def ask_model(messages: list) -> Dict[str, Any]:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=256,
        temperature=0.2,
    )
    raw = completion.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"action_type": "flag", "reason": raw}
 
 
# ---------------------------------------------------------------------------
# Logging helpers — exact format required by the benchmark
# ---------------------------------------------------------------------------
 
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)
 
 
def log_step(step_n: int, action_str: str, reward: float, done: bool,
             error: Optional[str] = None) -> None:
    error_field = error if error else "null"
    done_field  = "true" if done else "false"
    print(
        f"[STEP] step={step_n} action={action_str} "
        f"reward={reward:.2f} done={done_field} error={error_field}",
        flush=True,
    )
 
 
def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_field = "true" if success else "false"
    print(
        f"[END] success={success_field} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )
 
 
# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------
 
def main() -> None:
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
 
    step_n    = 0
    rewards: List[float] = []
    success   = False
    last_error: Optional[str] = None
 
    try:
        # --- reset ---
        obs = call_reset(case_id=TASK_NAME)
 
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
 
        for step_name in EPISODE_STEPS:
            step_n += 1
 
            # Build prompt from current observation
            user_content = (
                f"Step: {step_name}\n"
                f"Content: {json.dumps(obs.get('content', {}))}\n"
                f"Policy: {json.dumps(obs.get('policy', []))}\n"
                f"Message: {obs.get('message', '')}"
            )
            messages.append({"role": "user", "content": user_content})
 
            # Ask model
            action = ask_model(messages)
            if action.get("action_type") not in VALID_ACTIONS:
                action["action_type"] = "flag"
            messages.append({"role": "assistant", "content": json.dumps(action)})
 
            action_str = f"{action['action_type']}('{action.get('reason', '')[:60]}')"
 
            # Step environment
            try:
                obs = call_step(action)
                step_reward = float(obs.get("reward", 0.0))
                done        = bool(obs.get("done", False))
                last_error  = None
            except Exception as exc:
                step_reward = 0.0
                done        = True
                last_error  = str(exc)
 
            rewards.append(step_reward)
            log_step(step_n, action_str, step_reward, done, last_error)
 
            if done:
                break
 
        # --- episode summary ---
        try:
            summary     = call_get("/episode_summary")
            total_reward = float(summary.get("total_reward", sum(rewards)))
            final_action = summary.get("final_action") or ""
        except Exception:
            total_reward = sum(rewards)
            final_action = action.get("action_type", "")  # type: ignore[possibly-undefined]
 
        # score in [0, 1]
        score   = max(0.0, min(1.0, total_reward / MAX_EPISODE_REWARD))
        success = score >= 0.5
 
    except Exception as exc:
        last_error = str(exc)
        # Ensure rewards list has at least step_n entries
        while len(rewards) < step_n:
            rewards.append(0.0)
        score   = 0.0
        success = False
 
    log_end(success=success, steps=step_n, score=score, rewards=rewards)
 
 
if __name__ == "__main__":
    main()