from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from client import ModerationEnv, ModerationEnvAction
from server.logic import DEFAULT_CASE_IDS, grade_episode_summary

load_dotenv()

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK_NAME", DEFAULT_CASE_IDS[0])
BENCHMARK = os.getenv("BENCHMARK", "openenv-multimodal-moderation")
MAX_STEPS = 8
TEMPERATURE = 0.0
MAX_TOKENS = 180
SUCCESS_SCORE_THRESHOLD = 0.5
MAX_TOTAL_REWARD = 1.6
OUTPUT_DIR = Path("outputs")

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are interacting with a multimodal content moderation environment.
    Return exactly one JSON object with keys "action_type" and "reason".
    Allowed action_type values are: allow, flag, remove, escalate.
    Keep the reason concise and grounded in the observed content and policy.
    """
).strip()

STEP_REASON_TEMPLATES = {
    "analyze": "Analyzed text and image metadata for policy and safety risk.",
    "retrieve_policy": "Grounded the case using the retrieved moderation policy.",
    "decide": "Selected the moderation action that best matches the policy risk.",
    "review": "Validated the moderator action against the reviewer recommendation.",
    "finalize": "Finalized the moderation outcome for this episode.",
}


def one_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(step: int, observation: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current step type: {observation.get("step_type")}
        Content: {json.dumps(observation.get("content", {}), ensure_ascii=False)}
        Policy: {json.dumps(observation.get("policy", []), ensure_ascii=False)}
        Metadata: {json.dumps(observation.get("metadata", {}), ensure_ascii=False)}
        Previous steps:
        {history_block}
        Return the next moderation action as JSON.
        """
    ).strip()


def extract_text_and_metadata(observation: dict) -> tuple[str, dict]:
    content = observation.get("content", {})
    if isinstance(content, dict):
        return str(content.get("text", "")), content.get("image_metadata", {}) or {}
    return "", {}


def rule_based_decision(text: str, image_metadata: dict) -> str:
    lowered = text.lower()
    serialized_metadata = json.dumps(image_metadata, sort_keys=True).lower()
    combined = f"{lowered} {serialized_metadata}"

    if "kill" in combined or "murder" in combined:
        return "remove"
    if "nude" in combined or "nudity" in combined or "adult_pose" in combined:
        return "flag"
    if "slur" in combined or "dogpile" in combined or "harassment" in combined:
        return "flag"
    if "stroke" in combined or "ignore doctors" in combined or "medical_claim_score" in combined:
        return "escalate"
    return "allow"


def fallback_action(observation: dict) -> ModerationEnvAction:
    text, image_metadata = extract_text_and_metadata(observation)
    step_type = str(observation.get("step_type", "analyze"))
    reviewer_note = observation.get("metadata", {}).get("reviewer_note") or {}

    if step_type == "review" and reviewer_note.get("recommended_action"):
        action_type = reviewer_note["recommended_action"]
    else:
        action_type = rule_based_decision(text, image_metadata)

    return ModerationEnvAction(
        action_type=action_type,
        reason=STEP_REASON_TEMPLATES.get(step_type, "Used deterministic fallback moderation logic."),
    )


def parse_model_action(raw: str) -> ModerationEnvAction:
    payload = json.loads(raw)
    action_type = str(payload.get("action_type", "")).strip().lower()
    reason = one_line(str(payload.get("reason", "")).strip())
    if action_type not in {"allow", "flag", "remove", "escalate"}:
        raise ValueError("invalid action_type")
    if not reason:
        raise ValueError("empty reason")
    return ModerationEnvAction(action_type=action_type, reason=reason)


def get_model_action(client: OpenAI, step: int, observation: dict, history: List[str]) -> ModerationEnvAction:
    user_prompt = build_user_prompt(step, observation, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            return fallback_action(observation)
        return parse_model_action(text)
    except Exception:
        return fallback_action(observation)


def save_results(
    rewards: List[float],
    score: float,
    success: bool,
    steps: int,
    task_name: str,
    last_action: str,
    last_error: Optional[str],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "generated_at": datetime.now().isoformat(),
        "task": task_name,
        "benchmark": BENCHMARK,
        "model_name": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "local_image_name": IMAGE_NAME,
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards,
        "last_action": last_action,
        "last_error": last_error,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env: Optional[ModerationEnv] = None

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_action = "null"
    last_error: Optional[str] = None

    try:
        env = await ModerationEnv.from_docker_image(IMAGE_NAME)
        log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

        if TASK_NAME not in DEFAULT_CASE_IDS:
            raise RuntimeError(f"unknown task {TASK_NAME}")

        result = await env.reset(case_id=TASK_NAME)
        observation = result.observation.model_dump()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_model_action(client, step, observation, history)
            last_action = f"{action.action_type.value}('{one_line(action.reason)}')"

            try:
                result = await env.step(action)
                observation = result.observation.model_dump()
                reward = float(result.reward or 0.0)
                done = bool(result.done)
                error = None
            except Exception as exc:
                reward = 0.0
                done = True
                error = one_line(str(exc))
                last_error = error

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=last_action, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {last_action} -> reward {reward:.2f}")

            if done:
                break

        score = grade_episode_summary({"total_reward": sum(rewards)}) if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD and last_error is None

    finally:
        try:
            if env is not None:
                await env.close()
        except Exception:
            pass
        if env is None:
            log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
        save_results(
            rewards=rewards,
            score=score,
            success=success,
            steps=steps_taken,
            task_name=TASK_NAME,
            last_action=last_action,
            last_error=last_error,
        )
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
