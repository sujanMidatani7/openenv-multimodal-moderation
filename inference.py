from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI

from server.logic import DEFAULT_CASE_IDS
from dotenv import load_dotenv
load_dotenv()

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")


@dataclass
class EpisodeResult:
    case_id: str
    total_reward: float
    final_action: str | None
    expected_action: str | None


def llm_action(client: OpenAI, observation: dict[str, Any]) -> dict[str, str]:
    prompt = {
        "task": "Return a JSON object with keys action_type and reason.",
        "allowed_actions": ["allow", "flag", "remove", "escalate"],
        "current_step": observation["step_type"],
        "content": observation["content"],
        "policy": observation.get("policy", []),
        "metadata": observation.get("metadata", {}),
    }
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise content moderation agent. Follow policy, use the "
                    "available evidence, and respond with only valid JSON."
                    "Response format example: {\"action_type\": \"allow\", \"reason\": \"The content does not violate any policies.\"}"
                ),
            },
            {"role": "user", "content": json.dumps(prompt)},
        ],
    )
    # print(f"LLM raw response: {response}")
    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)
    # print(f"LLM response: {data}")
    return {
        "action_type": data["action_type"],
        "reason": data["reason"],
    }


def run_episode(http: httpx.Client, llm: OpenAI, case_id: str) -> EpisodeResult:
    reset_response = http.post("/reset", json={"case_id": case_id})
    reset_response.raise_for_status()
    payload = reset_response.json()
    observation = payload["observation"]
    done = bool(payload.get("done", False))

    while not done:
        action = llm_action(llm, observation)
        step_response = http.post("/step", json={"action": action})
        step_response.raise_for_status()
        payload = step_response.json()
        observation = payload["observation"]
        done = bool(payload.get("done", False))

    summary_response = http.get("/episode_summary")
    summary_response.raise_for_status()
    summary = summary_response.json()
    return EpisodeResult(
        case_id=case_id,
        total_reward=float(summary.get("total_reward", 0.0)),
        final_action=summary.get("final_action"),
        expected_action=summary.get("expected_action"),
    )


def main() -> None:
    # print(API_BASE_URL, MODEL_NAME, HF_TOKEN)
    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    results: list[EpisodeResult] = []
    with httpx.Client(base_url=ENV_BASE_URL, timeout=60.0) as http:
        for case_id in DEFAULT_CASE_IDS:
            results.append(run_episode(http, llm, case_id))
            print(f"Completed case_id={case_id} with total_reward={results[-1].total_reward:.3f}")

    aggregate = sum(result.total_reward for result in results)
    for result in results:
        print(
            f"case_id={result.case_id} total_reward={result.total_reward:.3f} "
            f"final_action={result.final_action} expected_action={result.expected_action}"
        )
    print(f"episodes={len(results)} aggregate_reward={aggregate:.3f} average_reward={aggregate / len(results):.3f}")


if __name__ == "__main__":
    main()
