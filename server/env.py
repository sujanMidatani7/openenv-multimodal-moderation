from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from .models import Action, ActionType, Content, Observation, PolicyChunk, State, StepType
    from .logic import (
        CASE_IDS,
        get_case,
        get_expected_action,
        compute_step_reward,
        load_tasks
    )
    from .rag.retriever import retrieve_policy_chunks
except ImportError:
    from models import Action, ActionType, Content, Observation, PolicyChunk, State, StepType
    from logic import (
        CASE_IDS,
        get_case,
        get_expected_action,
        compute_step_reward,
    )
    from rag.retriever import retrieve_policy_chunks


# Episode step flow — each step() call advances to the next stage
EPISODE_FLOW = ["analyze", "retrieve_policy", "decide", "review", "finalize"]


class ModerationEnvironment(Environment):
    """OpenEnv environment for multimodal content moderation."""

    def __init__(self) -> None:
        super().__init__()
        self._state = State()
        self._case: Optional[Dict[str, Any]] = None
        self._current_step_index: int = 0
        self.tasks = load_tasks()

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> Observation:
        eid = episode_id or str(uuid.uuid4())

        # Determine which case to use
        # Allow caller to pass case_id via kwargs (used by inference.py)
        case_id = kwargs.get("case_id")
        if case_id and case_id in CASE_IDS:
            chosen_id = case_id
        elif seed is not None:
            chosen_id = CASE_IDS[seed % len(CASE_IDS)]
        else:
            import random
            chosen_id = random.choice(CASE_IDS)

        self._case = get_case(chosen_id)
        self._current_step_index = 0

        self._state = State(
            episode_id=eid,
            step_count=0,
            done=False,
            selected_case_id=chosen_id,
            reward_breakdown={
                "analysis_step": 0.0,
                "retrieval_step": 0.0,
                "correct_decision": 0.0,
                "reviewer_agreement": 0.0,
                "unsafe_penalty": 0.0,
            },
            final_action=None,
            reviewer_note=None,
            action_history=[],
            retrieved_policy_chunks=[],
        )

        content = Content(**self._case["content"])
        return Observation(
            content=content,
            policy=[],
            step_type=StepType.analyze,
            step_count=0,
            message=f"Episode started. Case: {chosen_id}. Begin with analysis.",
            reward=0.0,
            done=False,
        )

    def step(self, action: Action, **kwargs) -> Observation:
        if self._case is None:
            raise RuntimeError("Call reset() before step()")

        if self._state.done:
            return Observation(
                step_type=StepType.finalize,
                step_count=self._state.step_count,
                message="Episode already finished.",
                reward=0.0,
                done=True,
            )

        step_name = EPISODE_FLOW[self._current_step_index]
        reward = compute_step_reward(step_name, action.action_type.value, self._case)

        # Record reward into breakdown
        breakdown = self._state.reward_breakdown
        if step_name == "analyze":
            breakdown["analysis_step"] += reward
        elif step_name == "retrieve_policy":
            breakdown["retrieval_step"] += reward
        elif step_name == "decide":
            if reward > 0:
                breakdown["correct_decision"] += reward
            else:
                breakdown["unsafe_penalty"] += reward
        elif step_name == "review":
            breakdown["reviewer_agreement"] += reward

        # Record action history
        self._state.action_history.append({
            "step": step_name,
            "action_type": action.action_type.value,
            "reason": action.reason,
            "reward": reward,
        })

        self._state.step_count += 1
        self._current_step_index += 1

        # Build observation for next step
        policy_chunks: list[PolicyChunk] = []
        message = ""
        next_step_type = StepType.finalize

        if step_name == "retrieve_policy":
            # Actually retrieve now that we're done with retrieve_policy
            raw_chunks = retrieve_policy_chunks(self._case["content"].get("text", ""), top_k=3)
            policy_chunks = [PolicyChunk(**c) for c in raw_chunks]
            self._state.retrieved_policy_chunks = policy_chunks
            message = "Policy retrieved. Now make your moderation decision."
        elif step_name == "analyze":
            message = "Analysis complete. Retrieve relevant policy next."
        elif step_name == "decide":
            self._state.final_action = action.action_type.value
            message = "Decision recorded. Awaiting reviewer validation."
        elif step_name == "review":
            self._state.reviewer_note = action.reason or "Reviewer note recorded."
            message = "Review complete. Finalizing episode."
        elif step_name == "finalize":
            message = "Episode finalized."

        done = self._current_step_index >= len(EPISODE_FLOW)
        self._state.done = done

        # Determine next step type for observation
        if not done and self._current_step_index < len(EPISODE_FLOW):
            next_step_type = StepType(EPISODE_FLOW[self._current_step_index])

        return Observation(
            content=Content(**self._case["content"]),
            policy=policy_chunks or self._state.retrieved_policy_chunks,
            step_type=next_step_type,
            step_count=self._state.step_count,
            message=message,
            reward=reward,
            done=done,
        )

    @property
    def state(self) -> State:
        return self._state