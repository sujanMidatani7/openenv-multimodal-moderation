from __future__ import annotations

from typing import Any

from openenv.core.env_server.interfaces import Environment

from server.logic import (
    EpisodeContext,
    build_query,
    build_retriever,
    compute_step_reward,
    convert_chunks,
    infer_rule_override,
    new_episode,
    next_step,
    reviewer_recommendation,
)
from server.models import Action, Content, Observation, PolicyChunk, State, StepType


class ContentModerationEnv(Environment[Action, Observation, State]):
    def __init__(self) -> None:
        super().__init__()
        self._retriever = build_retriever()
        self._episode: EpisodeContext | None = None
        self.state_data: dict[str, Any] = {}
        self._state = State()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        case_id: str | None = None,
        **_: Any,
    ) -> Observation:
        self._episode = new_episode(seed=seed, episode_id=episode_id, case_id=case_id)
        case = self._episode.case
        self.state_data = {
            "case_id": case.case_id,
            "content_text": case.content_text,
            "image_metadata": case.image_metadata,
            "expected_action": case.expected_action.value,
            "reviewer_action": case.reviewer_action.value,
            "history": [],
            "policy_chunks": [],
            "reviewer_note": None,
            "final_action": None,
            "reward_breakdown": [],
        }
        self._state = State(
            episode_id=self._episode.episode_id,
            step_count=0,
            done=False,
            current_step=StepType.ANALYZE,
            state_data=self.state_data,
            total_reward=0.0,
        )
        return self._build_observation(step_type=StepType.ANALYZE, reward=None, done=False)

    def step(
        self,
        action: Action,
        timeout_s: float | None = None,
        **_: Any,
    ) -> Observation:
        del timeout_s
        if self._episode is None:
            raise RuntimeError("Environment not initialized. Call reset() before step().")
        if self._state.done:
            raise RuntimeError("Episode already finished. Call reset() to start a new episode.")

        current_step = self._state.current_step
        self._state.step_count += 1
        self.state_data["history"].append(
            {
                "step": current_step.value,
                "action_type": action.action_type.value,
                "reason": action.reason,
            }
        )

        policy_chunks = self._policy_for_step(current_step, action.reason)
        reviewer_action_value = None
        if current_step == StepType.DECIDE:
            reviewer_action_value, agrees, message = reviewer_recommendation(
                self._episode.case,
                action.action_type,
            )
            self.state_data["reviewer_note"] = {
                "recommended_action": reviewer_action_value.value,
                "agrees_with_moderator": agrees,
                "message": message,
            }

        if current_step == StepType.REVIEW:
            self.state_data["final_action"] = action.action_type.value
            reviewer_note = self.state_data.get("reviewer_note") or {}
            recommended = reviewer_note.get("recommended_action")
            if recommended is not None:
                reviewer_action_value = action.action_type.__class__(recommended)

        reward_value, reward_details, summary = compute_step_reward(
            step_type=current_step,
            action_type=action.action_type,
            reason=action.reason,
            case=self._episode.case,
            policy=policy_chunks,
            reviewer_action_value=reviewer_action_value,
        )
        self._state.total_reward += reward_value
        self.state_data["reward_breakdown"].append(
            {
                "step": current_step.value,
                "reward": reward_value,
                "details": reward_details,
                "summary": summary,
            }
        )

        done = current_step == StepType.REVIEW
        next_step_type = StepType.FINALIZE if done else next_step(current_step)
        self._state.current_step = next_step_type or StepType.FINALIZE
        self._state.done = done
        self._state.state_data = self.state_data

        return self._build_observation(
            step_type=self._state.current_step,
            reward=reward_value,
            done=done,
            reward_details=reward_details,
            summary=summary,
        )

    @property
    def state(self) -> State:
        return self._state

    def _policy_for_step(self, step_type: StepType, latest_reason: str) -> list[PolicyChunk]:
        if self._episode is None:
            return []
        if step_type == StepType.ANALYZE:
            query = build_query(self._episode.case, analysis_reason=latest_reason)
            chunks = convert_chunks(self._retriever.retrieve(query=query, top_k=3))
            self.state_data["policy_chunks"] = [chunk.model_dump() for chunk in chunks]
            return chunks
        stored = self.state_data.get("policy_chunks", [])
        return [PolicyChunk(**chunk) for chunk in stored]

    def _build_observation(
        self,
        step_type: StepType,
        reward: float | None,
        done: bool,
        reward_details: dict[str, float | str] | None = None,
        summary: str | None = None,
    ) -> Observation:
        policy = [PolicyChunk(**chunk) for chunk in self.state_data.get("policy_chunks", [])]
        rule_override = infer_rule_override(self.state_data.get("content_text", ""))
        metadata: dict[str, Any] = {
            "episode_id": self._state.episode_id,
            "case_id": self.state_data.get("case_id"),
            "moderator_role": "decision step owner",
            "reviewer_role": "validation step owner",
            "rule_override": rule_override.value if rule_override else None,
            "state_data": self.state_data,
        }
        if step_type == StepType.ANALYZE:
            metadata["instruction"] = "Analyze text and image metadata for policy risk."
        elif step_type == StepType.RETRIEVE_POLICY:
            metadata["instruction"] = "Use the retrieved policy chunks to ground the moderation decision."
        elif step_type == StepType.DECIDE:
            metadata["instruction"] = "Choose the best moderation action for the case."
        elif step_type == StepType.REVIEW:
            metadata["instruction"] = "Validate the moderator decision against the reviewer signal."
            metadata["reviewer_note"] = self.state_data.get("reviewer_note")
        elif step_type == StepType.FINALIZE:
            metadata["instruction"] = "Episode complete. Inspect the final moderation outcome and total score."
            metadata["final_action"] = self.state_data.get("final_action")
            metadata["total_reward"] = round(self._state.total_reward, 4)

        if reward_details is not None:
            metadata["latest_reward"] = reward_details
        if summary is not None:
            metadata["latest_reward_summary"] = summary

        return Observation(
            content=Content(
                text=self.state_data.get("content_text", ""),
                image_metadata=self.state_data.get("image_metadata", {}),
            ),
            policy=policy if step_type != StepType.ANALYZE else [],
            step_type=step_type,
            step_count=self._state.step_count,
            done=done,
            reward=reward,
            metadata=metadata,
        )
