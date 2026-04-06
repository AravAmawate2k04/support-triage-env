"""
Support Triage Environment — WebSocket client.

Wraps the EnvClient with typed models for SupportTriageAction and
SupportTriageObservation so callers get IDE autocomplete and type safety.

Example (async)::

    async with SupportTriageEnv(base_url="ws://localhost:8000") as env:
        result = await env.reset(task="triage")
        obs = result.observation
        print(obs.subject)

        action = SupportTriageAction(
            category="billing", priority="high", assigned_team="billing"
        )
        result = await env.step(action)
        print(result.reward)

Example (sync)::

    env = SupportTriageEnv(base_url="ws://localhost:8000").sync()
    with env:
        result = env.reset(task="categorize")
        result = env.step(SupportTriageAction(category="billing"))
"""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import SupportTriageAction, SupportTriageObservation
except ImportError:
    from models import SupportTriageAction, SupportTriageObservation


class SupportTriageEnv(
    EnvClient[SupportTriageAction, SupportTriageObservation, State]
):
    """Typed WebSocket client for the Support Triage environment."""

    def _step_payload(self, action: SupportTriageAction) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"category": action.category}
        if action.priority is not None:
            payload["priority"] = action.priority
        if action.assigned_team is not None:
            payload["assigned_team"] = action.assigned_team
        if action.response_text is not None:
            payload["response_text"] = action.response_text
        return payload

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[SupportTriageObservation]:
        obs_data = payload.get("observation", {})
        observation = SupportTriageObservation(
            ticket_id=obs_data.get("ticket_id", ""),
            customer_name=obs_data.get("customer_name", ""),
            customer_tier=obs_data.get("customer_tier", "free"),
            subject=obs_data.get("subject", ""),
            body=obs_data.get("body", ""),
            task_name=obs_data.get("task_name", "categorize"),
            task_description=obs_data.get("task_description", ""),
            step=obs_data.get("step", 0),
            context=obs_data.get("context", {}),
            feedback=obs_data.get("feedback", ""),
            score_breakdown=obs_data.get("score_breakdown", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
