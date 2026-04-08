"""
Support Operations Environment v2 — WebSocket client.

Wraps EnvClient with typed models for SupportTriageAction and
SupportTriageObservation.

Example (async)::

    async with SupportTriageEnv(base_url="ws://localhost:8000") as env:
        result = await env.reset(task="investigate_and_resolve")
        obs = result.observation
        print(obs.subject, obs.available_actions)

        result = await env.step(SupportTriageAction(
            action_type="view_customer"
        ))
        print(result.observation.customer_profile)
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
    """Typed WebSocket client for the Support Operations environment."""

    def _step_payload(self, action: SupportTriageAction) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"action_type": action.action_type}
        if action.category is not None:
            payload["category"] = action.category
        if action.priority is not None:
            payload["priority"] = action.priority
        if action.assigned_team is not None:
            payload["assigned_team"] = action.assigned_team
        if action.response_text is not None:
            payload["response_text"] = action.response_text
        if action.query is not None:
            payload["query"] = action.query
        if action.policy_name is not None:
            payload["policy_name"] = action.policy_name
        if action.escalation_reason is not None:
            payload["escalation_reason"] = action.escalation_reason
        if action.applied_action is not None:
            payload["applied_action"] = action.applied_action
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
            task_name=obs_data.get("task_name", "classify_and_route"),
            task_description=obs_data.get("task_description", ""),
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 4),
            available_actions=obs_data.get("available_actions", []),
            customer_profile=obs_data.get("customer_profile"),
            order_history=obs_data.get("order_history"),
            kb_results=obs_data.get("kb_results"),
            policy_text=obs_data.get("policy_text"),
            feedback=obs_data.get("feedback", ""),
            score_breakdown=obs_data.get("score_breakdown", {}),
            step_reward=obs_data.get("step_reward", 0.0),
            cumulative_reward=obs_data.get("cumulative_reward", 0.0),
            context=obs_data.get("context", {}),
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
