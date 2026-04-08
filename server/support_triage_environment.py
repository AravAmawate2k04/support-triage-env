"""
Support Operations Environment v2.

Multi-step environment where an agent resolves customer support tickets
under incomplete information, policy constraints, and sequential decision-making.

Episode flow:
  reset(task, seed) → observe ticket (partial info only)
  step(view_customer)     → customer profile revealed
  step(search_kb, query)  → KB articles returned
  step(check_policy, ...) → exact policy text returned
  step(classify_ticket)   → category locked in (partial reward)
  step(assign)            → priority + team locked in
  step(apply_action)      → irreversible! reward or penalty
  step(draft_response)    → response graded
  step(close_ticket)      → episode ends, final composite score

State machine phases:
  open → classified → assigned → (responded) → closed
"""

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SupportTriageAction, SupportTriageObservation, ActionType
    from ..server.scenarios import (
        SCENARIOS, TASK_FAMILIES, TASK_MAX_STEPS, TASK_DESCRIPTIONS, Scenario,
    )
    from ..server.graders import (
        compute_final_score, per_step_reward, RETRIEVAL_ACTIONS,
    )
except ImportError:
    from models import SupportTriageAction, SupportTriageObservation, ActionType
    from server.scenarios import (
        SCENARIOS, TASK_FAMILIES, TASK_MAX_STEPS, TASK_DESCRIPTIONS, Scenario,
    )
    from server.graders import (
        compute_final_score, per_step_reward, RETRIEVAL_ACTIONS,
    )


VALID_TASKS = set(TASK_MAX_STEPS.keys())

# Retrieval action types (count toward evidence budget)
_RETRIEVAL = {"view_customer", "view_order_history", "search_kb", "check_policy"}


# ---------------------------------------------------------------------------
# Episode state
# ---------------------------------------------------------------------------

@dataclass
class EpisodeState:
    phase: str = "open"                      # open | classified | assigned | responded | closed
    category: Optional[str] = None
    priority: Optional[str] = None
    team: Optional[str] = None
    response_text: Optional[str] = None

    category_set: bool = False
    priority_set: bool = False
    team_set: bool = False
    response_drafted: bool = False
    escalated: bool = False
    escalation_reason: Optional[str] = None

    evidence_used: Set[str] = field(default_factory=set)
    # Track what was returned to avoid re-fetching (but still penalise redundant calls)
    customer_profile_fetched: bool = False
    order_history_fetched: bool = False
    last_kb_results: Optional[List[Dict[str, Any]]] = None  # result of last search_kb call
    last_policy_text: Optional[str] = None                  # result of last check_policy call

    applied_actions: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)

    total_retrieval_actions: int = 0
    steps_used: int = 0
    max_steps: int = 4
    cumulative_step_reward: float = 0.0


# ---------------------------------------------------------------------------
# Available actions per phase
# ---------------------------------------------------------------------------

def _available_actions(es: EpisodeState) -> List[str]:
    """Compute the list of valid action_types for the current episode state."""
    if es.phase == "closed":
        return []

    available = list(_RETRIEVAL)  # Retrieval always available

    # classify_ticket available until classified
    if not es.category_set:
        available.append("classify_ticket")

    # assign available once classified (or at any time — we allow it but warn)
    if es.category_set and not (es.priority_set and es.team_set):
        available.append("assign")
    elif not es.category_set:
        available.append("assign")  # allowed but grader penalises ordering

    # draft_response always available
    available.append("draft_response")

    # escalate always available
    available.append("escalate")

    # apply_action available (but penalised if no policy check)
    available.append("apply_action")

    # close_ticket requires at minimum classify + assign
    if es.category_set and es.priority_set and es.team_set:
        available.append("close_ticket")

    return available


# ---------------------------------------------------------------------------
# Knowledge base search
# ---------------------------------------------------------------------------

def _search_kb(query: str, scenario: Scenario) -> List[Dict[str, Any]]:
    """
    Relevance search over scenario KB articles.

    Ranking: query term overlap is the primary signal (each hit worth 3 points).
    A hidden relevance bonus (2/1/0 for high/medium/distractor) acts as a
    tie-breaker so that equal-scoring articles surface in a sensible order.

    The 'relevance' tag is NOT included in the returned results — the agent
    must judge relevance from the content, not a metadata label.

    An empty or off-topic query degrades to tie-breaker ordering, which puts
    high-relevance articles first but gives no advantage to the query terms.
    A precise query (e.g. "file upload crash 500") should score the correct
    article well above distractors.
    """
    if not scenario.kb_articles:
        return [{"title": "No results", "content": "No knowledge base articles found for this query."}]

    query_lower = query.lower() if query else ""
    terms = [t for t in query_lower.split() if len(t) > 2]

    results = []
    for article in scenario.kb_articles:
        content_lower = (article.get("title", "") + " " + article.get("content", "")).lower()
        hits = sum(1 for t in terms if t in content_lower) if terms else 0
        # Hidden tie-breaker: not exposed to agent
        rel_bonus = {"high": 2, "medium": 1, "distractor": 0}.get(
            article.get("relevance", "distractor"), 0
        )
        combined = hits * 3 + rel_bonus
        results.append({
            "title": article.get("title", ""),
            "content": article.get("content", ""),
            "_combined": combined,
        })

    results.sort(key=lambda a: -a["_combined"])
    return [{k: v for k, v in r.items() if k != "_combined"} for r in results]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SupportTriageEnvironment(Environment):
    """
    Customer Support Operations environment — multi-step, v2.

    Three tasks:
      classify_and_route      (easy,   max 4 steps)
      investigate_and_resolve (medium, max 8 steps)
      complex_operations      (hard,   max 12 steps)

    One episode = one ticket. Multiple WebSocket sessions are supported
    (SUPPORTS_CONCURRENT_SESSIONS = True).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task: str = "classify_and_route"
        self._scenario: Optional[Scenario] = None
        self._ep: EpisodeState = EpisodeState()
        self._episode_done: bool = False
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        task: str = "classify_and_route",
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> SupportTriageObservation:
        """
        Reset the environment for a new episode.

        Args:
            task: One of 'classify_and_route', 'investigate_and_resolve',
                  'complex_operations'. Falls back to 'classify_and_route'
                  if unknown.
            seed: Optional integer seed for reproducibility.
        """
        task = task.lower().strip()
        if task not in VALID_TASKS:
            task = "classify_and_route"

        if seed is not None:
            self._rng.seed(seed)

        self._task = task
        max_steps = TASK_MAX_STEPS[task]

        # Select a scenario from the appropriate families
        families = TASK_FAMILIES[task]
        candidates = [s for s in SCENARIOS if s.family in families]
        if not candidates:
            candidates = SCENARIOS  # Fallback: any scenario

        self._scenario = self._rng.choice(candidates)
        self._ep = EpisodeState(max_steps=max_steps)
        self._episode_done = False
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return self._make_observation(step_reward=0.0)

    def step(self, action: SupportTriageAction) -> SupportTriageObservation:  # type: ignore[override]
        """
        Execute one action in the current episode.

        Returns an observation with:
          - Updated retrieved information (if a retrieval action was taken)
          - step_reward for this action
          - feedback string describing what happened
          - done=True + score_breakdown only after close_ticket
        """
        if self._episode_done:
            raise RuntimeError("Episode already completed. Call reset() before step().")
        if self._scenario is None:
            raise RuntimeError("reset() must be called before step().")

        self._state.step_count += 1
        self._ep.steps_used += 1
        self._ep.actions_taken.append(action.action_type)

        action_type = (action.action_type or "").lower().strip()
        ep = self._ep
        scenario = self._scenario

        # Track retrieval action counts
        if action_type in _RETRIEVAL:
            ep.total_retrieval_actions += 1

        # Per-step reward
        step_r = per_step_reward(
            action_type=action_type,
            evidence_used=set(ep.evidence_used),
            required_evidence=scenario.ground_truth.must_gather_evidence,
            applied_actions=list(ep.applied_actions),
            ground_truth=scenario.ground_truth,
            step=ep.steps_used,
            max_steps=ep.max_steps,
        )
        ep.cumulative_step_reward += step_r

        feedback_lines: List[str] = []
        done = False
        final_reward: Optional[float] = None
        final_breakdown: Dict[str, float] = {}

        # ── Dispatch on action_type ──────────────────────────────────
        if action_type == "view_customer":
            ep.evidence_used.add("view_customer")
            ep.customer_profile_fetched = True
            feedback_lines.append("Customer profile retrieved.")

        elif action_type == "view_order_history":
            ep.evidence_used.add("view_order_history")
            ep.order_history_fetched = True
            feedback_lines.append("Order history retrieved.")

        elif action_type == "search_kb":
            ep.evidence_used.add("search_kb")
            ep.last_kb_results = _search_kb(action.query or "", scenario)
            feedback_lines.append(
                f"Knowledge base searched for: '{action.query or '(no query)'}'. "
                f"Returned {len(ep.last_kb_results)} article(s)."
            )

        elif action_type == "check_policy":
            ep.evidence_used.add("check_policy")
            policy_name = (action.policy_name or "").lower().strip()
            matched = next(
                (name for name in scenario.applicable_policies if policy_name in name),
                None,
            )
            if matched:
                ep.last_policy_text = f"[{matched}]\n{scenario.applicable_policies[matched]}"
                feedback_lines.append(f"Policy retrieved: '{matched}'.")
            elif scenario.applicable_policies:
                # Tell the agent what names exist, but don't expose content
                ep.last_policy_text = (
                    f"Policy '{policy_name}' not found. "
                    f"Available policies: {list(scenario.applicable_policies.keys())}. "
                    f"Use check_policy with one of these names to retrieve the full text."
                )
                feedback_lines.append(
                    f"Policy '{policy_name}' not found. "
                    f"Available: {list(scenario.applicable_policies.keys())}."
                )
            else:
                ep.last_policy_text = "No policies available for this scenario."
                feedback_lines.append("No policies found for this scenario.")

        elif action_type == "classify_ticket":
            cat = (action.category or "").lower().strip()
            ep.category = cat
            ep.category_set = True
            if ep.phase == "open":
                ep.phase = "classified"
            feedback_lines.append(f"Ticket classified as: '{cat}'.")

        elif action_type == "assign":
            pri = (action.priority or "").lower().strip()
            team = (action.assigned_team or "").lower().strip()
            ep.priority = pri
            ep.team = team
            ep.priority_set = bool(pri)
            ep.team_set = bool(team)
            if ep.phase in ("open", "classified") and ep.priority_set and ep.team_set:
                ep.phase = "assigned"
            feedback_lines.append(
                f"Ticket assigned: priority='{pri}', team='{team}'."
            )

        elif action_type == "draft_response":
            ep.response_text = action.response_text
            ep.response_drafted = True
            if ep.phase == "assigned":
                ep.phase = "responded"
            feedback_lines.append(
                "Response drafted. "
                f"Length: {len(action.response_text or '')} chars."
            )

        elif action_type == "escalate":
            ep.escalated = True
            ep.escalation_reason = action.escalation_reason
            if not action.escalation_reason:
                feedback_lines.append(
                    "Warning: escalation_reason is empty. "
                    "Provide a reason for full escalation credit."
                )
            else:
                feedback_lines.append(
                    f"Ticket escalated: '{action.escalation_reason}'."
                )

        elif action_type == "apply_action":
            applied = (action.applied_action or "").lower().strip()
            if not applied:
                feedback_lines.append("Warning: applied_action field is empty. No action taken.")
            else:
                ep.applied_actions.append(applied)
                if "check_policy" not in ep.evidence_used:
                    feedback_lines.append(
                        f"Warning: applied '{applied}' without checking policy first. "
                        "Penalty applied."
                    )
                else:
                    feedback_lines.append(f"Applied action: '{applied}'.")

        elif action_type == "close_ticket":
            if not (ep.category_set and ep.priority_set and ep.team_set):
                # Soft block: close allowed but will score poorly
                feedback_lines.append(
                    "Warning: closing without full classification (category/priority/team). "
                    "Score will be penalised."
                )
            ep.phase = "closed"
            self._episode_done = True
            done = True

            # Compute final composite score
            final_reward, final_breakdown = compute_final_score(
                category=ep.category,
                priority=ep.priority,
                assigned_team=ep.team,
                response_text=ep.response_text,
                customer_name=scenario.customer_name,
                evidence_used=ep.evidence_used,
                total_retrieval_actions=ep.total_retrieval_actions,
                applied_actions=ep.applied_actions,
                escalated=ep.escalated,
                steps_used=ep.steps_used,
                max_steps=ep.max_steps,
                cumulative_step_reward=ep.cumulative_step_reward,
                ground_truth=scenario.ground_truth,
            )
            feedback_lines.append(
                f"Episode closed. Final score: {final_reward:.3f}. "
                f"Steps used: {ep.steps_used}/{ep.max_steps}."
            )

        else:
            feedback_lines.append(
                f"Unknown action_type: '{action_type}'. No state change."
            )

        # If step budget exceeded and not already closed — force close with penalty
        if not self._episode_done and ep.steps_used >= ep.max_steps:
            ep.phase = "closed"
            self._episode_done = True
            done = True
            ep.cumulative_step_reward -= 0.10  # Budget exceeded penalty
            final_reward, final_breakdown = compute_final_score(
                category=ep.category,
                priority=ep.priority,
                assigned_team=ep.team,
                response_text=ep.response_text,
                customer_name=scenario.customer_name,
                evidence_used=ep.evidence_used,
                total_retrieval_actions=ep.total_retrieval_actions,
                applied_actions=ep.applied_actions,
                escalated=ep.escalated,
                steps_used=ep.steps_used,
                max_steps=ep.max_steps,
                cumulative_step_reward=ep.cumulative_step_reward,
                ground_truth=scenario.ground_truth,
            )
            feedback_lines.append(
                f"Step budget ({ep.max_steps}) exhausted. "
                f"Episode auto-closed. Final score: {final_reward:.3f}."
            )

        feedback = " ".join(feedback_lines)
        obs = self._make_observation(step_reward=step_r)
        obs.done = done
        obs.reward = final_reward if done else step_r
        obs.step_reward = step_r
        obs.cumulative_reward = ep.cumulative_step_reward
        obs.feedback = feedback
        if done:
            obs.score_breakdown = final_breakdown
        return obs

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self, step_reward: float) -> SupportTriageObservation:
        ep = self._ep
        scenario = self._scenario

        if scenario is None:
            return SupportTriageObservation(
                task_name=self._task,
                task_description=TASK_DESCRIPTIONS.get(self._task, ""),
                step=0,
                max_steps=TASK_MAX_STEPS.get(self._task, 4),
                available_actions=list(_RETRIEVAL) + ["classify_ticket", "assign", "escalate"],
                step_reward=0.0,
                cumulative_reward=0.0,
            )

        # Build observation — only include retrieved data that has been fetched
        obs = SupportTriageObservation(
            # Ticket (always visible)
            ticket_id=scenario.ticket_id,
            customer_name=scenario.customer_name,
            customer_tier=scenario.customer_tier,
            subject=scenario.subject,
            body=scenario.body,
            # Task context
            task_name=self._task,
            task_description=TASK_DESCRIPTIONS.get(self._task, ""),
            step=ep.steps_used,
            max_steps=ep.max_steps,
            # Available actions
            available_actions=_available_actions(ep),
            # Retrieved state (only if fetched)
            customer_profile=scenario.customer_profile if ep.customer_profile_fetched else None,
            order_history=scenario.order_history if ep.order_history_fetched else None,
            # kb_results and policy_text are set per-step below
            kb_results=None,
            policy_text=None,
            # Reward signals
            step_reward=step_reward,
            cumulative_reward=ep.cumulative_step_reward,
        )

        # Attach latest retrieval results to the observation
        last_action = ep.actions_taken[-1] if ep.actions_taken else ""
        if last_action == "search_kb":
            obs.kb_results = ep.last_kb_results
        elif last_action == "check_policy":
            obs.policy_text = ep.last_policy_text

        return obs
