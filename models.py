"""
Data models for the Support Operations Environment v2.

Multi-step environment where an agent resolves support tickets under
incomplete information, policy constraints, and sequential decision-making.

Tasks:
  classify_and_route      (easy)   — classify category + priority + team
  investigate_and_resolve (medium) — gather evidence before resolving
  complex_operations      (hard)   — high-stakes, irreversible actions, SLA pressure
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    CLASSIFY_TICKET    = "classify_ticket"      # set category
    VIEW_CUSTOMER      = "view_customer"         # retrieve customer profile
    VIEW_ORDER_HISTORY = "view_order_history"    # retrieve order/billing ledger
    SEARCH_KB          = "search_kb"             # search knowledge base
    CHECK_POLICY       = "check_policy"          # look up a specific policy by name
    ASSIGN             = "assign"                # set priority + team
    DRAFT_RESPONSE     = "draft_response"        # write customer-facing reply
    ESCALATE           = "escalate"              # escalate to specialist
    APPLY_ACTION       = "apply_action"          # issue refund / lock account / etc.
    CLOSE_TICKET       = "close_ticket"          # finish the episode


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

class SupportTriageAction(Action):
    """
    Unified action for the Support Operations environment.

    Set `action_type` to control which action is performed.
    Only populate the fields relevant to that action type.

    Examples::

        # Classify the ticket
        {"action_type": "classify_ticket", "category": "billing"}

        # Search knowledge base
        {"action_type": "search_kb", "query": "refund policy enterprise"}

        # Assign priority and team
        {"action_type": "assign", "priority": "high", "assigned_team": "billing"}

        # Apply an action (irreversible)
        {"action_type": "apply_action", "applied_action": "refund"}

        # Escalate with reason
        {"action_type": "escalate", "escalation_reason": "Unauthorized account access detected"}

        # Draft customer response
        {"action_type": "draft_response", "response_text": "Dear Sarah, ..."}

        # Close the ticket (requires classify + assign first)
        {"action_type": "close_ticket"}
    """

    action_type: str = Field(
        ...,
        description=(
            "Which action to perform. One of: "
            "classify_ticket, view_customer, view_order_history, search_kb, "
            "check_policy, assign, draft_response, escalate, apply_action, close_ticket"
        ),
    )

    # classify_ticket
    category: Optional[str] = Field(
        default=None,
        description="Ticket category: billing | technical | account | feedback | shipping",
    )

    # assign
    priority: Optional[str] = Field(
        default=None,
        description="Ticket priority: low | medium | high | urgent",
    )
    assigned_team: Optional[str] = Field(
        default=None,
        description="Team: billing | support | engineering | sales | logistics",
    )

    # draft_response
    response_text: Optional[str] = Field(
        default=None,
        description="Full customer-facing reply (for draft_response action)",
    )

    # search_kb
    query: Optional[str] = Field(
        default=None,
        description="Search query for the knowledge base (for search_kb action)",
    )

    # check_policy
    policy_name: Optional[str] = Field(
        default=None,
        description="Name of the policy to retrieve (for check_policy action)",
    )

    # escalate
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Reason for escalation (required for escalate action)",
    )

    # apply_action
    applied_action: Optional[str] = Field(
        default=None,
        description=(
            "Irreversible action to apply. One of: refund, lock_account, "
            "unlock_account, waive_charges, reship_order (for apply_action)"
        ),
    )


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

class SupportTriageObservation(Observation):
    """
    Observation returned after reset() or step().

    The ticket fields are always visible. Hidden state fields
    (customer_profile, order_history, kb_results, policy_text) are
    populated only after the corresponding retrieval action is taken.
    """

    # ── Ticket (always visible) ──────────────────────────────────────
    ticket_id: str = Field(default="", description="Unique ticket identifier")
    customer_name: str = Field(default="", description="Customer's full name")
    customer_tier: str = Field(
        default="free",
        description="Customer subscription tier: free | pro | enterprise",
    )
    subject: str = Field(default="", description="Ticket subject line")
    body: str = Field(default="", description="Full ticket body text")

    # ── Task context ─────────────────────────────────────────────────
    task_name: str = Field(
        default="classify_and_route",
        description="Current task: classify_and_route | investigate_and_resolve | complex_operations",
    )
    task_description: str = Field(
        default="",
        description="Instructions for the current task",
    )
    step: int = Field(default=0, description="Current step number (1-indexed after first action)")
    max_steps: int = Field(default=4, description="Step budget for this episode")

    # ── Available actions ────────────────────────────────────────────
    available_actions: List[str] = Field(
        default_factory=list,
        description="Action types valid at this step",
    )

    # ── Retrieved information (populated by retrieval actions) ───────
    customer_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Customer profile revealed by view_customer action",
    )
    order_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Order history revealed by view_order_history action",
    )
    kb_results: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="KB articles returned by search_kb action",
    )
    policy_text: Optional[str] = Field(
        default=None,
        description="Policy text returned by check_policy action",
    )

    # ── Feedback (populated after each step) ─────────────────────────
    feedback: str = Field(
        default="",
        description="Step-level feedback from the environment",
    )
    score_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Final score breakdown by dimension (populated after close_ticket)",
    )

    # ── Reward signals ───────────────────────────────────────────────
    step_reward: float = Field(
        default=0.0,
        description="Reward for the action just taken",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Total reward accumulated in this episode so far",
    )

    # ── Legacy context field (kept for backward compatibility) ───────
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (deprecated — use retrieved fields instead)",
    )
