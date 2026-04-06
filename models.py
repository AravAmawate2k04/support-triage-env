"""
Data models for the Support Triage Environment.

An AI agent triages customer support tickets across three difficulty levels:
  - categorize (easy):  classify the ticket into a category
  - triage    (medium): classify + set priority + assign a team
  - respond   (hard):   full triage + draft a customer-facing reply
"""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SupportTriageAction(Action):
    """Action for the Support Triage environment."""

    # Required for all tasks
    category: str = Field(
        ...,
        description=(
            "Ticket category. One of: billing, technical, account, feedback, shipping"
        ),
    )

    # Required for triage + respond tasks (optional for categorize)
    priority: Optional[str] = Field(
        default=None,
        description="Ticket priority. One of: low, medium, high, urgent",
    )
    assigned_team: Optional[str] = Field(
        default=None,
        description=(
            "Team to handle the ticket. One of: billing, support, engineering, "
            "sales, logistics"
        ),
    )

    # Required for respond task only
    response_text: Optional[str] = Field(
        default=None,
        description="Full customer-facing response text (respond task only)",
    )


class SupportTriageObservation(Observation):
    """Observation from the Support Triage environment."""

    ticket_id: str = Field(default="", description="Unique ticket identifier")
    customer_name: str = Field(default="", description="Customer's full name")
    customer_tier: str = Field(
        default="free",
        description="Customer subscription tier: free, pro, enterprise",
    )
    subject: str = Field(default="", description="Ticket subject line")
    body: str = Field(default="", description="Full ticket body text")
    task_name: str = Field(
        default="categorize",
        description="Current task: categorize, triage, or respond",
    )
    task_description: str = Field(
        default="",
        description="Instructions for the current task",
    )
    step: int = Field(default=0, description="Current step within the episode")
    # Extra context provided for the respond task
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (policies, FAQs) for the respond task",
    )
    # Feedback after a step (populated after step(), empty after reset())
    feedback: str = Field(
        default="",
        description="Grader feedback on the last action (populated after step)",
    )
    score_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-criterion score breakdown (populated after step)",
    )
