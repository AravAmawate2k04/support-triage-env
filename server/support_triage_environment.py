"""
Support Ticket Triage Environment.

Simulates the real-world task of a customer-support agent who must:
  1. Categorize incoming tickets (easy)
  2. Triage them: category + priority + team routing (medium)
  3. Write a full customer-facing response (hard)

Graders are fully deterministic — no external API calls.
"""

import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SupportTriageAction, SupportTriageObservation
except ImportError:
    from models import SupportTriageAction, SupportTriageObservation


# ---------------------------------------------------------------------------
# Ticket dataset
# ---------------------------------------------------------------------------

TICKETS: List[Dict[str, Any]] = [
    # ── Billing ──────────────────────────────────────────────────────────
    {
        "ticket_id": "TKT-001",
        "customer_name": "Sarah Johnson",
        "customer_tier": "pro",
        "subject": "Double charged on my account this month",
        "body": (
            "Hi, I noticed my credit card was charged twice for my Pro subscription "
            "this month. The charges appeared on March 1st and March 3rd. "
            "Please refund the duplicate charge as soon as possible."
        ),
        "category": "billing",
        "priority": "high",
        "team": "billing",
        "required_keywords": ["refund", "duplicate", "apologize"],
        "category_aliases": {"payment", "invoice", "charge", "subscription"},
        "context": {
            "policy": "Billing disputes are resolved within 3-5 business days. "
                      "Refunds are issued to the original payment method.",
            "faq": "Duplicate charges can occur during system updates. "
                   "We apologise for the inconvenience.",
        },
    },
    {
        "ticket_id": "TKT-002",
        "customer_name": "Michael Chen",
        "customer_tier": "free",
        "subject": "Cancel my subscription immediately",
        "body": (
            "I'd like to cancel my subscription. "
            "I'm no longer using the service and don't want to be charged again. "
            "Please process the cancellation and confirm via email."
        ),
        "category": "billing",
        "priority": "medium",
        "team": "billing",
        "required_keywords": ["cancel", "confirmation", "process"],
        "category_aliases": {"payment", "subscription", "account"},
        "context": {
            "policy": "Cancellations take effect at the end of the current billing cycle.",
        },
    },
    {
        "ticket_id": "TKT-003",
        "customer_name": "Emma Williams",
        "customer_tier": "enterprise",
        "subject": "Enterprise pricing and volume discount inquiry",
        "body": (
            "We're evaluating upgrading 50 seats to the Enterprise plan. "
            "Can you provide information about volume discounts, SLA guarantees, "
            "and what's included in the enterprise tier?"
        ),
        "category": "billing",
        "priority": "low",
        "team": "sales",
        "required_keywords": ["pricing", "enterprise", "contact"],
        "category_aliases": {"payment", "upgrade", "plan", "quote"},
        "context": {
            "policy": "Enterprise sales inquiries are handled by the Sales team within 24 hours.",
        },
    },
    # ── Technical ────────────────────────────────────────────────────────
    {
        "ticket_id": "TKT-004",
        "customer_name": "David Park",
        "customer_tier": "pro",
        "subject": "App crashes when uploading files larger than 10 MB",
        "body": (
            "Whenever I try to upload a file larger than 10 MB the app crashes "
            "with 'Upload failed: server error 500'. "
            "This started after the latest update. Tested on Chrome 122 and Firefox 123."
        ),
        "category": "technical",
        "priority": "high",
        "team": "engineering",
        "required_keywords": ["bug", "investigate", "workaround"],
        "category_aliases": {"bug", "error", "crash", "issue", "upload"},
        "context": {
            "faq": "Known issue: files >10 MB may fail. Workaround: compress the file before uploading.",
        },
    },
    {
        "ticket_id": "TKT-005",
        "customer_name": "Lisa Anderson",
        "customer_tier": "enterprise",
        "subject": "URGENT: API returning 500 errors for 30 % of requests",
        "body": (
            "URGENT: Our production system is partially down. "
            "The /api/v2/process endpoint has returned 500 errors for ~30 % of requests "
            "for the past 2 hours. This is directly impacting our customers. "
            "We need immediate assistance."
        ),
        "category": "technical",
        "priority": "urgent",
        "team": "engineering",
        "required_keywords": ["escalate", "investigate", "status"],
        "category_aliases": {"bug", "error", "api", "outage", "down"},
        "context": {
            "policy": "P0 incidents for enterprise customers receive an SLA response within 30 minutes.",
        },
    },
    {
        "ticket_id": "TKT-006",
        "customer_name": "James Wilson",
        "customer_tier": "free",
        "subject": "How do I integrate with Slack?",
        "body": (
            "Hi, I'm trying to set up the Slack integration but can't find clear "
            "instructions in the documentation. Could you point me to the right guide "
            "or walk me through the configuration steps?"
        ),
        "category": "technical",
        "priority": "low",
        "team": "support",
        "required_keywords": ["documentation", "steps", "link"],
        "category_aliases": {"question", "integration", "how-to", "help", "slack"},
        "context": {
            "faq": "Slack integration guide: https://docs.example.com/integrations/slack",
        },
    },
    # ── Account ──────────────────────────────────────────────────────────
    {
        "ticket_id": "TKT-007",
        "customer_name": "Rachel Brown",
        "customer_tier": "pro",
        "subject": "Can't reset password — reset email never arrives",
        "body": (
            "I've been trying to reset my password for over an hour and the reset "
            "email never arrives. I've checked spam and all folders. "
            "My account email is rachel.brown@example.com. I'm locked out."
        ),
        "category": "account",
        "priority": "high",
        "team": "support",
        "required_keywords": ["password", "reset", "email"],
        "category_aliases": {"login", "access", "locked", "credentials"},
        "context": {
            "policy": "Password reset emails may be delayed up to 10 minutes. "
                      "Support can trigger a manual reset if needed.",
        },
    },
    {
        "ticket_id": "TKT-008",
        "customer_name": "Tom Martinez",
        "customer_tier": "pro",
        "subject": "Transfer account ownership to new email address",
        "body": (
            "I've changed my company email. I need to migrate my account and all data "
            "from tom.old@company.com to tom.new@company.com, including my subscription. "
            "How do I do this?"
        ),
        "category": "account",
        "priority": "medium",
        "team": "support",
        "required_keywords": ["transfer", "email", "verify"],
        "category_aliases": {"email", "profile", "migrate", "change"},
        "context": {
            "policy": "Account email transfers require verification from both addresses.",
        },
    },
    {
        "ticket_id": "TKT-009",
        "customer_name": "Nina Patel",
        "customer_tier": "enterprise",
        "subject": "Locked out after switching phones — 2FA issue",
        "body": (
            "I switched to a new phone and can no longer access my authenticator app. "
            "I'm completely locked out. This is urgent — I have a critical presentation "
            "in 2 hours that requires account access."
        ),
        "category": "account",
        "priority": "urgent",
        "team": "support",
        "required_keywords": ["two-factor", "bypass", "verify"],
        "category_aliases": {"2fa", "mfa", "authentication", "locked", "security"},
        "context": {
            "policy": "2FA bypass requires identity verification via backup email or ID.",
        },
    },
    # ── Feedback ─────────────────────────────────────────────────────────
    {
        "ticket_id": "TKT-010",
        "customer_name": "Alex Turner",
        "customer_tier": "free",
        "subject": "Feature request: dark mode",
        "body": (
            "I'd love to see a dark mode added to the app. "
            "The bright white interface is hard on the eyes during late-night sessions. "
            "Many modern apps offer this — it would be a great addition!"
        ),
        "category": "feedback",
        "priority": "low",
        "team": "support",
        "required_keywords": ["feature", "request", "team"],
        "category_aliases": {"feature request", "suggestion", "enhancement", "idea"},
        "context": {},
    },
    {
        "ticket_id": "TKT-011",
        "customer_name": "Sophie Lee",
        "customer_tier": "pro",
        "subject": "Compliment for your support team",
        "body": (
            "I just wanted to say your support agent Jessica was incredibly helpful "
            "last week with my billing issue. She resolved everything quickly and "
            "professionally. You have a fantastic team!"
        ),
        "category": "feedback",
        "priority": "low",
        "team": "support",
        "required_keywords": ["thank", "appreciate", "feedback"],
        "category_aliases": {"compliment", "praise", "positive", "review"},
        "context": {},
    },
    {
        "ticket_id": "TKT-012",
        "customer_name": "Carlos Mendez",
        "customer_tier": "pro",
        "subject": "Suggestion: CSV export for reports",
        "body": (
            "It would be very useful to export reports as CSV files. "
            "Currently I have to manually copy data into Excel, which is time-consuming. "
            "This feature would save me several hours each week."
        ),
        "category": "feedback",
        "priority": "low",
        "team": "support",
        "required_keywords": ["suggestion", "export", "feature"],
        "category_aliases": {"feature request", "suggestion", "export", "enhancement"},
        "context": {},
    },
    # ── Shipping ─────────────────────────────────────────────────────────
    {
        "ticket_id": "TKT-013",
        "customer_name": "Amy Thompson",
        "customer_tier": "free",
        "subject": "Order #98765 not arrived after 2 weeks",
        "body": (
            "I placed order #98765 on March 1st. It is now March 15th and it still "
            "shows 'in transit' but hasn't moved in 5 days. I need this item urgently. "
            "Please investigate."
        ),
        "category": "shipping",
        "priority": "high",
        "team": "logistics",
        "required_keywords": ["order", "track", "investigate"],
        "category_aliases": {"delivery", "package", "shipment", "order"},
        "context": {
            "policy": "Lost shipment investigations are opened after 10 business days.",
        },
    },
    {
        "ticket_id": "TKT-014",
        "customer_name": "Kevin O'Brien",
        "customer_tier": "pro",
        "subject": "Received the wrong item in my order",
        "body": (
            "I ordered the Blue Widget Pro (SKU: BW-PRO-001) but received "
            "a Red Widget Basic (SKU: RW-BAS-001). "
            "I need the correct item — I have a project deadline tomorrow."
        ),
        "category": "shipping",
        "priority": "high",
        "team": "logistics",
        "required_keywords": ["return", "replace", "correct"],
        "category_aliases": {"wrong item", "incorrect", "return", "order", "delivery"},
        "context": {
            "policy": "Wrong-item cases: we ship the correct item same-day and issue a pre-paid return label.",
        },
    },
    {
        "ticket_id": "TKT-015",
        "customer_name": "Grace Kim",
        "customer_tier": "free",
        "subject": "Need to change shipping address before dispatch",
        "body": (
            "I just placed order #45678 and realised I entered the wrong shipping "
            "address. The correct address is 456 Oak Ave, Portland OR 97201. "
            "Can you update this before it ships?"
        ),
        "category": "shipping",
        "priority": "medium",
        "team": "logistics",
        "required_keywords": ["address", "update", "order"],
        "category_aliases": {"address", "delivery", "change", "order"},
        "context": {
            "policy": "Address changes are possible up to 1 hour after order placement.",
        },
    },
]

# Task descriptions shown to the agent in the observation
TASK_DESCRIPTIONS = {
    "categorize": (
        "Classify this support ticket into exactly one category. "
        "Valid categories: billing, technical, account, feedback, shipping. "
        "Set the `category` field in your action. Leave priority and assigned_team as null."
    ),
    "triage": (
        "Triage this support ticket. Set: "
        "(1) category — billing, technical, account, feedback, or shipping; "
        "(2) priority — low, medium, high, or urgent; "
        "(3) assigned_team — billing, support, engineering, sales, or logistics."
    ),
    "respond": (
        "You are a customer-support agent. "
        "Triage the ticket (category, priority, assigned_team) AND write a professional "
        "customer-facing reply in `response_text`. "
        "Your reply must: greet the customer by name, acknowledge their issue, "
        "provide clear resolution steps, and end with a professional closing."
    ),
}

VALID_CATEGORIES = {"billing", "technical", "account", "feedback", "shipping"}
VALID_PRIORITIES = {"low", "medium", "high", "urgent"}
VALID_TEAMS = {"billing", "support", "engineering", "sales", "logistics"}

PRIORITY_ORDER = ["low", "medium", "high", "urgent"]
GREETING_WORDS = {"dear", "hi", "hello", "hey", "greetings", "good morning", "good afternoon"}
CLOSING_WORDS = {"regards", "sincerely", "thank you", "thanks", "best", "cheers", "warm regards", "kind regards"}


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------

def _category_score(predicted: str, ground_truth: str, aliases: set) -> float:
    p = predicted.lower().strip()
    if p == ground_truth:
        return 1.0
    if p in aliases:
        return 0.5
    return 0.0


def _priority_score(predicted: Optional[str], ground_truth: str) -> float:
    if predicted is None:
        return 0.0
    p = predicted.lower().strip()
    if p == ground_truth:
        return 1.0
    # Partial credit for adjacent priorities
    if p in PRIORITY_ORDER and ground_truth in PRIORITY_ORDER:
        dist = abs(PRIORITY_ORDER.index(p) - PRIORITY_ORDER.index(ground_truth))
        if dist == 1:
            return 0.4
    return 0.0


def _team_score(predicted: Optional[str], ground_truth: str) -> float:
    if predicted is None:
        return 0.0
    return 1.0 if predicted.lower().strip() == ground_truth else 0.0


def _response_score(response: Optional[str], ticket: Dict[str, Any]) -> Dict[str, float]:
    """
    Grade the response_text on five sub-criteria.
    Returns a dict of criterion → score (each already weighted).
    """
    if not response:
        return {
            "has_greeting": 0.0,
            "has_customer_name": 0.0,
            "keyword_coverage": 0.0,
            "proper_structure": 0.0,
            "length_ok": 0.0,
        }

    resp_lower = response.lower()
    customer_first = ticket["customer_name"].split()[0].lower()

    # Greeting: first 50 chars contain a greeting word
    first_50 = resp_lower[:50]
    has_greeting = any(word in first_50 for word in GREETING_WORDS)

    # Customer name present anywhere in response
    has_customer_name = customer_first in resp_lower

    # Keyword coverage (required_keywords)
    required = ticket["required_keywords"]
    hits = sum(1 for kw in required if kw.lower() in resp_lower)
    keyword_coverage = hits / len(required) if required else 1.0

    # Structure: has both a greeting and a closing line
    last_100 = resp_lower[-100:]
    has_closing = any(word in last_100 for word in CLOSING_WORDS)
    proper_structure = 1.0 if (has_greeting and has_closing) else (0.5 if (has_greeting or has_closing) else 0.0)

    # Length: 80–1000 chars is ideal; very short or very long penalised
    ln = len(response)
    if 80 <= ln <= 1000:
        length_ok = 1.0
    elif 40 <= ln < 80 or 1000 < ln <= 1500:
        length_ok = 0.5
    else:
        length_ok = 0.0

    return {
        "has_greeting": float(has_greeting),
        "has_customer_name": float(has_customer_name),
        "keyword_coverage": keyword_coverage,
        "proper_structure": proper_structure,
        "length_ok": length_ok,
    }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SupportTriageEnvironment(Environment):
    """
    Customer Support Ticket Triage environment.

    Three tasks of increasing difficulty:
      categorize – classify the ticket category (easy)
      triage     – category + priority + team routing (medium)
      respond    – full triage + customer-facing reply (hard)

    One episode = one ticket.  Each episode has a single action step.
    The environment supports concurrent sessions (each WebSocket client gets
    its own instance).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Reward weights per task
    _WEIGHTS_CATEGORIZE = {"category": 1.0}
    _WEIGHTS_TRIAGE = {"category": 0.40, "priority": 0.30, "team": 0.30}
    _WEIGHTS_RESPOND = {
        "category": 0.10,
        "priority": 0.05,
        "team": 0.05,
        "has_greeting": 0.10,
        "has_customer_name": 0.10,
        "keyword_coverage": 0.40,
        "proper_structure": 0.10,
        "length_ok": 0.10,
    }

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task: str = "categorize"
        self._ticket: Optional[Dict[str, Any]] = None
        self._episode_done: bool = False
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        task: str = "categorize",
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> SupportTriageObservation:
        """
        Reset the environment and return the first observation.

        Args:
            task:  One of 'categorize', 'triage', 'respond'.
            seed:  Optional random seed for reproducibility.
        """
        task = task.lower().strip()
        if task not in TASK_DESCRIPTIONS:
            task = "categorize"

        if seed is not None:
            self._rng.seed(seed)

        self._task = task
        self._ticket = self._rng.choice(TICKETS)
        self._episode_done = False
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return self._make_observation(done=False, reward=None)

    def step(self, action: SupportTriageAction) -> SupportTriageObservation:  # type: ignore[override]
        """
        Execute one action and return a graded observation.

        Returns:
            SupportTriageObservation with reward, done=True, and score breakdown.
        """
        if self._episode_done:
            raise RuntimeError("Episode already completed. Call reset() before step().")

        self._state.step_count += 1
        # Auto-initialise if reset() was never called (e.g. stateless HTTP mode)
        if self._ticket is None:
            self._ticket = self._rng.choice(TICKETS)
        ticket = self._ticket

        reward, breakdown = self._grade(action, ticket)

        # Episode ends after one step for all tasks
        self._episode_done = True
        obs = self._make_observation(done=True, reward=reward)
        obs.feedback = self._format_feedback(breakdown, ticket)
        obs.score_breakdown = breakdown
        return obs

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(
        self,
        done: bool,
        reward: Optional[float],
    ) -> SupportTriageObservation:
        ticket = self._ticket or {}
        context: Dict[str, Any] = {}
        if self._task == "respond":
            context = ticket.get("context", {})

        return SupportTriageObservation(
            ticket_id=ticket.get("ticket_id", ""),
            customer_name=ticket.get("customer_name", ""),
            customer_tier=ticket.get("customer_tier", "free"),
            subject=ticket.get("subject", ""),
            body=ticket.get("body", ""),
            task_name=self._task,
            task_description=TASK_DESCRIPTIONS.get(self._task, ""),
            step=self._state.step_count,
            context=context,
            done=done,
            reward=reward,
        )

    def _grade(
        self, action: SupportTriageAction, ticket: Dict[str, Any]
    ) -> tuple[float, Dict[str, float]]:
        """Return (total_reward_in_0_1, breakdown_dict)."""
        breakdown: Dict[str, float] = {}

        cat_s = _category_score(action.category, ticket["category"], ticket["category_aliases"])
        breakdown["category"] = cat_s

        if self._task == "categorize":
            weights = self._WEIGHTS_CATEGORIZE
            breakdown = {"category": cat_s}

        elif self._task == "triage":
            weights = self._WEIGHTS_TRIAGE
            breakdown["priority"] = _priority_score(action.priority, ticket["priority"])
            breakdown["team"] = _team_score(action.assigned_team, ticket["team"])

        else:  # respond
            weights = self._WEIGHTS_RESPOND
            breakdown["priority"] = _priority_score(action.priority, ticket["priority"])
            breakdown["team"] = _team_score(action.assigned_team, ticket["team"])
            resp_scores = _response_score(action.response_text, ticket)
            breakdown.update(resp_scores)

        reward = sum(weights[k] * breakdown.get(k, 0.0) for k in weights)
        reward = max(0.0, min(1.0, reward))
        return round(reward, 4), {k: round(v, 4) for k, v in breakdown.items()}

    def _format_feedback(
        self, breakdown: Dict[str, float], ticket: Dict[str, Any]
    ) -> str:
        lines = [
            f"Task: {self._task}",
            f"Ground truth → category={ticket['category']}, "
            f"priority={ticket['priority']}, team={ticket['team']}",
            "Score breakdown:",
        ]
        for criterion, score in breakdown.items():
            lines.append(f"  {criterion}: {score:.3f}")
        return "\n".join(lines)
