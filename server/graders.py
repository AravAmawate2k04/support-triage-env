"""
Compositional grading for the Support Operations Environment v2.

Grading is fully deterministic — no LLM calls, no external APIs.

Two layers:
  1. Per-step reward signals  — small immediate feedback on each action
  2. Final composite score    — evaluated at close_ticket

Final reward = clamp(composite_score + cumulative_step_rewards, 0.0, 1.0)
"""

from typing import Any, Dict, List, Optional, Set

try:
    from ..server.scenarios import GroundTruth
except ImportError:
    from server.scenarios import GroundTruth


# ---------------------------------------------------------------------------
# Synonym groups for response grading
# ---------------------------------------------------------------------------

SYNONYM_GROUPS: Dict[str, Set[str]] = {
    "apologize": {
        "apologize", "sorry", "apologies", "apology", "regret",
        "sincerely sorry", "we apologize", "i apologize",
    },
    "refund": {
        "refund", "reimburse", "credit back", "return the charge",
        "reverse the charge", "credit", "reimbursement",
    },
    "investigate": {
        "investigate", "look into", "examine", "review", "check on",
        "look at", "trace", "investigation",
    },
    "escalate": {
        "escalate", "forward to", "refer to", "pass to", "hand off",
        "escalated", "escalating", "escalation",
    },
    "workaround": {
        "workaround", "alternative", "temporary solution", "meanwhile",
        "in the meantime", "compress", "bulk upload",
    },
    "locked": {
        "locked", "suspended", "secured", "lock", "protecting",
    },
    "reship": {
        "reship", "resend", "send the correct", "ship the correct",
        "replacement", "new order",
    },
}


def _matches_synonym(text_lower: str, group_key: str) -> bool:
    """Return True if any synonym in the group appears in text."""
    return any(syn in text_lower for syn in SYNONYM_GROUPS.get(group_key, set()))


# ---------------------------------------------------------------------------
# Grading weights (must sum to 1.0)
# ---------------------------------------------------------------------------

GRADING_WEIGHTS: Dict[str, float] = {
    # Classification
    "category_correct":         0.10,
    "priority_correct":         0.05,
    "team_correct":             0.05,
    # Evidence gathering
    "evidence_gathered":        0.15,
    "evidence_efficiency":      0.05,
    # Policy compliance
    "policy_checked":           0.10,
    "action_correct":           0.10,
    "no_forbidden_action":      0.05,
    # Response quality
    "response_addresses_issue": 0.10,
    "response_no_hallucination":0.05,
    "response_structure":       0.05,
    "response_length":          0.03,
    "response_uses_name":       0.02,
    # Efficiency
    "step_efficiency":          0.05,
    "sla_compliance":           0.05,
}
# Sanity check
assert abs(sum(GRADING_WEIGHTS.values()) - 1.0) < 1e-6, "Grading weights must sum to 1.0"


# ---------------------------------------------------------------------------
# Priority helpers
# ---------------------------------------------------------------------------

PRIORITY_ORDER = ["low", "medium", "high", "urgent"]
GREETING_WORDS = {
    "dear", "hi ", "hi,", "hello", "hey ", "hey,",
    "greetings", "good morning", "good afternoon", "good evening",
}
CLOSING_WORDS = {
    "regards", "sincerely", "thank you", "thanks", "best",
    "cheers", "warm regards", "kind regards", "yours truly",
}


def _category_score(predicted: Optional[str], ground_truth: str) -> float:
    if not predicted:
        return 0.0
    p = predicted.lower().strip()
    if p == ground_truth:
        return 1.0
    # Aliases that get partial credit
    aliases: Dict[str, Set[str]] = {
        "billing":   {"payment", "invoice", "charge", "subscription", "refund"},
        "technical": {"bug", "error", "crash", "issue", "api", "outage"},
        "account":   {"login", "access", "locked", "credentials", "security", "2fa"},
        "feedback":  {"feature request", "suggestion", "enhancement", "idea", "compliment"},
        "shipping":  {"delivery", "package", "shipment", "order", "logistics"},
    }
    if p in aliases.get(ground_truth, set()):
        return 0.5
    return 0.0


def _priority_score(predicted: Optional[str], ground_truth: str) -> float:
    if not predicted:
        return 0.0
    p = predicted.lower().strip()
    if p == ground_truth:
        return 1.0
    if p in PRIORITY_ORDER and ground_truth in PRIORITY_ORDER:
        dist = abs(PRIORITY_ORDER.index(p) - PRIORITY_ORDER.index(ground_truth))
        if dist == 1:
            return 0.4
    return 0.0


def _team_score(predicted: Optional[str], ground_truth: str) -> float:
    if not predicted:
        return 0.0
    return 1.0 if predicted.lower().strip() == ground_truth else 0.0


def _evidence_gathered_score(
    evidence_used: Set[str],
    required_evidence: List[str],
) -> float:
    """Fraction of required evidence sources that were actually consulted."""
    if not required_evidence:
        return 1.0
    hits = sum(1 for e in required_evidence if e in evidence_used)
    return hits / len(required_evidence)


def _evidence_efficiency_score(
    evidence_used: Set[str],
    required_evidence: List[str],
    total_retrieval_actions: int,
) -> float:
    """Penalise redundant/irrelevant retrieval steps."""
    if total_retrieval_actions == 0:
        return 1.0
    required_set = set(required_evidence)
    useful = sum(1 for e in evidence_used if e in required_set)
    wasted = total_retrieval_actions - useful
    if wasted <= 0:
        return 1.0
    # Penalise wasted steps proportionally
    return max(0.0, 1.0 - 0.2 * wasted)


def _policy_checked_score(
    evidence_used: Set[str],
    required_evidence: List[str],
    applied_actions: List[str],
) -> float:
    """
    Did the agent check the relevant policy before applying an irreversible action?
    Returns 1.0 if: (a) no irreversible action was applied, or
                    (b) policy was checked before applying.
    Returns 0.0 if an irreversible action was applied without checking policy.
    """
    if not applied_actions:
        return 1.0  # No irreversible action — nothing to penalise
    policy_checked = "check_policy" in evidence_used
    return 1.0 if policy_checked else 0.0


def _action_correct_score(
    applied_actions: List[str],
    escalated: bool,
    ground_truth: GroundTruth,
) -> float:
    """
    Was the correct action taken (or correctly omitted)?
    """
    required = ground_truth.required_action
    eligible = ground_truth.action_eligible
    must_esc = ground_truth.must_escalate

    score = 0.0

    if required is None and not must_esc:
        # No action needed — score 1.0 if nothing was applied
        if not applied_actions and not escalated:
            score = 1.0
        elif applied_actions:
            score = 0.0  # applied something when shouldn't have
        else:
            score = 0.5  # escalated when not required — minor penalty
    elif required == "escalate" or must_esc:
        score = 1.0 if escalated else 0.0
    else:
        # required_action is "refund" or "lock_account"
        if eligible and required in applied_actions:
            score = 1.0
        elif eligible and required not in applied_actions:
            score = 0.0  # Should have applied but didn't
        elif not eligible and required in applied_actions:
            score = 0.0  # Applied when ineligible
        elif not eligible and required not in applied_actions:
            score = 1.0  # Correctly withheld

    return score


def _no_forbidden_action_score(
    applied_actions: List[str],
    ground_truth: GroundTruth,
) -> float:
    """Penalise if an ineligible action was applied."""
    if not ground_truth.action_eligible and applied_actions:
        return 0.0
    return 1.0


def _response_score(
    response: Optional[str],
    customer_name: str,
    ground_truth: GroundTruth,
) -> Dict[str, float]:
    """Grade the customer-facing response on four dimensions."""
    if not response:
        return {
            "response_addresses_issue": 0.0,
            "response_no_hallucination": 1.0,  # no response = no hallucination
            "response_structure": 0.0,
            "response_length": 0.0,
            "response_uses_name": 0.0,
        }

    resp_lower = response.lower()
    first_name = customer_name.split()[0].lower() if customer_name else ""

    # 1. Required facts present (synonym-aware)
    required_facts = ground_truth.required_response_facts
    fact_hits = 0
    for fact in required_facts:
        fact_lower = fact.lower()
        # Check direct match or synonym group match
        matched = fact_lower in resp_lower
        if not matched:
            for group_key, synonyms in SYNONYM_GROUPS.items():
                if fact_lower in synonyms and _matches_synonym(resp_lower, group_key):
                    matched = True
                    break
        if matched:
            fact_hits += 1
    addresses_issue = fact_hits / len(required_facts) if required_facts else 1.0

    # 2. No hallucination — prohibited claims absent
    prohibited = ground_truth.prohibited_claims
    violations = sum(1 for claim in prohibited if claim.lower() in resp_lower)
    no_hallucination = max(0.0, 1.0 - 0.5 * violations)

    # 3. Structure: greeting + closing
    first_80 = resp_lower[:80]
    last_100 = resp_lower[-100:]
    has_greeting = any(w in first_80 for w in GREETING_WORDS)
    has_closing = any(w in last_100 for w in CLOSING_WORDS)
    if has_greeting and has_closing:
        structure = 1.0
    elif has_greeting or has_closing:
        structure = 0.5
    else:
        structure = 0.0

    # 4. Length: 80–1000 chars is ideal
    ln = len(response)
    if 80 <= ln <= 1000:
        length_score = 1.0
    elif 40 <= ln < 80 or 1000 < ln <= 1500:
        length_score = 0.5
    else:
        length_score = 0.0

    # 5. Customer name in response
    uses_name = 1.0 if (first_name and first_name in resp_lower) else 0.0

    return {
        "response_addresses_issue": round(addresses_issue, 4),
        "response_no_hallucination": round(no_hallucination, 4),
        "response_structure": structure,
        "response_length": length_score,
        "response_uses_name": uses_name,
    }


def _step_efficiency_score(steps_used: int, max_steps: int) -> float:
    """Reward completing the episode well under the step budget."""
    if max_steps <= 0:
        return 1.0
    ratio = steps_used / max_steps
    if ratio <= 0.5:
        return 1.0
    elif ratio <= 0.75:
        return 0.7
    elif ratio <= 1.0:
        return 0.4
    else:
        return 0.0  # Exceeded budget


def _sla_compliance_score(
    steps_used: int,
    sla_deadline_steps: Optional[int],
    escalated: bool,
    ground_truth: GroundTruth,
) -> float:
    """Full credit if SLA met (or no SLA). Zero if SLA required and missed."""
    if sla_deadline_steps is None:
        return 1.0
    # SLA requires escalation or close within deadline
    if steps_used <= sla_deadline_steps:
        return 1.0
    # Partial credit for close call
    if steps_used <= sla_deadline_steps + 2:
        return 0.3
    return 0.0


# ---------------------------------------------------------------------------
# Final composite score
# ---------------------------------------------------------------------------

def compute_final_score(
    *,
    category: Optional[str],
    priority: Optional[str],
    assigned_team: Optional[str],
    response_text: Optional[str],
    customer_name: str,
    evidence_used: Set[str],
    total_retrieval_actions: int,
    applied_actions: List[str],
    escalated: bool,
    steps_used: int,
    max_steps: int,
    cumulative_step_reward: float,
    ground_truth: GroundTruth,
) -> tuple[float, Dict[str, float]]:
    """
    Compute the final composite score at close_ticket.

    Returns (total_reward, breakdown_dict).
    Total reward = clamp(composite + cumulative_step_reward, 0.0, 1.0)
    """
    w = GRADING_WEIGHTS
    bd: Dict[str, float] = {}

    # Classification
    bd["category_correct"]  = _category_score(category, ground_truth.category)
    bd["priority_correct"]  = _priority_score(priority, ground_truth.priority)
    bd["team_correct"]      = _team_score(assigned_team, ground_truth.team)

    # Evidence
    bd["evidence_gathered"] = _evidence_gathered_score(
        evidence_used, ground_truth.must_gather_evidence
    )
    bd["evidence_efficiency"] = _evidence_efficiency_score(
        evidence_used, ground_truth.must_gather_evidence, total_retrieval_actions
    )

    # Policy compliance
    bd["policy_checked"]      = _policy_checked_score(
        evidence_used, ground_truth.must_gather_evidence, applied_actions
    )
    bd["action_correct"]      = _action_correct_score(
        applied_actions, escalated, ground_truth
    )
    bd["no_forbidden_action"] = _no_forbidden_action_score(applied_actions, ground_truth)

    # Response
    resp_scores = _response_score(response_text, customer_name, ground_truth)
    bd["response_addresses_issue"]  = resp_scores["response_addresses_issue"]
    bd["response_no_hallucination"] = resp_scores["response_no_hallucination"]
    bd["response_structure"]        = resp_scores["response_structure"]
    bd["response_length"]           = resp_scores["response_length"]
    bd["response_uses_name"]        = resp_scores["response_uses_name"]

    # Efficiency
    bd["step_efficiency"] = _step_efficiency_score(steps_used, max_steps)
    bd["sla_compliance"]  = _sla_compliance_score(
        steps_used, ground_truth.sla_deadline_steps, escalated, ground_truth
    )

    # Weighted composite
    composite = sum(w[k] * bd.get(k, 0.0) for k in w)
    total = composite + cumulative_step_reward
    total = round(max(0.0, min(1.0, total)), 4)
    bd = {k: round(v, 4) for k, v in bd.items()}

    return total, bd


# ---------------------------------------------------------------------------
# Per-step reward signals
# ---------------------------------------------------------------------------

RETRIEVAL_ACTIONS = {"view_customer", "view_order_history", "search_kb", "check_policy"}


def per_step_reward(
    action_type: str,
    evidence_used: Set[str],    # BEFORE this action
    required_evidence: List[str],
    applied_actions: List[str],
    ground_truth: GroundTruth,
    step: int,
    max_steps: int,
) -> float:
    """
    Compute a small reward signal for a single step.

    Signals:
      Useful retrieval (needed)       +0.02
      Redundant retrieval (repeat)    -0.01
      Correct classification          +0.05  (on classify_ticket)
      Wrong classification            -0.02
      Policy check before irrev.      +0.03
      Irrev. action without policy    -0.10
      Escalation when required        +0.05
      Unnecessary escalation          -0.03
      Failure to escalate (fraud)     -0.15  (only at close)
      Each step                       -0.005 (time pressure)
      Exceeding step budget           -0.10
    """
    r = -0.005  # Time cost per step

    if step > max_steps:
        r -= 0.10  # Over budget

    if action_type in RETRIEVAL_ACTIONS:
        if action_type in evidence_used:
            r -= 0.01  # Redundant retrieval
        elif action_type in required_evidence:
            r += 0.02  # Useful retrieval
        # else: neutral (irrelevant but first time)

    elif action_type == "apply_action":
        if "check_policy" not in evidence_used:
            r -= 0.10  # Applied without checking policy

    elif action_type == "escalate":
        if ground_truth.must_escalate:
            r += 0.05
        else:
            r -= 0.03  # Unnecessary escalation

    return round(r, 4)
