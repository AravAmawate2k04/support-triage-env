# Implementation Plan: Support Operations Environment v2

## The Pivot

**From:** Customer support ticket triage (single-shot classification)
**To:** Customer support operations under incomplete information, policy constraints, and sequential decision-making

**One-line thesis:** An agent should not just label tickets. It should resolve them safely, efficiently, and policy-correctly under uncertainty.

**What we keep:** The domain, the repo, the infra (Docker, HF Space, openenv.yaml, validation).
**What we rebuild:** The environment core, models, grading, tickets, and inference.

---

## Phase 1: Multi-Step Environment Core (MUST-DO FIRST)

This is the single biggest improvement. Everything else builds on it.

### 1.1 New Action Types

Replace the single `SupportTriageAction` with a discriminated union of action types.

**File: `models.py`**

```python
class ActionType(str, Enum):
    CLASSIFY_TICKET   = "classify_ticket"      # set category
    VIEW_CUSTOMER     = "view_customer"         # retrieve customer profile
    VIEW_ORDER_HISTORY = "view_order_history"   # retrieve order/billing ledger
    SEARCH_KB         = "search_kb"             # search knowledge base (query string)
    CHECK_POLICY      = "check_policy"          # look up a specific policy by name
    ASSIGN            = "assign"                # set priority + team
    DRAFT_RESPONSE    = "draft_response"        # write customer-facing reply
    ESCALATE          = "escalate"              # escalate to specialist (with reason)
    APPLY_ACTION      = "apply_action"          # issue refund / lock account / etc.
    CLOSE_TICKET      = "close_ticket"          # finish the episode
```

The `SupportTriageAction` model gets an `action_type` discriminator field plus optional payloads:

```python
class SupportTriageAction(Action):
    action_type: str                       # which action (see ActionType enum)
    category: Optional[str] = None         # for classify_ticket
    priority: Optional[str] = None         # for assign
    assigned_team: Optional[str] = None    # for assign
    response_text: Optional[str] = None    # for draft_response
    query: Optional[str] = None            # for search_kb
    policy_name: Optional[str] = None      # for check_policy
    escalation_reason: Optional[str] = None # for escalate
    applied_action: Optional[str] = None   # for apply_action (e.g. "refund", "lock_account")
```

### 1.2 Richer Observation Space

**File: `models.py`**

```python
class SupportTriageObservation(Observation):
    # Ticket (always visible)
    ticket_id: str
    customer_name: str
    customer_tier: str            # free | pro | enterprise
    subject: str
    body: str

    # Task context
    task_name: str                # easy | medium | hard
    task_description: str
    step: int
    max_steps: int                # step budget (e.g. 8-12)

    # Available actions at this step
    available_actions: List[str]  # which action_types are valid now

    # Retrieved information (populated by retrieval actions)
    customer_profile: Optional[Dict] = None    # from view_customer
    order_history: Optional[List[Dict]] = None  # from view_order_history
    kb_results: Optional[List[Dict]] = None     # from search_kb
    policy_text: Optional[str] = None           # from check_policy

    # Grading feedback (populated after close_ticket)
    feedback: str = ""
    score_breakdown: Dict[str, float] = {}

    # Step-level reward signal
    step_reward: float = 0.0      # reward for this specific step
    cumulative_reward: float = 0.0 # total reward so far
```

### 1.3 Episode Flow

**File: `server/support_triage_environment.py`**

Each episode is now multi-step. The agent receives a ticket with incomplete info and must gather evidence before acting.

```
reset(task="medium", seed=42)
  → observe ticket (partial info only — no customer profile, no order history)

step(action_type="view_customer")
  → customer profile revealed (account age, tier, abuse flags, SLA tier)
  → small time cost (-0.01 reward, but info gained)

step(action_type="search_kb", query="refund policy enterprise")
  → KB articles returned (relevant + distractors)
  → small time cost

step(action_type="check_policy", policy_name="refund_eligibility")
  → exact policy text returned

step(action_type="classify_ticket", category="billing")
  → category locked in, partial reward if correct

step(action_type="assign", priority="high", assigned_team="billing")
  → assignment locked in, partial reward

step(action_type="apply_action", applied_action="refund")
  → irreversible! Reward or penalty based on whether refund was warranted

step(action_type="draft_response", response_text="Dear Sarah, ...")
  → response graded

step(action_type="close_ticket")
  → episode ends, final composite score
```

### 1.4 State Machine

Track episode phase internally:

```python
class EpisodeState:
    phase: str              # "open" | "classified" | "assigned" | "responded" | "closed"
    category_set: bool
    priority_set: bool
    team_set: bool
    response_drafted: bool
    evidence_gathered: Set[str]   # which retrieval actions were used
    actions_taken: List[str]      # full action log
    applied_actions: List[str]    # irreversible actions taken
    steps_used: int
    max_steps: int
```

Enforce ordering constraints:
- Cannot `close_ticket` without at least `classify_ticket` + `assign`
- Cannot `apply_action` (refund/lock) without first checking the relevant policy (penalty if skipped)
- `escalate` is always available but must include a reason
- `draft_response` can happen at any point but scores better if evidence was gathered first

---

## Phase 2: Hidden State and Information Asymmetry

### 2.1 Structured Scenario Data

Each scenario has a **ground truth** the agent cannot see directly — it must be discovered through retrieval actions.

**File: `server/scenarios.py`** (new file)

```python
@dataclass
class Scenario:
    # Visible at reset
    ticket_id: str
    customer_name: str
    customer_tier: str          # free | pro | enterprise
    subject: str
    body: str                   # the raw ticket text

    # Hidden — revealed by view_customer
    customer_profile: Dict      # account_age, plan, abuse_history, sla_tier
    
    # Hidden — revealed by view_order_history
    order_history: List[Dict]   # orders with dates, amounts, refund history
    
    # Hidden — revealed by search_kb / check_policy
    applicable_policies: Dict[str, str]   # policy_name → policy_text
    kb_articles: List[Dict]               # title, content, relevance
    
    # Ground truth for grading (never shown to agent)
    ground_truth: GroundTruth

@dataclass
class GroundTruth:
    category: str
    priority: str
    team: str
    required_action: Optional[str]        # "refund" | "lock_account" | "escalate" | None
    action_eligible: bool                 # is the required_action actually warranted?
    must_escalate: bool                   # mandatory escalation (security/fraud)
    must_gather_evidence: List[str]       # which retrieval actions are needed
    required_response_facts: List[str]    # facts that must appear in response
    prohibited_claims: List[str]          # things the response must NOT say
    sla_deadline_steps: Optional[int]     # must resolve within N steps for full credit
    response_keywords: List[str]          # for partial keyword matching (kept for backward compat)
```

### 2.2 Scenario Categories

Build ~40 scenarios across these families:

| Family | Count | Key challenge |
|--------|-------|---------------|
| Billing disputes (refund eligible) | 5 | Must verify eligibility before refunding |
| Billing disputes (refund ineligible) | 4 | Must deny refund politely with policy reference |
| Account security (real threat) | 4 | Must escalate + lock, cannot just respond |
| Account security (false alarm) | 3 | Must verify before locking (costly mistake) |
| Technical bugs (known issue) | 4 | Must find KB article, provide workaround |
| Technical bugs (new issue) | 3 | Must escalate to engineering, not promise fix |
| Enterprise SLA pressure | 4 | Time-sensitive, tier-aware priority |
| Feedback / feature requests | 4 | Simple routing, but with distractors |
| Shipping / logistics | 4 | Order history lookup required |
| Ambiguous / multi-issue | 5 | Straddle categories, conflicting signals |

Each scenario has:
- Realistic messy ticket text (typos, emoji, angry tone, vague complaints)
- A customer profile with relevant history
- 2-3 KB articles (1 relevant, 1-2 distractors)
- Applicable policies with specific conditions
- Ground truth that requires evidence gathering to determine correctly

### 2.3 Seeded Scenario Generation

For reproducibility + scale, add a lightweight scenario generator:

```python
def generate_scenario(seed: int, family: str) -> Scenario:
    rng = random.Random(seed)
    # Pick a template from the family
    # Randomize: customer name, tier, order amounts, dates, abuse flags
    # Render ticket text from template + randomized vars
    # Set ground truth based on structured variables
    ...
```

This lets us have deterministic episodes with seed, but unlimited diversity.

---

## Phase 3: Compositional Grading

### 3.1 New Grading Dimensions

**File: `server/graders.py`** (new file)

Replace keyword-only grading with a compositional rubric:

```python
GRADING_DIMENSIONS = {
    # Classification correctness
    "category_correct":        0.10,  # exact match or alias
    "priority_correct":        0.05,  # exact or adjacent
    "team_correct":            0.05,  # exact match
    
    # Evidence gathering
    "evidence_gathered":       0.15,  # did agent check required sources before acting?
    "evidence_efficiency":     0.05,  # didn't waste steps on irrelevant lookups
    
    # Policy compliance
    "policy_checked":          0.10,  # consulted policy before irreversible action
    "action_correct":          0.10,  # refund/escalate/lock decision correct
    "no_forbidden_action":     0.05,  # didn't do something prohibited (e.g. refund when ineligible)
    
    # Response quality
    "response_addresses_issue": 0.10, # required facts present in response
    "response_no_hallucination": 0.05, # prohibited claims absent
    "response_structure":      0.05,  # greeting + name + closing
    "response_length":         0.03,  # 80-1000 chars
    "response_uses_name":      0.02,  # customer name in response
    
    # Efficiency
    "step_efficiency":         0.05,  # completed within SLA / step budget
    "sla_compliance":          0.05,  # urgent tickets resolved quickly
}
# Total: 1.00
```

### 3.2 Per-Step Rewards

Instead of grading only at the end, give small signals throughout:

| Action | Reward signal |
|--------|--------------|
| Useful retrieval (needed evidence) | +0.02 |
| Redundant retrieval (already had info) | -0.01 |
| Correct classification | +0.05 |
| Wrong classification | -0.02 |
| Policy check before irreversible action | +0.03 |
| Irreversible action without policy check | -0.10 |
| Escalation when required | +0.05 |
| Failure to escalate security issue | -0.15 |
| Unnecessary escalation | -0.03 |
| Each step taken | -0.005 (time pressure) |
| Exceeding step budget | -0.10 |

Final reward = clamp(composite_score + cumulative_step_rewards, 0.0, 1.0)

### 3.3 Synonym Matching for Response Grading

Replace pure keyword matching with synonym groups:

```python
SYNONYM_GROUPS = {
    "apologize": {"apologize", "sorry", "apologies", "apology", "regret", "sincerely sorry"},
    "refund": {"refund", "reimburse", "credit back", "return the charge", "reverse the charge"},
    "investigate": {"investigate", "look into", "examine", "review", "check on"},
    "escalate": {"escalate", "forward to", "refer to", "pass to", "hand off"},
    ...
}
```

Still fully deterministic, but much harder to game.

---

## Phase 4: Redesigned Task Difficulty

### Easy: `classify_and_route`
- Ticket has complete information — no hidden state needed
- Agent must classify category + priority + team
- Single-step is fine (but multi-step is allowed)
- Straightforward tickets, no ambiguity
- Scores: classification correctness + step efficiency
- **Max steps: 4** (generous — can be done in 1-2)

### Medium: `investigate_and_resolve`
- Ticket has partial/ambiguous information
- Agent MUST use retrieval actions (view_customer, search_kb, check_policy) to determine correct resolution
- Must apply the correct action (refund, workaround, etc.) based on evidence
- Distractors in KB results
- Scores: evidence gathering + policy compliance + classification + response quality
- **Max steps: 8**

### Hard: `complex_operations`
- High-stakes tickets with conflicting signals
- Security/fraud scenarios requiring mandatory escalation
- Enterprise SLA pressure (must resolve within step budget)
- Irreversible actions with real consequences (wrong refund = large penalty)
- Ambiguous categories, multi-issue tickets
- Must reason about: risk, compliance, missing data, action ordering, cost of mistakes
- Scores: all dimensions including action correctness, no_forbidden_action, sla_compliance
- **Max steps: 12**

Why frontier models still fail on hard:
- Premature closure (closing without gathering evidence)
- Hallucinated policy (claiming a refund policy that doesn't exist)
- Skipping evidence gathering (acting on assumptions)
- Over-escalation (escalating simple issues, wasting specialist time)
- Wrong risk tradeoff (refunding when ineligible, or denying when SLA requires it)
- Missing mandatory compliance steps (security lock before responding to fraud)

---

## Phase 5: Two Baselines

### `inference.py` (default — simple one-shot baseline)

Same structure as current, but adapted for multi-step:
- Agent gets the ticket, immediately classifies, assigns, drafts response, closes
- No retrieval actions used
- Shows the env works but scores poorly on evidence gathering / policy compliance

### `inference_tool_use.py` (smart baseline)

- Agent uses a system prompt that describes available actions
- LLM decides which action to take at each step
- Follows a gather-then-act strategy: view_customer → search_kb → check_policy → classify → assign → draft → close
- Should score significantly higher than simple baseline

Include both in the README with a comparison table:

```
| Task       | Simple Baseline | Tool-Use Baseline | Gap    |
|------------|----------------|-------------------|--------|
| easy       | ~0.85          | ~0.95             | +0.10  |
| medium     | ~0.45          | ~0.80             | +0.35  |
| hard       | ~0.25          | ~0.65             | +0.40  |
```

The gap proves the environment rewards sophisticated reasoning, not just prompt-following.

---

## Phase 6: Polish

1. **README rewrite** — reframe as "support operations benchmark", not "triage"
2. **Episode flow diagram** in README showing the multi-step decision process
3. **Example trajectories** — one good, one bad, one unsafe (with scores)
4. **Failure case analysis** — short section on why frontier models fail on hard
5. **Unit tests** for graders and seeded scenario reproducibility
6. **Ensure all validators pass** — openenv validate, Docker build, HF Space deploy

---

## File Changes Summary

| File | Change |
|------|--------|
| `models.py` | Rewrite — new Action with action_type discriminator, richer Observation |
| `server/support_triage_environment.py` | Major rewrite — multi-step state machine, phase tracking, per-step rewards |
| `server/scenarios.py` | **New** — scenario dataclass, 40+ scenarios, scenario generator |
| `server/graders.py` | **New** — compositional grading with 15 dimensions, synonym matching |
| `server/knowledge_base.py` | **New** — KB articles + policies keyed per scenario |
| `server/app.py` | Minor — update imports |
| `client.py` | Update — handle new observation fields |
| `inference.py` | Rewrite — multi-step loop, simple baseline strategy |
| `inference_tool_use.py` | **New** — smart baseline with retrieval strategy |
| `openenv.yaml` | No change |
| `Dockerfile` | No change |
| `README.md` | Major rewrite — new framing, diagrams, baselines, failure analysis |
| `test_env.py` | Rewrite — test multi-step flow |

---

## Execution Order

**Day 1: Foundation**
1. `server/scenarios.py` — build 15 scenarios first (expand to 40 later)
2. `models.py` — new Action + Observation models
3. `server/support_triage_environment.py` — multi-step state machine
4. `server/knowledge_base.py` — KB + policies for initial scenarios
5. Smoke test: can we reset → step through a multi-step episode manually?

**Day 2: Grading + Baselines**
6. `server/graders.py` — compositional grading
7. `inference.py` — simple baseline (multi-step)
8. `inference_tool_use.py` — smart baseline
9. Run both baselines, verify score gap
10. Expand to 40 scenarios

**Day 3: Polish + Ship**
11. `README.md` — full rewrite with new framing
12. `test_env.py` — updated tests
13. `client.py` — updated client
14. Run `openenv validate`, Docker build, deploy to HF Space
15. Final validation: all checklist items pass

---

## What NOT to Spend Time On

- Fancy frontend / Space UI (judges care about task design, not presentation)
- LLM-as-judge grading (keep it deterministic)
- Overly long technical writeup (TECHNICAL.md is already solid)
- Adding more trivial tasks beyond 3
- Excessive packaging polish beyond what the validator checks
- Domain pivot (support is fine — the upgrade is in depth, not domain)

---

## Success Criteria

The environment is "winning-tier" when:

1. A naive one-shot agent scores ~0.25-0.45 on hard (proves it's genuinely hard)
2. A smart tool-use agent scores ~0.65-0.80 on hard (proves the env rewards reasoning)
3. The grader catches hallucinated policies, premature closure, and unsafe actions
4. Episodes are 4-10 steps long with meaningful decisions at each step
5. All validators pass, Docker builds, HF Space responds, inference runs in <20 min
