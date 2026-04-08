"""
Scenario definitions for the Support Operations Environment v2.

Each Scenario bundles:
  - The visible ticket (what the agent sees at reset)
  - Hidden state (revealed only by retrieval actions)
  - Ground truth (never shown; used only by the grader)

15 scenarios across 9 families:
  billing_eligible    (2) — refund warranted, agent must verify
  billing_ineligible  (2) — refund not warranted, agent must deny politely
  security_real       (2) — fraud/compromise, must lock + escalate
  security_false      (1) — looks suspicious, actually benign
  tech_known          (2) — bug with KB workaround
  tech_new            (1) — no KB entry, must escalate to engineering
  enterprise_sla      (2) — time-sensitive, tier-aware
  feedback            (1) — simple routing
  shipping            (2) — order history required
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Ground truth dataclass
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
    category: str                            # billing | technical | account | feedback | shipping
    priority: str                            # low | medium | high | urgent
    team: str                                # billing | support | engineering | sales | logistics
    required_action: Optional[str]           # "refund" | "lock_account" | "escalate" | None
    action_eligible: bool                    # if agent applies required_action, is it correct?
    must_escalate: bool                      # mandatory escalation (security/fraud/enterprise)
    must_gather_evidence: List[str]          # retrieval actions needed before acting
    required_response_facts: List[str]       # strings that MUST appear in response (lowercased)
    prohibited_claims: List[str]             # strings that must NOT appear in response (lowercased)
    sla_deadline_steps: Optional[int]        # must close within N steps for full SLA credit
    response_keywords: List[str]             # legacy keyword list for partial matching


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    # ── Visible at reset ──────────────────────────────────────────────
    scenario_id: str
    family: str
    ticket_id: str
    customer_name: str
    customer_tier: str          # free | pro | enterprise
    subject: str
    body: str                   # raw ticket text (may have typos / emotional tone)

    # ── Hidden: revealed by view_customer ────────────────────────────
    customer_profile: Dict[str, Any]
    # keys: account_age_months, plan, abuse_history (list), sla_tier, open_tickets, vip

    # ── Hidden: revealed by view_order_history ───────────────────────
    order_history: List[Dict[str, Any]]
    # each entry: order_id, date, amount, status, items, refund_issued

    # ── Hidden: revealed by search_kb / check_policy ─────────────────
    kb_articles: List[Dict[str, Any]]
    # each entry: title, content, relevance ("high"|"medium"|"distractor")

    applicable_policies: Dict[str, str]
    # policy_name → policy_text

    # ── Ground truth (grader only) ───────────────────────────────────
    ground_truth: GroundTruth


# ---------------------------------------------------------------------------
# Scenario data — 15 scenarios
# ---------------------------------------------------------------------------

SCENARIOS: List[Scenario] = [

    # ════════════════════════════════════════════════════════════════
    # BILLING — REFUND ELIGIBLE
    # ════════════════════════════════════════════════════════════════

    Scenario(
        scenario_id="SCEN-001",
        family="billing_eligible",
        ticket_id="TKT-B001",
        customer_name="Sarah Johnson",
        customer_tier="pro",
        subject="Double charged on my account this month",
        body=(
            "Hi, I noticed my credit card was charged TWICE for my Pro subscription "
            "this month — once on April 1st ($49) and again on April 3rd ($49). "
            "I definitely only have one account. Please refund the extra charge ASAP, "
            "this is really frustrating!!"
        ),
        customer_profile={
            "account_age_months": 26,
            "plan": "pro",
            "abuse_history": [],
            "sla_tier": "standard",
            "open_tickets": 0,
            "vip": False,
        },
        order_history=[
            {
                "order_id": "INV-2024-0401",
                "date": "2024-04-01",
                "amount": 49.00,
                "status": "charged",
                "items": ["Pro subscription renewal"],
                "refund_issued": False,
            },
            {
                "order_id": "INV-2024-0403",
                "date": "2024-04-03",
                "amount": 49.00,
                "status": "charged",
                "items": ["Pro subscription renewal"],
                "refund_issued": False,
            },
        ],
        kb_articles=[
            {
                "title": "Duplicate charge investigation process",
                "content": (
                    "Duplicate charges may occur during billing system maintenance windows. "
                    "To verify: check if two identical charges appear within 5 days. "
                    "If confirmed, issue refund to original payment method within 3-5 business days."
                ),
                "relevance": "high",
            },
            {
                "title": "How to update payment method",
                "content": "Customers can update their payment method in Account Settings > Billing.",
                "relevance": "distractor",
            },
        ],
        applicable_policies={
            "refund_policy": (
                "Refunds are issued for: (1) verified duplicate charges, "
                "(2) charges within 30 days that the customer did not use the service. "
                "Refunds are processed within 3-5 business days to the original payment method. "
                "No refunds are issued for consumed services beyond 30 days."
            ),
            "billing_dispute_policy": (
                "Billing disputes must be verified against order history before any refund is issued. "
                "Support agents may not issue refunds without verifying the charge in order history."
            ),
        },
        ground_truth=GroundTruth(
            category="billing",
            priority="high",
            team="billing",
            required_action="refund",
            action_eligible=True,
            must_escalate=False,
            must_gather_evidence=["view_order_history", "check_policy"],
            required_response_facts=["refund", "duplicate", "apologize"],
            prohibited_claims=["cannot refund", "ineligible", "policy does not allow"],
            sla_deadline_steps=None,
            response_keywords=["refund", "duplicate", "apologize"],
        ),
    ),

    Scenario(
        scenario_id="SCEN-002",
        family="billing_eligible",
        ticket_id="TKT-B002",
        customer_name="Emma Williams",
        customer_tier="enterprise",
        subject="Invoice discrepancy — charged for 55 seats, only have 48",
        body=(
            "Hello, our finance team flagged our April invoice. "
            "We are being billed for 55 seats ($2,750/mo) but we only have 48 active users. "
            "That's a $350 overcharge. Please correct our invoice and issue a credit."
        ),
        customer_profile={
            "account_age_months": 18,
            "plan": "enterprise",
            "abuse_history": [],
            "sla_tier": "enterprise",
            "open_tickets": 1,
            "vip": True,
        },
        order_history=[
            {
                "order_id": "INV-ENT-APR",
                "date": "2024-04-01",
                "amount": 2750.00,
                "status": "charged",
                "items": ["Enterprise plan — 55 seats"],
                "refund_issued": False,
            },
        ],
        kb_articles=[
            {
                "title": "Enterprise seat reconciliation process",
                "content": (
                    "Enterprise seat counts are reconciled monthly. If the billed seat count "
                    "exceeds active users, a credit is issued for the difference. "
                    "To verify: pull order history and cross-reference with active user count "
                    "from customer profile."
                ),
                "relevance": "high",
            },
            {
                "title": "How to add enterprise seats",
                "content": "Admins can add seats under Settings > Team > Manage Seats.",
                "relevance": "distractor",
            },
        ],
        applicable_policies={
            "enterprise_billing_policy": (
                "Enterprise customers are billed based on contracted seat count. "
                "If billed seat count exceeds active users, issue a credit memo for the overage. "
                "Credits are applied to the next invoice unless customer requests a cash refund."
            ),
            "refund_policy": (
                "Refunds are issued for: (1) verified duplicate charges, "
                "(2) charges within 30 days that the customer did not use the service. "
                "Refunds are processed within 3-5 business days to the original payment method."
            ),
        },
        ground_truth=GroundTruth(
            category="billing",
            priority="high",
            team="billing",
            required_action="refund",
            action_eligible=True,
            must_escalate=False,
            must_gather_evidence=["view_customer", "view_order_history", "check_policy"],
            required_response_facts=["credit", "overage", "48"],
            prohibited_claims=["cannot refund", "no credit available"],
            sla_deadline_steps=None,
            response_keywords=["credit", "seats", "invoice"],
        ),
    ),

    # ════════════════════════════════════════════════════════════════
    # BILLING — REFUND INELIGIBLE
    # ════════════════════════════════════════════════════════════════

    Scenario(
        scenario_id="SCEN-003",
        family="billing_ineligible",
        ticket_id="TKT-B003",
        customer_name="Jake Peters",
        customer_tier="free",
        subject="Refund request — didnt know i was still subscribed",
        body=(
            "hi i just noticed i've been charged $19/month for 4 months. "
            "i thought i cancelled ages ago. i want ALL of it back ($76). "
            "this is ridiculous i havent used the app at all!!!"
        ),
        customer_profile={
            "account_age_months": 8,
            "plan": "starter",
            "abuse_history": ["chargeback_attempt_2023"],
            "sla_tier": "standard",
            "open_tickets": 0,
            "vip": False,
        },
        order_history=[
            {
                "order_id": "INV-2024-JAN",
                "date": "2024-01-01",
                "amount": 19.00,
                "status": "charged",
                "items": ["Starter plan renewal"],
                "refund_issued": False,
            },
            {
                "order_id": "INV-2024-FEB",
                "date": "2024-02-01",
                "amount": 19.00,
                "status": "charged",
                "items": ["Starter plan renewal"],
                "refund_issued": False,
            },
            {
                "order_id": "INV-2024-MAR",
                "date": "2024-03-01",
                "amount": 19.00,
                "status": "charged",
                "items": ["Starter plan renewal"],
                "refund_issued": False,
            },
            {
                "order_id": "INV-2024-APR",
                "date": "2024-04-01",
                "amount": 19.00,
                "status": "charged",
                "items": ["Starter plan renewal"],
                "refund_issued": False,
            },
        ],
        kb_articles=[
            {
                "title": "Refund eligibility policy",
                "content": (
                    "Refunds are only issued within 30 days of the charge. "
                    "Usage data shows this customer logged in 47 times across those 4 months. "
                    "Services that have been consumed are not eligible for refund."
                ),
                "relevance": "high",
            },
            {
                "title": "How to cancel a subscription",
                "content": "Go to Account > Subscription > Cancel Plan. Cancellation takes effect at end of billing cycle.",
                "relevance": "distractor",
            },
        ],
        applicable_policies={
            "refund_policy": (
                "Refunds are issued for: (1) verified duplicate charges, "
                "(2) charges within 30 days that the customer did not use the service. "
                "Refunds are processed within 3-5 business days to the original payment method. "
                "No refunds are issued for consumed services beyond 30 days."
            ),
        },
        ground_truth=GroundTruth(
            category="billing",
            priority="medium",
            team="billing",
            required_action=None,
            action_eligible=False,
            must_escalate=False,
            must_gather_evidence=["view_order_history", "check_policy"],
            required_response_facts=["policy", "30 days", "cancel"],
            prohibited_claims=["refund", "reimburse", "credit back"],
            sla_deadline_steps=None,
            response_keywords=["policy", "cannot", "cancel"],
        ),
    ),

    Scenario(
        scenario_id="SCEN-004",
        family="billing_ineligible",
        ticket_id="TKT-B004",
        customer_name="Marcus Lee",
        customer_tier="free",
        subject="Annual plan refund after 8 months",
        body=(
            "I signed up for the annual plan in August. It's now April and I want to cancel "
            "and get a refund for the remaining months. I've decided to go with a competitor. "
            "How do I get my refund?"
        ),
        customer_profile={
            "account_age_months": 8,
            "plan": "annual_basic",
            "abuse_history": [],
            "sla_tier": "standard",
            "open_tickets": 0,
            "vip": False,
        },
        order_history=[
            {
                "order_id": "INV-2023-AUG",
                "date": "2023-08-01",
                "amount": 120.00,
                "status": "charged",
                "items": ["Annual Basic plan (12 months)"],
                "refund_issued": False,
            },
        ],
        kb_articles=[
            {
                "title": "Annual plan cancellation policy",
                "content": (
                    "Annual plans are non-refundable after the initial 30-day trial period. "
                    "Customers who cancel mid-term retain access until the end of the annual period "
                    "but are not entitled to a partial refund."
                ),
                "relevance": "high",
            },
            {
                "title": "Switching from annual to monthly",
                "content": "You can switch to monthly billing at the end of your annual term in Account > Subscription.",
                "relevance": "distractor",
            },
        ],
        applicable_policies={
            "refund_policy": (
                "Refunds are issued for: (1) verified duplicate charges, "
                "(2) charges within 30 days that the customer did not use the service. "
                "No refunds are issued for consumed services beyond 30 days."
            ),
            "annual_plan_policy": (
                "Annual plans are non-refundable after the 30-day grace period. "
                "Cancellation mid-term retains access to end of term but no partial refund is issued."
            ),
        },
        ground_truth=GroundTruth(
            category="billing",
            priority="low",
            team="billing",
            required_action=None,
            action_eligible=False,
            must_escalate=False,
            must_gather_evidence=["view_order_history", "check_policy"],
            required_response_facts=["annual plan", "non-refundable", "access until"],
            prohibited_claims=["refund", "partial refund", "reimburse"],
            sla_deadline_steps=None,
            response_keywords=["policy", "annual", "access"],
        ),
    ),

    # ════════════════════════════════════════════════════════════════
    # ACCOUNT SECURITY — REAL THREAT
    # ════════════════════════════════════════════════════════════════

    Scenario(
        scenario_id="SCEN-005",
        family="security_real",
        ticket_id="TKT-A001",
        customer_name="Rachel Brown",
        customer_tier="pro",
        subject="Someone accessed my account — didn't authorise this login",
        body=(
            "I just got an email saying my account was logged in from Romania at 3am. "
            "I am in the US and have not traveled. I did NOT authorise this. "
            "My account has payment methods saved — please help immediately!"
        ),
        customer_profile={
            "account_age_months": 14,
            "plan": "pro",
            "abuse_history": [],
            "sla_tier": "standard",
            "open_tickets": 0,
            "vip": False,
            "last_login_ip": "185.220.101.45",  # known TOR exit node
            "last_login_country": "RO",
            "login_attempts_last_24h": 7,
            "password_changed_at": "2024-04-07T03:12:00Z",
            "payment_methods_on_file": 2,
        },
        order_history=[],
        kb_articles=[
            {
                "title": "Compromised account response protocol",
                "content": (
                    "If an account shows signs of compromise (unauthorized login, unknown IP, "
                    "unexpected password change): (1) lock the account immediately, "
                    "(2) escalate to security team, (3) do NOT send customer-facing response "
                    "until account is secured."
                ),
                "relevance": "high",
            },
            {
                "title": "How to enable two-factor authentication",
                "content": "2FA can be enabled under Account > Security > Two-Factor Authentication.",
                "relevance": "distractor",
            },
        ],
        applicable_policies={
            "account_security_policy": (
                "Accounts with confirmed unauthorized access MUST be locked before any other action. "
                "The security team must be notified within 15 minutes. "
                "Do not send a customer response before the account is secured. "
                "Failure to follow this protocol is a compliance violation."
            ),
            "fraud_escalation_policy": (
                "Any account with payment methods on file AND signs of compromise must be "
                "escalated to the fraud team immediately. Locking the account is mandatory."
            ),
        },
        ground_truth=GroundTruth(
            category="account",
            priority="urgent",
            team="support",
            required_action="lock_account",
            action_eligible=True,
            must_escalate=True,
            must_gather_evidence=["view_customer", "check_policy"],
            required_response_facts=["locked", "security team", "safe"],
            prohibited_claims=["cannot lock", "will not escalate"],
            sla_deadline_steps=4,
            response_keywords=["secured", "escalated", "locked"],
        ),
    ),

    Scenario(
        scenario_id="SCEN-006",
        family="security_real",
        ticket_id="TKT-A002",
        customer_name="Tom Martinez",
        customer_tier="enterprise",
        subject="Unauthorized API calls on our account — possible breach",
        body=(
            "Our security monitoring detected 5 unauthorized API calls to our production "
            "environment in the last hour using our API key. We did NOT make these calls. "
            "Total unexpected charges so far: ~$2,400. This is a potential breach. "
            "Rotate our API key and investigate immediately."
        ),
        customer_profile={
            "account_age_months": 31,
            "plan": "enterprise",
            "abuse_history": [],
            "sla_tier": "enterprise",
            "open_tickets": 0,
            "vip": True,
            "last_login_ip": "203.0.113.99",  # unknown external IP
            "login_attempts_last_24h": 5,
            "api_calls_last_hour": 5,
            "unexpected_charges_usd": 2400,
        },
        order_history=[
            {
                "order_id": "INV-ENT-APR-USAGE",
                "date": "2024-04-07",
                "amount": 2400.00,
                "status": "pending",
                "items": ["API usage overage"],
                "refund_issued": False,
            },
        ],
        kb_articles=[
            {
                "title": "Enterprise breach response protocol",
                "content": (
                    "For enterprise accounts with suspected API key compromise: "
                    "(1) immediately revoke and rotate the compromised API key, "
                    "(2) lock account pending investigation, "
                    "(3) escalate to security team within 5 minutes, "
                    "(4) waive fraudulent usage charges after investigation confirms breach."
                ),
                "relevance": "high",
            },
            {
                "title": "API rate limits and usage",
                "content": "API rate limits can be reviewed in the developer portal under Settings > API.",
                "relevance": "distractor",
            },
        ],
        applicable_policies={
            "account_security_policy": (
                "Accounts with confirmed unauthorized access MUST be locked before any other action. "
                "The security team must be notified within 15 minutes."
            ),
            "fraud_escalation_policy": (
                "Suspected API key compromise with unauthorized charges must be escalated immediately. "
                "Charges resulting from confirmed breaches are waived after investigation."
            ),
        },
        ground_truth=GroundTruth(
            category="account",
            priority="urgent",
            team="support",
            required_action="lock_account",
            action_eligible=True,
            must_escalate=True,
            must_gather_evidence=["view_customer", "view_order_history", "check_policy"],
            required_response_facts=["api key", "locked", "escalated", "security"],
            prohibited_claims=["charges will stand", "cannot waive"],
            sla_deadline_steps=4,
            response_keywords=["secured", "escalated", "investigate"],
        ),
    ),

    # ════════════════════════════════════════════════════════════════
    # ACCOUNT SECURITY — FALSE ALARM
    # ════════════════════════════════════════════════════════════════

    Scenario(
        scenario_id="SCEN-007",
        family="security_false",
        ticket_id="TKT-A003",
        customer_name="Grace Kim",
        customer_tier="pro",
        subject="Got login alert from new location — is my account safe?",
        body=(
            "I received a login notification from Portland, OR. "
            "I normally login from Seattle. I just moved last week so this was me, "
            "but wanted to check if my account is okay. No suspicious activity noticed."
        ),
        customer_profile={
            "account_age_months": 22,
            "plan": "pro",
            "abuse_history": [],
            "sla_tier": "standard",
            "open_tickets": 0,
            "vip": False,
            "last_login_ip": "73.94.20.1",   # residential US IP
            "last_login_country": "US",
            "login_attempts_last_24h": 1,
            "password_changed_at": None,
            "payment_methods_on_file": 1,
        },
        order_history=[],
        kb_articles=[
            {
                "title": "Login from new location alerts",
                "content": (
                    "Login alerts are triggered when a user logs in from a new city or country. "
                    "If the login was made by the account owner, no action is needed. "
                    "Advise customer to review their active sessions and enable 2FA if not already set."
                ),
                "relevance": "high",
            },
            {
                "title": "Compromised account response protocol",
                "content": (
                    "If an account shows signs of compromise (unauthorized login, unknown IP, "
                    "unexpected password change): (1) lock the account immediately, "
                    "(2) escalate to security team."
                ),
                "relevance": "distractor",
            },
        ],
        applicable_policies={
            "account_security_policy": (
                "Accounts with confirmed unauthorized access MUST be locked before any other action. "
                "If the customer confirms the login was their own, no lock is required. "
                "Recommend enabling 2FA as a precaution."
            ),
        },
        ground_truth=GroundTruth(
            category="account",
            priority="medium",
            team="support",
            required_action=None,
            action_eligible=False,   # locking would be wrong
            must_escalate=False,
            must_gather_evidence=["view_customer"],
            required_response_facts=["confirmed safe", "2fa", "sessions"],
            prohibited_claims=["locked your account", "escalated to security"],
            sla_deadline_steps=None,
            response_keywords=["safe", "2fa", "sessions"],
        ),
    ),

    # ════════════════════════════════════════════════════════════════
    # TECHNICAL — KNOWN ISSUE (KB WORKAROUND EXISTS)
    # ════════════════════════════════════════════════════════════════

    Scenario(
        scenario_id="SCEN-008",
        family="tech_known",
        ticket_id="TKT-T001",
        customer_name="David Park",
        customer_tier="pro",
        subject="App crashes when uploading files larger than 10 MB",
        body=(
            "Whenever I try to upload a file larger than 10 MB the app crashes "
            "with 'Upload failed: server error 500'. "
            "This started after your last update last week. Tested on Chrome 122 and Firefox 123. "
            "I need to upload a 25 MB file urgently for a client."
        ),
        customer_profile={
            "account_age_months": 9,
            "plan": "pro",
            "abuse_history": [],
            "sla_tier": "standard",
            "open_tickets": 0,
            "vip": False,
        },
        order_history=[],
        kb_articles=[
            {
                "title": "File upload size limit bug (v2.4.1)",
                "content": (
                    "Known issue introduced in v2.4.1: files larger than 10 MB fail with error 500. "
                    "Root cause: server-side upload buffer reduced during infrastructure migration. "
                    "Workaround: compress the file to under 10 MB before uploading, or use the "
                    "bulk upload API endpoint (/api/v1/upload/bulk) which has a 100 MB limit. "
                    "Fix scheduled for v2.4.2 (ETA: 2 weeks)."
                ),
                "relevance": "high",
            },
            {
                "title": "Supported file formats",
                "content": "We support PDF, DOCX, XLSX, PNG, JPG, and ZIP files up to 100 MB.",
                "relevance": "distractor",
            },
            {
                "title": "How to clear browser cache",
                "content": "Clearing browser cache can resolve many upload issues. Press Ctrl+Shift+Delete.",
                "relevance": "distractor",
            },
        ],
        applicable_policies={
            "bug_response_policy": (
                "For known bugs with existing workarounds, provide the workaround immediately. "
                "Do not promise a specific fix date unless confirmed in the KB article. "
                "If the ETA is known, share it with the customer."
            ),
        },
        ground_truth=GroundTruth(
            category="technical",
            priority="high",
            team="engineering",
            required_action=None,
            action_eligible=False,
            must_escalate=False,
            must_gather_evidence=["search_kb"],
            required_response_facts=["workaround", "compress", "bulk upload api"],
            prohibited_claims=["we have no record", "not a known issue"],
            sla_deadline_steps=None,
            response_keywords=["workaround", "compress", "known issue"],
        ),
    ),

    Scenario(
        scenario_id="SCEN-009",
        family="tech_known",
        ticket_id="TKT-T002",
        customer_name="James Wilson",
        customer_tier="free",
        subject="API returning 401 errors — key looks correct",
        body=(
            "Hi, my API key keeps getting 401 Unauthorized responses even though "
            "I copy-pasted it directly from my dashboard. I haven't changed anything. "
            "This broke yesterday — my automation scripts stopped working."
        ),
        customer_profile={
            "account_age_months": 4,
            "plan": "free",
            "abuse_history": [],
            "sla_tier": "standard",
            "open_tickets": 0,
            "vip": False,
        },
        order_history=[],
        kb_articles=[
            {
                "title": "API key format change in v2.4.0",
                "content": (
                    "Breaking change in v2.4.0: API keys now use the prefix 'sk-v2-' instead of 'sk-'. "
                    "Old keys ('sk-*') were deprecated on April 5, 2024. "
                    "Action required: regenerate your API key from the dashboard. "
                    "New keys will have the format: sk-v2-XXXXXXXXXXXXXXXX"
                ),
                "relevance": "high",
            },
            {
                "title": "Where to find your API key",
                "content": "API keys are in Settings > Developer > API Keys. Click 'Reveal' to view.",
                "relevance": "distractor",
            },
        ],
        applicable_policies={
            "api_deprecation_policy": (
                "Breaking API changes are announced 30 days in advance via email and release notes. "
                "Support should provide regeneration instructions and confirm the customer received "
                "the deprecation notice."
            ),
        },
        ground_truth=GroundTruth(
            category="technical",
            priority="medium",
            team="support",
            required_action=None,
            action_eligible=False,
            must_escalate=False,
            must_gather_evidence=["search_kb"],
            required_response_facts=["regenerate", "sk-v2", "new format"],
            prohibited_claims=["cannot help", "contact developer"],
            sla_deadline_steps=None,
            response_keywords=["regenerate", "api key", "new format"],
        ),
    ),

    # ════════════════════════════════════════════════════════════════
    # TECHNICAL — NEW ISSUE (NO KB ENTRY, MUST ESCALATE)
    # ════════════════════════════════════════════════════════════════

    Scenario(
        scenario_id="SCEN-010",
        family="tech_new",
        ticket_id="TKT-T003",
        customer_name="Lisa Anderson",
        customer_tier="enterprise",
        subject="URGENT: /api/v2/process returning 500 for 30% of requests — production down",
        body=(
            "URGENT: Our production system is partially down. "
            "The /api/v2/process endpoint has returned HTTP 500 errors for approximately "
            "30% of requests for the past 2 hours. This is directly impacting our customers. "
            "Error message: 'Internal processing error: queue overflow'. "
            "We need immediate assistance — this is P0 for us."
        ),
        customer_profile={
            "account_age_months": 36,
            "plan": "enterprise",
            "abuse_history": [],
            "sla_tier": "enterprise",
            "open_tickets": 0,
            "vip": True,
            "sla_response_minutes": 30,
        },
        order_history=[],
        kb_articles=[
            {
                "title": "Common API error codes",
                "content": "400: bad request. 401: unauthorized. 403: forbidden. 404: not found. 429: rate limit.",
                "relevance": "distractor",
            },
            {
                "title": "How to monitor API health",
                "content": "Check our status page at status.example.com for live API health metrics.",
                "relevance": "distractor",
            },
        ],
        applicable_policies={
            "enterprise_sla_policy": (
                "Enterprise customers with a 30-minute SLA must receive a response within 30 minutes. "
                "P0 incidents (production impacted) must be escalated to engineering on-call immediately. "
                "The support agent must not attempt to resolve P0 incidents without engineering involvement."
            ),
            "escalation_policy": (
                "Issues not covered in the knowledge base that affect production systems must be "
                "escalated to engineering. Do not promise fixes or timelines without engineering confirmation."
            ),
        },
        ground_truth=GroundTruth(
            category="technical",
            priority="urgent",
            team="engineering",
            required_action="escalate",
            action_eligible=True,
            must_escalate=True,
            must_gather_evidence=["search_kb", "view_customer"],
            required_response_facts=["escalated", "engineering", "30 minutes"],
            prohibited_claims=["cannot escalate", "known issue", "workaround"],
            sla_deadline_steps=3,
            response_keywords=["escalated", "P0", "engineering"],
        ),
    ),

    # ════════════════════════════════════════════════════════════════
    # ENTERPRISE SLA PRESSURE
    # ════════════════════════════════════════════════════════════════

    Scenario(
        scenario_id="SCEN-011",
        family="enterprise_sla",
        ticket_id="TKT-E001",
        customer_name="Carlos Mendez",
        customer_tier="enterprise",
        subject="Critical: dashboard not loading for entire team — 47 users affected",
        body=(
            "Our entire team of 47 users cannot load the analytics dashboard. "
            "The page spins indefinitely then shows 'Failed to load data'. "
            "This started at 9:15 AM EST. It's now 10:00 AM and we have a board presentation at 11 AM. "
            "This is unacceptable for an enterprise customer."
        ),
        customer_profile={
            "account_age_months": 24,
            "plan": "enterprise",
            "abuse_history": [],
            "sla_tier": "enterprise",
            "open_tickets": 0,
            "vip": True,
            "sla_response_minutes": 30,
            "active_users": 47,
        },
        order_history=[],
        kb_articles=[
            {
                "title": "Dashboard loading issues — caching",
                "content": "Try clearing browser cache or using an incognito window.",
                "relevance": "distractor",
            },
            {
                "title": "Enterprise incident escalation",
                "content": (
                    "If dashboard is down for more than 5 users simultaneously, this constitutes "
                    "a service incident. Escalate immediately to the engineering on-call team. "
                    "Do not suggest workarounds for service-level incidents."
                ),
                "relevance": "high",
            },
        ],
        applicable_policies={
            "enterprise_sla_policy": (
                "Enterprise customers with a 30-minute SLA must receive a response within 30 minutes. "
                "P0 incidents (production impacted) must be escalated to engineering on-call immediately."
            ),
        },
        ground_truth=GroundTruth(
            category="technical",
            priority="urgent",
            team="engineering",
            required_action="escalate",
            action_eligible=True,
            must_escalate=True,
            must_gather_evidence=["view_customer", "search_kb"],
            required_response_facts=["escalated", "engineering", "priority"],
            prohibited_claims=["clear cache", "try incognito", "working on your end"],
            sla_deadline_steps=3,
            response_keywords=["escalated", "incident", "engineering"],
        ),
    ),

    Scenario(
        scenario_id="SCEN-012",
        family="enterprise_sla",
        ticket_id="TKT-E002",
        customer_name="Priya Sharma",
        customer_tier="enterprise",
        subject="Need urgent review of our contract — renewal meeting in 4 hours",
        body=(
            "We have a contract renewal meeting with our CFO in 4 hours. "
            "We need a revised quote for 80 seats (up from 55) with the new compliance add-on. "
            "Our account manager has been unresponsive for 2 days. "
            "Please escalate — we're at risk of losing this renewal."
        ),
        customer_profile={
            "account_age_months": 42,
            "plan": "enterprise",
            "abuse_history": [],
            "sla_tier": "enterprise",
            "open_tickets": 2,
            "vip": True,
            "sla_response_minutes": 30,
            "contract_renewal_date": "2024-04-15",
        },
        order_history=[
            {
                "order_id": "INV-ENT-2024-MAR",
                "date": "2024-03-01",
                "amount": 5500.00,
                "status": "charged",
                "items": ["Enterprise plan — 55 seats"],
                "refund_issued": False,
            },
        ],
        kb_articles=[
            {
                "title": "Enterprise renewal process",
                "content": (
                    "Enterprise renewals must be processed by the sales team. "
                    "For urgent renewals within 24 hours, flag as urgent and assign to sales. "
                    "Support agents cannot issue quotes directly."
                ),
                "relevance": "high",
            },
            {
                "title": "How to add seats to enterprise plan",
                "content": "Contact your account manager or the sales team to add seats.",
                "relevance": "distractor",
            },
        ],
        applicable_policies={
            "enterprise_sla_policy": (
                "Enterprise customers must receive a response within 30 minutes."
            ),
            "sales_escalation_policy": (
                "Contract renewals, upsells, and quote requests must be routed to sales immediately. "
                "For urgent requests (meeting within 24 hours), mark as high priority and assign sales lead."
            ),
        },
        ground_truth=GroundTruth(
            category="billing",
            priority="urgent",
            team="sales",
            required_action="escalate",
            action_eligible=True,
            must_escalate=True,
            must_gather_evidence=["view_customer", "search_kb"],
            required_response_facts=["sales team", "urgent", "quote"],
            prohibited_claims=["cannot help", "wait for account manager"],
            sla_deadline_steps=3,
            response_keywords=["sales", "urgent", "renewal"],
        ),
    ),

    # ════════════════════════════════════════════════════════════════
    # FEEDBACK
    # ════════════════════════════════════════════════════════════════

    Scenario(
        scenario_id="SCEN-013",
        family="feedback",
        ticket_id="TKT-F001",
        customer_name="Alex Turner",
        customer_tier="free",
        subject="Feature request: dark mode + keyboard shortcuts",
        body=(
            "Love the app but really need dark mode — the bright white is brutal for late-night work. "
            "Also, keyboard shortcuts for common actions would be a huge productivity boost. "
            "Would pay for a pro plan if these were added!"
        ),
        customer_profile={
            "account_age_months": 3,
            "plan": "free",
            "abuse_history": [],
            "sla_tier": "standard",
            "open_tickets": 0,
            "vip": False,
        },
        order_history=[],
        kb_articles=[
            {
                "title": "How to submit feature requests",
                "content": "Feature requests can be upvoted on our public roadmap at roadmap.example.com.",
                "relevance": "high",
            },
        ],
        applicable_policies={},
        ground_truth=GroundTruth(
            category="feedback",
            priority="low",
            team="support",
            required_action=None,
            action_eligible=False,
            must_escalate=False,
            must_gather_evidence=[],
            required_response_facts=["roadmap", "thank", "feature request"],
            prohibited_claims=["will not implement", "not planned"],
            sla_deadline_steps=None,
            response_keywords=["thank", "roadmap", "feedback"],
        ),
    ),

    # ════════════════════════════════════════════════════════════════
    # SHIPPING
    # ════════════════════════════════════════════════════════════════

    Scenario(
        scenario_id="SCEN-014",
        family="shipping",
        ticket_id="TKT-S001",
        customer_name="Amy Thompson",
        customer_tier="free",
        subject="Order #98765 hasn't moved in 7 days — where is my package??",
        body=(
            "I placed order #98765 on March 28th. It's now April 8th and the tracking shows "
            "'In transit — departed Chicago sorting facility' for the past 7 days with no updates. "
            "I need this item for a project this Friday. What is going on??"
        ),
        customer_profile={
            "account_age_months": 6,
            "plan": "free",
            "abuse_history": [],
            "sla_tier": "standard",
            "open_tickets": 0,
            "vip": False,
        },
        order_history=[
            {
                "order_id": "ORD-98765",
                "date": "2024-03-28",
                "amount": 89.99,
                "status": "in_transit_stalled",
                "items": ["Wireless Keyboard Pro"],
                "refund_issued": False,
                "tracking_last_update": "2024-04-01",
                "carrier": "UPS",
                "tracking_number": "1Z999AA1234567890",
            },
        ],
        kb_articles=[
            {
                "title": "Stalled shipment investigation process",
                "content": (
                    "Shipments with no tracking update for 5+ business days are considered stalled. "
                    "Steps: (1) verify via order history, (2) open a carrier trace with the carrier, "
                    "(3) if no resolution within 2 business days, offer reship or refund. "
                    "Investigations typically take 3-5 business days."
                ),
                "relevance": "high",
            },
            {
                "title": "Standard shipping timelines",
                "content": "Standard shipping takes 5-7 business days. Express takes 2-3 business days.",
                "relevance": "distractor",
            },
        ],
        applicable_policies={
            "lost_shipment_policy": (
                "Shipments stalled for 7+ days with no tracking movement qualify for a trace investigation. "
                "If the trace confirms the package is lost, a replacement or full refund is issued. "
                "Traces must be opened with the carrier before any reship or refund is approved."
            ),
        },
        ground_truth=GroundTruth(
            category="shipping",
            priority="high",
            team="logistics",
            required_action="escalate",
            action_eligible=True,
            must_escalate=False,
            must_gather_evidence=["view_order_history"],
            required_response_facts=["trace", "investigation", "3-5 business days"],
            prohibited_claims=["cannot help", "contact carrier yourself"],
            sla_deadline_steps=None,
            response_keywords=["investigate", "trace", "logistics"],
        ),
    ),

    Scenario(
        scenario_id="SCEN-015",
        family="shipping",
        ticket_id="TKT-S002",
        customer_name="Kevin O'Brien",
        customer_tier="pro",
        subject="Wrong item received — I ordered Blue Widget Pro, got Red Widget Basic",
        body=(
            "I just opened my delivery and the wrong item was sent. "
            "I ordered Blue Widget Pro (SKU: BW-PRO-001) but received Red Widget Basic (SKU: RW-BAS-001). "
            "I have a project deadline tomorrow and desperately need the correct item. "
            "Please send the right one ASAP and send a return label for this one."
        ),
        customer_profile={
            "account_age_months": 11,
            "plan": "pro",
            "abuse_history": [],
            "sla_tier": "standard",
            "open_tickets": 0,
            "vip": False,
        },
        order_history=[
            {
                "order_id": "ORD-45321",
                "date": "2024-04-04",
                "amount": 149.99,
                "status": "delivered",
                "items": ["Blue Widget Pro (BW-PRO-001)"],
                "refund_issued": False,
                "carrier": "FedEx",
                "tracking_number": "274899172137",
                "delivered_date": "2024-04-07",
            },
        ],
        kb_articles=[
            {
                "title": "Wrong item received — resolution process",
                "content": (
                    "For wrong-item deliveries: (1) verify order history to confirm what was ordered, "
                    "(2) arrange same-day or next-day reship of the correct item, "
                    "(3) email a pre-paid return label for the incorrect item. "
                    "Customer is not responsible for return shipping costs."
                ),
                "relevance": "high",
            },
            {
                "title": "Return policy",
                "content": "Items can be returned within 30 days for a full refund. Customer pays return shipping for change-of-mind returns.",
                "relevance": "distractor",
            },
        ],
        applicable_policies={
            "wrong_item_policy": (
                "Wrong-item cases: the correct item is reshipped at no additional charge, same-day if possible. "
                "A pre-paid return label is emailed for the incorrect item. "
                "Customer does not pay return shipping. Verify the order in order history before reshipping."
            ),
        },
        ground_truth=GroundTruth(
            category="shipping",
            priority="high",
            team="logistics",
            required_action=None,
            action_eligible=False,
            must_escalate=False,
            must_gather_evidence=["view_order_history"],
            required_response_facts=["reship", "return label", "correct item"],
            prohibited_claims=["cannot reship", "customer pays return"],
            sla_deadline_steps=None,
            response_keywords=["reship", "return label", "correct"],
        ),
    ),
]

# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

_BY_ID: Dict[str, Scenario] = {s.scenario_id: s for s in SCENARIOS}
_BY_FAMILY: Dict[str, List[Scenario]] = {}
for _s in SCENARIOS:
    _BY_FAMILY.setdefault(_s.family, []).append(_s)


def get_scenario(scenario_id: str) -> Optional[Scenario]:
    return _BY_ID.get(scenario_id)


def get_scenarios_by_family(family: str) -> List[Scenario]:
    return _BY_FAMILY.get(family, [])


# Task → families mapping
TASK_FAMILIES = {
    "classify_and_route": ["billing_eligible", "billing_ineligible", "feedback"],
    "investigate_and_resolve": [
        "billing_eligible", "billing_ineligible", "tech_known", "shipping",
        "security_false",
    ],
    "complex_operations": [
        "security_real", "enterprise_sla", "tech_new",
    ],
}

TASK_MAX_STEPS = {
    "classify_and_route": 4,
    "investigate_and_resolve": 8,
    "complex_operations": 12,
}

TASK_DESCRIPTIONS = {
    "classify_and_route": (
        "Classify this support ticket: set category, priority, and assigned team. "
        "You may gather information before acting, but the ticket is straightforward. "
        "Close the ticket when done. Available actions: classify_ticket, assign, "
        "view_customer, search_kb, check_policy, draft_response, escalate, close_ticket."
    ),
    "investigate_and_resolve": (
        "Investigate this ticket before acting. Gather evidence (view_customer, "
        "view_order_history, search_kb, check_policy) to determine the correct resolution. "
        "Then classify, assign, apply any required action, draft a response, and close. "
        "Grading rewards evidence gathering and policy compliance."
    ),
    "complex_operations": (
        "High-stakes ticket requiring careful reasoning. Gather evidence, check policies, "
        "determine the correct action (which may include escalation or account locking). "
        "Wrong irreversible actions (refund when ineligible, locking when not warranted) "
        "carry significant penalties. SLA deadlines apply."
    ),
}
