---
title: Support Triage Environment
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - nlp
---

# Support Triage Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible environment where an AI agent triages customer support tickets — a task performed millions of times daily across every industry.

Built for the **Meta OpenEnv Hackathon** as a complete, real-world environment that satisfies every requirement in the spec: typed models, 3 graded tasks, partial-credit reward, baseline inference, Docker deployment, and HuggingFace Spaces hosting.

---

## Why This Environment?

Customer support triage is one of the most common AI use-cases in production today. Every company that has customers needs to:

1. **Classify** what kind of issue is being reported
2. **Prioritise** how urgent it is and route it to the right team
3. **Respond** with a professional, helpful reply

This environment lets you train and evaluate language model agents on exactly that workflow — with 15 realistic hand-crafted tickets, deterministic graders, and a reward function that gives useful signal at every step, not just at the end.

---

## Live Demo

| Resource | URL |
|---|---|
| HF Space | https://huggingface.co/spaces/AravAmawate/support-triage-env |
| API (live) | https://aravamawate-support-triage-env.hf.space |
| Swagger UI | https://aravamawate-support-triage-env.hf.space/docs |

---

## The Three Tasks

| Task | Difficulty | What the agent does | Reward signal |
|------|-----------|---------------------|---------------|
| `categorize` | Easy | Classify the ticket into one of 5 categories | 1.0 exact, 0.5 alias, 0.0 wrong |
| `triage` | Medium | Category + urgency priority + team routing | Weighted partial credit across 3 fields |
| `respond` | Hard | Full triage + write a professional customer reply | 8-criterion rubric, keyword coverage drives 40% |

Each task is a **single-step episode** — the agent gets one ticket, takes one action, receives a score. This keeps episodes fast and evaluation clean.

---

## Action Space

```python
class SupportTriageAction(Action):
    category:       str            # billing | technical | account | feedback | shipping
    priority:       Optional[str]  # low | medium | high | urgent          (triage + respond)
    assigned_team:  Optional[str]  # billing | support | engineering | sales | logistics
    response_text:  Optional[str]  # full customer-facing reply             (respond only)
```

## Observation Space

```python
class SupportTriageObservation(Observation):
    ticket_id:        str    # unique ticket identifier e.g. "TKT-005"
    customer_name:    str    # customer full name
    customer_tier:    str    # free | pro | enterprise
    subject:          str    # email subject line
    body:             str    # full email body text
    task_name:        str    # which task is active: categorize | triage | respond
    task_description: str    # natural-language instructions for the agent
    step:             int    # step count within this episode
    context:          dict   # relevant FAQs and policies (respond task only)
    feedback:         str    # grader explanation of what was right/wrong (after step)
    score_breakdown:  dict   # per-criterion scores (after step)
    done:             bool   # True after the first step (single-step episodes)
    reward:           float  # score in [0.0, 1.0] (None at reset)
```

---

## Reward Function

All rewards are in **[0.0, 1.0]** and give partial-credit signal — not just binary win/lose.

### categorize (easy)
| Criterion | Score |
|-----------|-------|
| Exact category match | 1.0 |
| Alias / related category | 0.5 |
| Wrong category | 0.0 |

### triage (medium)
| Criterion | Weight |
|-----------|--------|
| Correct category | 0.40 |
| Correct priority | 0.30 (0.12 for adjacent level) |
| Correct team | 0.30 |

### respond (hard)
| Criterion | Weight | How it's measured |
|-----------|--------|-------------------|
| Correct category | 0.10 | string match |
| Correct priority | 0.05 | string match |
| Correct team | 0.05 | string match |
| Greeting uses customer name | 0.10 | first name in first 50 chars |
| Customer name anywhere | 0.10 | first name in full response |
| Keyword coverage | 0.40 | fraction of required keywords present |
| Proper structure | 0.10 | has greeting word AND closing word |
| Appropriate length | 0.10 | 80–1000 chars = full credit |

The **keyword coverage** criterion (0.40 weight) directly measures whether the response addresses the actual issue. Each ticket has 3 required keywords — e.g. a double-charge ticket requires `["refund", "duplicate", "apologize"]`. No LLM judge needed — fully deterministic string matching.

---

## Ticket Dataset

15 hand-crafted realistic tickets, 3 per category:

| Category | Tickets |
|----------|---------|
| `billing` | Double charge, subscription cancellation, enterprise pricing inquiry |
| `technical` | File upload crash (500 error), API outage (urgent), Slack integration question |
| `account` | Password reset not arriving, email transfer, 2FA lockout (urgent) |
| `feedback` | Dark mode request, support team compliment, CSV export suggestion |
| `shipping` | Late delivery, wrong item received, address change request |

Each ticket has: ground-truth `category`, `priority`, `team`, `required_keywords`, `category_aliases` (for partial credit), and `context` (policies/FAQs shown only for the respond task).

---

## Setup

### Prerequisites
- Python 3.10+
- Docker

### Local Installation

```bash
git clone https://huggingface.co/spaces/AravAmawate/support-triage-env
cd support-triage-env

python -m venv venv
source venv/bin/activate

pip install openenv-core fastapi uvicorn pydantic openai
pip install -e .
```

### Run the server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run via Docker

```bash
docker build -t support-triage-env .
docker run -p 7860:7860 support-triage-env
curl http://localhost:7860/health
```

---

## Usage

### Try it in the browser

Open the **interactive Swagger UI** — no code needed:
```
https://aravamawate-support-triage-env.hf.space/docs
```
Click any endpoint → "Try it out" → fill in the fields → Execute.

### HTTP API (curl)

```bash
# Start an episode (pick a task and optional seed for reproducibility)
curl -s -X POST https://aravamawate-support-triage-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task":"triage","seed":42}'

# Submit an action (all on one line)
curl -s -X POST https://aravamawate-support-triage-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"category":"billing","priority":"high","assigned_team":"billing"}}'
```

> **Note:** The HTTP `/step` is stateless — each call creates a fresh environment.
> Use the Python WebSocket client for connected reset→step episodes.

### Python Client (WebSocket — stateful)

```python
import asyncio
from client import SupportTriageEnv
from models import SupportTriageAction

async def main():
    async with SupportTriageEnv(base_url="ws://localhost:8000") as env:

        # Task 1: categorize
        result = await env.reset(task="categorize", seed=42)
        print(result.observation["subject"])
        result = await env.step({"category": "feedback"})
        print(f"reward={result.reward:.3f}")   # 1.000 if correct
        print(result.observation["feedback"])  # explains what was right/wrong

        # Task 2: triage
        result = await env.reset(task="triage", seed=42)
        result = await env.step({
            "category": "feedback",
            "priority": "low",
            "assigned_team": "support"
        })
        print(f"reward={result.reward:.3f}")   # 1.000 if all 3 fields correct

asyncio.run(main())
```

### Interactive Manual Test

```bash
python test_env.py
```

Asks you to pick a task, shows you the ticket, accepts your answer, prints score and feedback.

---

## Running the Baseline Inference Script

```bash
export HF_TOKEN=your_huggingface_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct          # optional
export API_BASE_URL=https://router.huggingface.co/v1  # optional

python inference.py
```

The script:
1. Automatically starts a local uvicorn server
2. Connects via WebSocket
3. Runs one episode per task using the specified LLM
4. Emits mandatory structured logs

**Actual output from a real run:**

```
[DEBUG] Starting local environment server ...
[DEBUG] Server ready.
[START] task=categorize env=support_triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"category": "account"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1.00
[START] task=triage env=support_triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"category": "account", "priority": "high", "assigned_team": "support"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1.00
[START] task=respond env=support_triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"category": "Feature Request", "priority": "Medium", "assigned_team": "Product Development", "response_text": "Hello Carlos,  Thank you for reaching out and sharing your s..."} reward=0.87 done=true error=null
[END] success=true steps=1 score=0.870 rewards=0.87
```

---

## Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Actual Score | Analysis |
|------|-------------|----------|
| `categorize` | **1.000** | Perfect — model correctly identified the ticket category |
| `triage` | **1.000** | Perfect — correct category, priority, and team routing |
| `respond` | **0.870** | Strong response; lost points on category capitalisation (`Feature Request` vs `feedback`) and non-standard team name (`Product Development` vs `support`) |

The `respond` task's 0.87 score shows the rubric is working correctly — the model wrote a good customer reply and captured keywords, but the triage fields were slightly off.

---

## Validation

All 3 official validator checks pass:

```
✅ Step 1 — HF Space is live and responds to /reset
✅ Step 2 — Docker build succeeded
✅ Step 3 — openenv validate passed
```

Run it yourself:
```bash
bash validate.py https://aravamawate-support-triage-env.hf.space .
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | HuggingFace / API key for LLM calls |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM inference endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model to use for inference |
| `ENV_BASE_URL` | No | `http://127.0.0.1:8000` | Override to use a remote env server |

---

## Project Structure

```
.
├── models.py                          # Pydantic Action + Observation types
├── client.py                          # Typed WebSocket client (SupportTriageEnv)
├── __init__.py                        # Package exports
├── server/
│   ├── app.py                         # FastAPI app with landing page
│   └── support_triage_environment.py  # Environment class + ticket dataset + graders
├── openenv.yaml                       # OpenEnv spec (spec_version, app, port)
├── pyproject.toml                     # Package metadata + server entry point
├── uv.lock                            # Reproducible dependency lockfile
├── Dockerfile                         # Container (port 7860 for HF Spaces)
├── inference.py                       # Baseline LLM inference script
├── test_env.py                        # Interactive manual testing script
├── push_to_hf.py                      # HuggingFace Space deployment helper
├── TECHNICAL.md                       # Deep-dive into implementation details
└── README.md                          # This file
```

---

## OpenEnv Compliance

```bash
# Validate local structure
openenv validate .
# → [OK] : Ready for multi-mode deployment

# Validate running server
openenv validate --url https://aravamawate-support-triage-env.hf.space
```
