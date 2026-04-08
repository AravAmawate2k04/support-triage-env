"""
Inference script — Smart Tool-Use Baseline for Support Operations Environment v2.
==================================================================================

Strategy: Gather evidence first, then act.
The agent follows a structured gather-then-act loop:
  1. view_customer
  2. view_order_history (if billing/shipping)
  3. search_kb (if technical/account)
  4. check_policy (if action required)
  5. classify_ticket
  6. assign (priority + team)
  7. apply_action / escalate (if needed)
  8. draft_response
  9. close_ticket

At each step the LLM decides the next action based on the full observation
(ticket + all retrieved info so far). This should score significantly higher
than the simple baseline on medium/hard tasks.

Environment variables: same as inference.py

STDOUT FORMAT
-------------
[START] task=<name> env=support_triage model=<model>
[STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,...,rn>
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME: str = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL") or "http://127.0.0.1:8000"

BENCHMARK = "support_triage"
TEMPERATURE = 0.2
MAX_TOKENS = 1000
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = ["classify_and_route", "investigate_and_resolve", "complex_operations"]
DEFAULT_SEEDS = [7, 42, 99]


def _parse_eval_seeds() -> List[int]:
    raw = os.getenv("EVAL_SEEDS", "")
    if not raw.strip():
        return DEFAULT_SEEDS
    seeds: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    return seeds or DEFAULT_SEEDS

# ---------------------------------------------------------------------------
# Logging helpers (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str, seed: int) -> None:
    print(f"[START] task={task} seed={seed} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action.replace(chr(10), ' ')[:200]} "
        f"reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt: describes all available actions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a customer-support operations agent. You resolve support tickets step-by-step
    using available actions.

    AVAILABLE ACTIONS (use one per turn):

    Retrieval (gather evidence before acting):
      {"action_type": "view_customer"}
      {"action_type": "view_order_history"}
      {"action_type": "search_kb", "query": "<search terms>"}
      {"action_type": "check_policy", "policy_name": "<policy name>"}

    Classification (required before closing):
      {"action_type": "classify_ticket", "category": "billing|technical|account|feedback|shipping"}
      {"action_type": "assign", "priority": "low|medium|high|urgent", "assigned_team": "billing|support|engineering|sales|logistics"}

    Resolution:
      {"action_type": "draft_response", "response_text": "<full customer-facing reply>"}
      {"action_type": "apply_action", "applied_action": "refund|lock_account|reship_order|waive_charges"}
      {"action_type": "escalate", "escalation_reason": "<specific reason>"}
      {"action_type": "close_ticket"}

    STRATEGY:
    1. For billing tickets: view_order_history → check_policy → decide if action eligible → classify → assign → draft → close
    2. For technical tickets: search_kb → classify → assign → draft (with workaround or escalate if not in KB) → close
    3. For account/security: view_customer → check_policy → (lock if real threat) → escalate if required → classify → assign → draft → close
    4. For shipping: view_order_history → classify → assign → draft → close
    5. For feedback: classify → assign → draft → close (no retrieval needed)
    6. ALWAYS check policy before applying irreversible actions (refund, lock_account)
    7. Escalate if: security breach, enterprise P0, or issue not in KB

    RULES:
    - Must classify_ticket and assign before close_ticket
    - draft_response must greet customer by name, acknowledge issue, give resolution, end with "Best regards, Support Team"
    - Respond ONLY with a JSON object (one action per turn). No extra text.
""").strip()


def build_observation_prompt(obs: Dict[str, Any], history: List[Dict[str, str]]) -> str:
    """Build a user message from the current observation."""
    lines = [
        f"=== TICKET ===",
        f"ID: {obs.get('ticket_id', '')}",
        f"Customer: {obs.get('customer_name', '')} ({obs.get('customer_tier', '')} tier)",
        f"Subject: {obs.get('subject', '')}",
        f"Body: {obs.get('body', '')}",
        "",
        f"=== TASK ===",
        f"Task: {obs.get('task_name', '')}",
        f"Step: {obs.get('step', 0)} / {obs.get('max_steps', 12)}",
        f"Available actions: {obs.get('available_actions', [])}",
    ]

    if obs.get("customer_profile"):
        lines += ["", "=== CUSTOMER PROFILE (retrieved) ==="]
        for k, v in obs["customer_profile"].items():
            lines.append(f"  {k}: {v}")

    if obs.get("order_history"):
        lines += ["", "=== ORDER HISTORY (retrieved) ==="]
        for order in obs["order_history"]:
            lines.append(f"  {order}")

    if obs.get("kb_results"):
        lines += ["", "=== KB RESULTS (retrieved) ==="]
        for article in obs["kb_results"]:
            lines.append(f"  [{article.get('relevance','?')}] {article.get('title','')}")
            lines.append(f"    {article.get('content','')[:200]}")

    if obs.get("policy_text"):
        lines += ["", "=== POLICY (retrieved) ===", obs["policy_text"]]

    if obs.get("feedback"):
        lines += ["", f"=== LAST ACTION FEEDBACK ===", obs["feedback"]]

    if obs.get("cumulative_reward") is not None:
        lines.append(f"\nCumulative reward so far: {obs.get('cumulative_reward', 0):.3f}")

    lines += ["", "What is your next action? Respond with a single JSON object."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smart tool-use episode runner
# ---------------------------------------------------------------------------

async def run_task(task: str, base_url: str, seed: int) -> None:
    from openenv.core.generic_client import GenericEnvClient

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg: Optional[str] = None

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME, seed=seed)

    client_openai = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    # Running conversation history for the LLM
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async with GenericEnvClient(base_url=base_url) as env:
        try:
            step_result = await env.reset(task=task, seed=seed)
            obs: Dict[str, Any] = step_result.observation  # type: ignore[assignment]
            done = step_result.done

            while not done:
                # Build user message from current observation
                user_msg = build_observation_prompt(obs, messages)
                messages.append({"role": "user", "content": user_msg})

                # Get next action from LLM
                action_dict = _get_next_action(client_openai, messages)

                # Append assistant response to history
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(action_dict, ensure_ascii=False),
                })

                # Execute the action
                step_result = await env.step(action_dict)
                obs = step_result.observation  # type: ignore[assignment]
                reward = float(step_result.reward or 0.0)
                done = step_result.done

                rewards.append(reward)
                steps_taken += 1

                # Build readable log string
                log_action = json.dumps(
                    {k: v for k, v in action_dict.items() if k != "response_text"},
                    ensure_ascii=False,
                )
                if "response_text" in action_dict:
                    rt = str(action_dict.get("response_text", ""))[:60].replace("\n", " ")
                    log_action = log_action[:-1] + f', "response_text": "{rt}..."}}'

                log_step(step=steps_taken, action=log_action, reward=reward, done=done, error=error_msg)

                if done:
                    break

                # Safety: stop if we've used too many messages (context management)
                if len(messages) > 40:
                    # Force close
                    step_result = await env.step({"action_type": "close_ticket"})
                    reward = float(step_result.reward or 0.0)
                    done = step_result.done
                    rewards.append(reward)
                    steps_taken += 1
                    log_step(steps_taken, '{"action_type": "close_ticket"}', reward, done, "forced close: context limit")
                    break

        except Exception as exc:
            error_msg = str(exc)[:120]
            print(f"[DEBUG] Episode error: {exc}", flush=True)
        finally:
            score = rewards[-1] if rewards else 0.0
            success = score >= SUCCESS_SCORE_THRESHOLD
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def _get_next_action(client: OpenAI, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Call LLM and parse the next action JSON."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,  # type: ignore[arg-type]
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip code fences if present
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        parsed = json.loads(text)
        if "action_type" not in parsed:
            raise ValueError(
                f"LLM response missing 'action_type' field. Raw text: {text[:200]}"
            )
        return parsed

    except Exception as exc:
        # Log raw output for debugging, then re-raise so the episode fails
        # loudly rather than silently degrading to a close_ticket no-op.
        print(f"[DEBUG] LLM action parse failed: {exc}", flush=True)
        raise


# ---------------------------------------------------------------------------
# Server lifecycle (same as inference.py)
# ---------------------------------------------------------------------------

_SERVER_PROC: Optional[subprocess.Popen] = None  # type: ignore[type-arg]


def start_local_server() -> None:
    global _SERVER_PROC
    env_dir = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable
    _SERVER_PROC = subprocess.Popen(
        [python, "-m", "uvicorn", "server.app:app",
         "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"],
        cwd=env_dir,
    )
    import urllib.request
    for _ in range(30):
        time.sleep(1)
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
            return
        except Exception:
            pass
    raise RuntimeError("Server failed to start within 30 s")


def stop_local_server() -> None:
    global _SERVER_PROC
    if _SERVER_PROC is not None:
        _SERVER_PROC.send_signal(signal.SIGTERM)
        try:
            _SERVER_PROC.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _SERVER_PROC.kill()
        _SERVER_PROC = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    needs_local_server = ENV_BASE_URL in (
        "http://127.0.0.1:8000", "http://localhost:8000"
    ) and not os.getenv("ENV_BASE_URL")

    if needs_local_server:
        print("[DEBUG] Starting local environment server ...", flush=True)
        start_local_server()
        print("[DEBUG] Server ready.", flush=True)

    try:
        seeds = _parse_eval_seeds()
        for seed in seeds:
            for task in TASKS:
                await run_task(task=task, base_url=ENV_BASE_URL, seed=seed)
    finally:
        if needs_local_server:
            stop_local_server()


if __name__ == "__main__":
    asyncio.run(main())
