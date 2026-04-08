"""
Inference script for the Support Triage Environment.
====================================================

Runs all three tasks against the live OpenEnv server and emits structured
logs in the mandatory [START] / [STEP] / [END] format.

Environment variables
---------------------
API_BASE_URL    LLM inference endpoint   (default: HuggingFace router)
MODEL_NAME      Model identifier         (default: Qwen/Qwen2.5-72B-Instruct)
HF_TOKEN        HuggingFace / API key    (required)
ENV_BASE_URL    Running env server URL   (default: http://127.0.0.1:8000 — starts locally)

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

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        """Gracefully skip .env loading when python-dotenv is unavailable."""
        return False

from openai import OpenAI

# Load variables from .env file (values already in the environment take precedence)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME: str = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL") or "http://127.0.0.1:8000"

BENCHMARK = "support_triage"
MAX_STEPS = 1          # Single-step episodes
TEMPERATURE = 0.2
MAX_TOKENS = 600
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = ["categorize", "triage", "respond"]
TASK_SEEDS = {
    "categorize": 101,
    "triage": 202,
    "respond": 303,
}

# ---------------------------------------------------------------------------
# Logging helpers (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Keep action on one line; truncate if very long
    action_safe = action.replace("\n", " ")[:200]
    print(
        f"[STEP] step={step} action={action_safe} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM prompts per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "categorize": textwrap.dedent("""
        You are a customer-support triage system.
        Given a support ticket, classify it into EXACTLY ONE category.
        Valid categories: billing, technical, account, feedback, shipping

        Respond with a JSON object with a single field:
        {"category": "<one of the valid categories>"}

        Do not include any explanation or extra fields.
    """).strip(),

    "triage": textwrap.dedent("""
        You are a customer-support triage system.
        Given a support ticket, output a JSON object with these EXACT fields:
        - category: one of billing, technical, account, feedback, shipping
        - priority: one of low, medium, high, urgent
        - assigned_team: one of billing, support, engineering, sales, logistics

        Priority guidance:
          urgent = production down / locked out / delivery overdue
          high   = significant disruption, billing error, wrong item
          medium = account changes, subscription cancellation
          low    = questions, feature requests, compliments

        Team routing:
          billing    → payment / invoice / subscription issues
          support    → password, account access, general questions, feedback
          engineering → bugs, API errors, crashes
          sales      → pricing, enterprise, upgrades
          logistics  → delivery, shipping, wrong item

        Respond ONLY with the JSON object. No extra text.
    """).strip(),

    "respond": textwrap.dedent("""
        You are a friendly, professional customer-support agent.
        Given a support ticket plus any relevant context/policies, do TWO things:

        1. Triage the ticket (JSON fields: category, priority, assigned_team).
        2. Write a complete customer-facing reply (JSON field: response_text).

        The reply MUST:
        - Start with a greeting that uses the customer's first name
        - Acknowledge the specific issue
        - Provide clear, actionable resolution steps
        - Reference any relevant policy or FAQ if provided
        - End with a professional closing (e.g. "Best regards, Support Team")

        Respond with a JSON object containing:
        {
          "category": "...",
          "priority": "...",
          "assigned_team": "...",
          "response_text": "..."
        }

        No extra text outside the JSON.
    """).strip(),
}


def build_user_prompt(obs: Dict[str, Any]) -> str:
    """Build the user-facing prompt from an observation dict."""
    ticket_block = (
        f"Ticket ID: {obs.get('ticket_id', '')}\n"
        f"Customer: {obs.get('customer_name', '')} ({obs.get('customer_tier', '')} tier)\n"
        f"Subject: {obs.get('subject', '')}\n\n"
        f"{obs.get('body', '')}"
    )
    context = obs.get("context", {})
    if context:
        context_lines = "\n".join(f"  {k}: {v}" for k, v in context.items())
        ticket_block += f"\n\n--- Context & Policies ---\n{context_lines}"
    return ticket_block


def get_action_from_model(
    client: OpenAI,
    task: str,
    obs: Dict[str, Any],
) -> Dict[str, Any]:
    """Call the LLM and parse the JSON action."""
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task]},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Try to extract JSON from the response
        if "```" in text:
            # Strip code fences
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]

        parsed = json.loads(text)
        return parsed

    except Exception:
        # Return safe fallback action
        fallback: Dict[str, Any] = {"category": "technical"}
        if task in ("triage", "respond"):
            fallback["priority"] = "medium"
            fallback["assigned_team"] = "support"
        if task == "respond":
            fallback["response_text"] = (
                f"Dear {obs.get('customer_name', 'Customer').split()[0]},\n\n"
                "Thank you for contacting us. We have received your request and will "
                "follow up shortly.\n\nBest regards,\nSupport Team"
            )
        return fallback


# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------

_SERVER_PROC: Optional[subprocess.Popen] = None  # type: ignore[type-arg]


def start_local_server() -> None:
    """Start the uvicorn server as a background subprocess."""
    global _SERVER_PROC

    env_dir = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable

    _SERVER_PROC = subprocess.Popen(
        [
            python, "-m", "uvicorn",
            "server.app:app",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--log-level", "warning",
        ],
        cwd=env_dir,
    )

    # Wait until the server responds to /health
    import urllib.request
    health_url = "http://127.0.0.1:8000/health"
    for _ in range(30):
        time.sleep(1)
        try:
            urllib.request.urlopen(health_url, timeout=2)
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
# Main episode runner
# ---------------------------------------------------------------------------

async def run_task(
    task: str,
    base_url: str,
) -> None:
    """Run a single-task episode and emit [START]/[STEP]/[END] logs."""
    from openenv.core.generic_client import GenericEnvClient

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg: Optional[str] = None

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    client_openai = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    async with GenericEnvClient(base_url=base_url) as env:
        try:
            # reset() kwargs are forwarded to environment's reset(task=...)
            step_result = await env.reset(task=task, seed=TASK_SEEDS[task])
            obs: Dict[str, Any] = step_result.observation  # type: ignore[assignment]

            for step in range(1, MAX_STEPS + 1):
                if step_result.done:
                    break

                # Get action from the model
                action_dict = get_action_from_model(client_openai, task, obs)

                # Execute step
                step_result = await env.step(action_dict)
                obs = step_result.observation  # type: ignore[assignment]

                reward = float(step_result.reward or 0.0)
                done = step_result.done

                rewards.append(reward)
                steps_taken = step

                action_str = json.dumps(
                    {k: v for k, v in action_dict.items() if k != "response_text"},
                    ensure_ascii=False,
                )
                if "response_text" in action_dict:
                    rt = str(action_dict["response_text"])[:60].replace("\n", " ")
                    action_str = action_str[:-1] + f', "response_text": "{rt}..."}}'

                log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

                if done:
                    break

        except Exception as exc:
            error_msg = str(exc)[:120]
            if steps_taken == 0:
                steps_taken = 1
                log_step(step=1, action="{}", reward=0.0, done=True, error=error_msg)
        finally:
            score = rewards[-1] if rewards else 0.0
            success = score >= SUCCESS_SCORE_THRESHOLD
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    # Decide whether to start a local server or connect to a running one
    needs_local_server = ENV_BASE_URL in (
        "http://127.0.0.1:8000", "http://localhost:8000"
    ) and not os.getenv("ENV_BASE_URL")

    if needs_local_server:
        start_local_server()

    try:
        for task in TASKS:
            await run_task(task=task, base_url=ENV_BASE_URL)
    finally:
        if needs_local_server:
            stop_local_server()


if __name__ == "__main__":
    asyncio.run(main())
