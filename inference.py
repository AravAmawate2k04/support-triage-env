"""
Inference script — Simple Baseline for the Support Operations Environment v2.
==============================================================================

Strategy: One-shot classification.
The agent immediately classifies, assigns, drafts a response, and closes —
WITHOUT using any retrieval actions (view_customer, search_kb, etc.).

This is the *simple* baseline. It scores well on easy tasks but poorly on
medium/hard where evidence gathering and policy compliance are rewarded.
See inference_tool_use.py for the smart baseline.

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
MAX_STEPS = 12
TEMPERATURE = 0.2
MAX_TOKENS = 800
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


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
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
# One-shot system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a customer-support agent. Given a support ticket, you must:

    1. Classify the ticket:
       - category: billing | technical | account | feedback | shipping
       - priority: low | medium | high | urgent
       - assigned_team: billing | support | engineering | sales | logistics

    2. Write a professional customer-facing reply (response_text) that:
       - Starts with a greeting using the customer's first name
       - Acknowledges the specific issue
       - Provides clear, actionable resolution steps
       - References relevant policy or workaround if you know one
       - Ends with "Best regards, Support Team"

    Respond ONLY with a JSON object. No extra text. No code fences.

    {
      "category": "...",
      "priority": "...",
      "assigned_team": "...",
      "response_text": "..."
    }
""").strip()


def build_user_prompt(obs: Dict[str, Any]) -> str:
    parts = [
        f"Ticket ID: {obs.get('ticket_id', '')}",
        f"Customer: {obs.get('customer_name', '')} ({obs.get('customer_tier', '')} tier)",
        f"Subject: {obs.get('subject', '')}",
        "",
        obs.get("body", ""),
    ]
    return "\n".join(parts)


def get_action_from_model(
    client: OpenAI,
    obs: Dict[str, Any],
) -> Dict[str, Any]:
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()

        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        return json.loads(text)

    except Exception as exc:
        # Log raw model output for debugging, then re-raise so the episode
        # fails loudly rather than silently scoring a dummy action.
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        raise


# ---------------------------------------------------------------------------
# Multi-step episode runner — simple one-shot strategy
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

    async with GenericEnvClient(base_url=base_url) as env:
        try:
            step_result = await env.reset(task=task, seed=seed)
            obs: Dict[str, Any] = step_result.observation  # type: ignore[assignment]

            # One LLM call, then execute 4 fixed steps
            llm_action = get_action_from_model(client_openai, obs)

            action_sequence = [
                {
                    "action_type": "classify_ticket",
                    "category": llm_action.get("category", "technical"),
                },
                {
                    "action_type": "assign",
                    "priority": llm_action.get("priority", "medium"),
                    "assigned_team": llm_action.get("assigned_team", "support"),
                },
                {
                    "action_type": "draft_response",
                    "response_text": llm_action.get("response_text", ""),
                },
                {"action_type": "close_ticket"},
            ]

            for step_num, action_dict in enumerate(action_sequence, start=1):
                if steps_taken >= MAX_STEPS:
                    break

                step_result = await env.step(action_dict)
                obs = step_result.observation  # type: ignore[assignment]
                reward = float(step_result.reward or 0.0)
                done = step_result.done

                rewards.append(reward)
                steps_taken = step_num

                log_action = json.dumps(
                    {k: v for k, v in action_dict.items() if k != "response_text"},
                    ensure_ascii=False,
                )
                if "response_text" in action_dict:
                    rt = str(action_dict.get("response_text", ""))[:60].replace("\n", " ")
                    log_action = log_action[:-1] + f', "response_text": "{rt}..."}}'

                log_step(
                    step=step_num,
                    action=log_action,
                    reward=reward,
                    done=done,
                    error=error_msg,
                )

                if done:
                    break

        except Exception as exc:
            error_msg = str(exc)[:120]
            print(f"[DEBUG] Episode error: {exc}", flush=True)
        finally:
            score = rewards[-1] if rewards else 0.0
            success = score >= SUCCESS_SCORE_THRESHOLD
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

_SERVER_PROC: Optional[subprocess.Popen] = None  # type: ignore[type-arg]


def start_local_server() -> None:
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
