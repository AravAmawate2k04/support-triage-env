# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An OpenEnv-compatible environment where an AI agent triages customer support tickets. Built for the Meta OpenEnv Hackathon. The environment exposes three single-step tasks (categorize, triage, respond) with deterministic graders and partial-credit reward in [0.0, 1.0].

## Commands

```bash
# Run the server locally (development)
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Run via Docker (HF Spaces uses port 7860)
docker build -t support-triage-env .
docker run -p 7860:7860 support-triage-env

# Run inference (requires HF_TOKEN env var or .env file)
python inference.py

# Interactive manual test (connects to live HF Space by default)
python test_env.py
python test_env.py --url http://localhost:8000

# Validate submission (requires Docker + openenv-core)
bash validate.sh https://aravamawate-support-triage-env.hf.space .

# OpenEnv validation
openenv validate .

# Push to HuggingFace Spaces
python push_to_hf.py
```

## Architecture

The project follows the OpenEnv spec with a client-server split:

- **`models.py`** - Pydantic `SupportTriageAction` and `SupportTriageObservation` extending OpenEnv's `Action`/`Observation` base types. All tasks share the same action model; `priority`, `assigned_team`, and `response_text` are optional fields used by harder tasks.

- **`server/support_triage_environment.py`** - Core environment class (`SupportTriageEnvironment`) extending OpenEnv's `Environment` interface. Contains: the 15-ticket dataset (3 per category), task descriptions, grading functions, and reward weight configs. `reset()` picks a random ticket; `step()` grades the action and returns a single-step episode result. All grading is deterministic string matching — no LLM judge.

- **`server/app.py`** - FastAPI app created via `openenv.core.env_server.http_server.create_app()`. Provides REST endpoints (`/reset`, `/step`, `/state`, `/schema`, `/health`) and a WebSocket endpoint (`/ws`) for stateful sessions. The `create_app` factory handles all routing; `app.py` just adds a landing page.

- **`client.py`** - `SupportTriageEnv` wraps OpenEnv's `EnvClient` with typed models. Used for programmatic access via WebSocket. Supports both async (`async with`) and sync (`.sync()`) usage.

- **`inference.py`** - Baseline script that starts a local uvicorn server, connects via `GenericEnvClient`, calls an LLM (OpenAI-compatible API) for each task, and emits structured `[START]`/`[STEP]`/`[END]` logs. The log format is mandatory for evaluation scoring.

- **`openenv.yaml`** - Spec metadata (spec_version 1, FastAPI runtime, port 7860).

## Key Design Decisions

- **Single-step episodes**: Every task completes in one step. `reset()` returns the ticket, `step()` grades and ends the episode.
- **Stateless HTTP vs stateful WS**: The HTTP `/step` endpoint creates a fresh env per call. For connected reset-then-step flows, use the WebSocket client.
- **Import fallbacks**: All modules have `try/except ImportError` blocks to support both package imports (`from ..models`) and direct execution (`from models`).
- **Dual port config**: Dockerfile exposes 7860 (HF Spaces requirement), but local dev uses 8000. `inference.py` starts its own server on 8000.

## Environment Variables

| Variable | Required | Default | Used by |
|----------|----------|---------|---------|
| `HF_TOKEN` | For inference | — | `inference.py` (as OpenAI API key) |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | `inference.py` |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | `inference.py` |
| `ENV_BASE_URL` | No | `http://127.0.0.1:8000` | `inference.py` (skip local server if set) |

Variables can be set in a `.env` file (loaded via `python-dotenv`).

## Grading System

Reward weights by task (defined as class-level dicts in `SupportTriageEnvironment`):
- **categorize**: category=1.0
- **triage**: category=0.40, priority=0.30, team=0.30
- **respond**: category=0.10, priority=0.05, team=0.05, has_greeting=0.10, has_customer_name=0.10, keyword_coverage=0.40, proper_structure=0.10, length_ok=0.10

Category matching gives 0.5 partial credit for aliases. Priority gives 0.4 for adjacent levels. All response criteria are deterministic string checks.

## Submission Checklist

1. HF Space returns 200 on POST `/reset`
2. `docker build` succeeds
3. `openenv validate` passes
4. `inference.py` runs under 20 min on vcpu=2, 8GB RAM
5. All 3 tasks produce grader scores in [0.0, 1.0]
