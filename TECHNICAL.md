# Technical Deep-Dive: Support Triage Environment

This document explains every technical decision made in this project —
how OpenEnv works internally, why the code is structured the way it is,
what problems we ran into and how we fixed them, and what you would need
to change to build your own environment from scratch.

---

## Table of Contents

1. [What is OpenEnv and Why It Matters](#1-what-is-openenv-and-why-it-matters)
2. [Project Goal and How We Met It](#2-project-goal-and-how-we-met-it)
3. [The OpenEnv Spec — What the Validator Checks](#3-the-openenv-spec)
4. [Pydantic Models — Action and Observation](#4-pydantic-models)
5. [The Environment Class](#5-the-environment-class)
6. [The HTTP Server — How create_app Works](#6-the-http-server)
7. [WebSocket Sessions vs HTTP Requests](#7-websocket-vs-http)
8. [The Client — EnvClient](#8-the-client)
9. [The Reward Function in Detail](#9-reward-function)
10. [The Inference Script](#10-the-inference-script)
11. [Packaging — pyproject.toml and uv.lock](#11-packaging)
12. [Docker and HuggingFace Spaces](#12-docker-and-huggingface-spaces)
13. [Problems We Hit and How We Fixed Them](#13-problems-we-hit-and-how-we-fixed-them)
14. [How All the Pieces Connect — Full Call Chain](#14-full-call-chain)
15. [Key Lessons if You Build Your Own](#15-key-lessons)

---

## 1. What is OpenEnv and Why It Matters

OpenEnv is a framework for building **reinforcement-learning environments**
that expose a standard API so any AI agent can interact with them.

It is inspired by OpenAI Gym (`gymnasium`) but designed specifically for
**language-model agents** rather than game-playing agents. Instead of pixel
observations and joystick actions, you have text observations and structured
JSON actions.

The mental model:

```
Agent                          Environment
  |                                 |
  |-------- reset(task=...) ------->|  "Give me a new episode"
  |<------- observation ------------|  "Here is the starting state"
  |                                 |
  |-------- step(action) ---------->|  "Here is my action"
  |<------- observation+reward -----|  "Here is what happened + your score"
  |                                 |
  |-------- step(action) ---------->|  (repeat until done=True)
```

One full loop from `reset()` to `done=True` is called an **episode**.

**Why it matters for AI development:**
- A standardised API means any agent (fine-tuned LLM, GPT-4, rule-based system)
  can be evaluated against the same environment without custom integration work.
- The reward signal can be used to train agents via reinforcement learning (RLHF, PPO, etc.)
  or simply to benchmark how capable a model is at a specific real-world task.
- OpenEnv environments deployed on HuggingFace become publicly discoverable benchmarks
  that the whole research community can use.

Our environment — **customer support ticket triage** — adds a genuinely useful
benchmark to the OpenEnv ecosystem. Companies spend enormous engineering effort
building and evaluating AI for this task. Having a public, standardised environment
makes it easier to compare approaches.

---

## 2. Project Goal and How We Met It

The hackathon required building a complete, real-world OpenEnv environment.
Here is every requirement and what we built to satisfy it:

| Requirement | What we built |
|---|---|
| Real-world task (not a game) | Customer support ticket triage — used in production by every company |
| Typed Pydantic models | `SupportTriageAction`, `SupportTriageObservation` in `models.py` |
| `step()` / `reset()` / `state()` | Implemented in `SupportTriageEnvironment` |
| `openenv.yaml` | Present at root with spec_version, app, port |
| 3+ tasks easy→hard | `categorize` / `triage` / `respond` |
| Graders scoring 0.0–1.0 | Deterministic rubrics in `_grade()` method |
| Partial-credit reward | Weighted multi-criterion scoring, not binary |
| `inference.py` in root | Present, uses OpenAI client, emits `[START]/[STEP]/[END]` |
| Uses `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` | All three read from env vars |
| Dockerfile | Present, builds cleanly, tested |
| HF Space deployed | Live at `aravamawate-support-triage-env.hf.space` |
| README with all required sections | Present |
| Official validator passes | ✅ All 3/3 checks pass |

---

## 3. The OpenEnv Spec

### `openenv.yaml`

Every OpenEnv project must have this file at the root:

```yaml
spec_version: 1
name: support_triage
type: space
runtime: fastapi
app: server.app:app   # Python import path to the FastAPI app object
port: 7860            # port the server listens on
```

`openenv validate .` reads this file first. If it is missing, validation stops immediately.

### What `openenv validate` Actually Checks

The validator (`openenv/cli/_validation.py`) runs `validate_multi_mode_deployment()`:

```python
def validate_multi_mode_deployment(env_path: Path) -> tuple[bool, list[str]]:
    issues = []

    # 1. pyproject.toml must exist
    if not (env_path / "pyproject.toml").exists():
        issues.append("Missing pyproject.toml")
        return False, issues

    # 2. uv.lock must exist (reproducible installs)
    if not (env_path / "uv.lock").exists():
        issues.append("Missing uv.lock")

    # 3. pyproject.toml must have [project.scripts] with a "server" entry
    scripts = pyproject["project"]["scripts"]
    if "server" not in scripts:
        issues.append("Missing [project.scripts] server entry point")

    # 4. server entry point must reference a main() function
    if ":main" not in scripts["server"]:
        issues.append("Server entry point should reference main function")

    # 5. openenv-core must be in dependencies
    if not any(dep.startswith("openenv") for dep in deps):
        issues.append("Missing required dependency: openenv-core")

    # 6. server/app.py must exist
    if not (env_path / "server" / "app.py").exists():
        issues.append("Missing server/app.py")

    # 7. server/app.py must have def main( and call main()
    if "def main(" not in app_content:
        issues.append("server/app.py missing main() function")
    if "main()" not in app_content:
        issues.append("main() function not callable")

    return len(issues) == 0, issues
```

**The tricky part — rule 7:** the validator does a raw string search for `"main()"`.
It does NOT parse the Python AST. This means `main(host=args.host)` does NOT satisfy
the check — only the bare string `main()` does.

We originally had:
```python
if __name__ == "__main__":
    main(host=args.host, port=args.port)   # ← FAILS the validator
```

Fixed to:
```python
if __name__ == "__main__":
    main()   # ← passes — the literal string "main()" is present
```

---

## 4. Pydantic Models

### Why Pydantic?

Pydantic is a Python library that validates data automatically using type annotations.
When you define a model, Pydantic:
- Rejects JSON missing required fields (returns HTTP 422)
- Accepts optional fields being absent (uses the default)
- Rejects unknown fields if `extra="forbid"` is set
- Converts types automatically (`"42"` → `42` for an `int` field)

This matters for an environment API because agents will send JSON that may be
malformed, missing fields, or have wrong types. Pydantic catches all of this
before your environment code ever sees the data.

### The Base Classes

`Action` and `Observation` are defined in `openenv.core.env_server.types`:

```python
class Action(BaseModel):
    model_config = ConfigDict(
        extra="forbid",           # unknown fields → HTTP 422
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", ...)

    done:     bool              # is the episode over?
    reward:   float | None      # reward from last action (None at reset)
    metadata: Dict[str, Any]
```

### Our Action Model

```python
class SupportTriageAction(Action):
    category:       str            # required for all tasks
    priority:       Optional[str]  # required for triage + respond
    assigned_team:  Optional[str]  # required for triage + respond
    response_text:  Optional[str]  # required for respond only
```

All four fields are defined even if only `category` is used for the easy task.
If `response_text` were not declared, an agent sending it would get a 422 error.
Declaring it as `Optional[str] = None` means it is accepted but defaults to None
when not provided.

### Our Observation Model

The observation carries everything the agent needs to take an action, plus
feedback after the step:

```python
class SupportTriageObservation(Observation):
    # The ticket (shown at reset, unchanged during episode)
    ticket_id, customer_name, customer_tier, subject, body

    # Task context
    task_name          # which task: categorize | triage | respond
    task_description   # plain-English instructions for the agent
    context            # policies/FAQs dict (only populated for respond task)

    # Episode tracking
    step               # step count (0 at reset, 1 after first step)

    # Grader output (empty at reset, populated after step)
    feedback           # plain-English explanation of what was right/wrong
    score_breakdown    # dict of criterion → float score
```

---

## 5. The Environment Class

### The Three Required Methods

```python
from openenv.core.env_server.interfaces import Environment

class SupportTriageEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def reset(self, task="categorize", seed=None, **kwargs):
        # Pick a random ticket, set the task, return initial observation
        ...

    def step(self, action: SupportTriageAction):
        # Grade the action, return observation with reward
        ...

    @property
    def state(self) -> State:
        return self._state   # episode_id + step_count
```

### `SUPPORTS_CONCURRENT_SESSIONS = True`

This flag tells the HTTP server it is safe to create multiple environment
instances running simultaneously (one per WebSocket client). If it were `False`,
setting `max_concurrent_envs > 1` would raise an error.

Our environment is safe to parallelise because all state is stored in instance
variables (`self._ticket`, `self._task`, `self._state`) — there are no shared
globals, no database connections, no file handles.

### How `reset()` Receives the `task` Argument

This is the most non-obvious part of the design. When the server receives:

```json
POST /reset
{"task": "triage", "seed": 42}
```

It creates a `ResetRequest`:

```python
class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="allow")   # ← accepts unknown fields
    seed: Optional[int] = None
    episode_id: Optional[str] = None
```

`extra="allow"` means `task` is stored even though it is not a declared field.
The server then uses Python's `inspect` module to forward only the kwargs
that your `reset()` actually accepts:

```python
kwargs = request.model_dump(exclude_unset=True)
# → {"task": "triage", "seed": 42}

sig = inspect.signature(env.reset)
valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
# → {"task": "triage", "seed": 42}   (both are in reset's signature)

observation = env.reset(**valid_kwargs)
# → env.reset(task="triage", seed=42)
```

**Practical implication:** you can add any custom parameter to your `reset()` method
and it will automatically be forwarded from the HTTP body or WebSocket message data.
No changes needed to the server code.

The same mechanism works over WebSocket — `WSResetMessage.data` is forwarded identically.

---

## 6. The HTTP Server

### `create_app()`

The entire FastAPI application is created by one function call:

```python
from openenv.core.env_server.http_server import create_app

app = create_app(
    SupportTriageEnvironment,   # class, not instance — called as factory
    SupportTriageAction,
    SupportTriageObservation,
    env_name="support_triage",
    max_concurrent_envs=4,
)
```

`create_app` instantiates an `HTTPEnvServer`, registers all routes on a `FastAPI()` app,
and returns it. You pass the environment **class** — not an instance. The server calls
`SupportTriageEnvironment()` (with no arguments) each time it needs a new instance.

### Registered Routes

| Route | Method | What it does |
|-------|--------|--------------|
| `/reset` | POST | Calls `env.reset(**kwargs)`, returns first observation |
| `/step` | POST | Calls `env.step(action)`, returns observation + reward + done |
| `/state` | GET | Returns `env.state` (episode_id + step_count) |
| `/schema` | GET | JSON Schema for Action, Observation, State |
| `/metadata` | GET | env name, description, version |
| `/health` | GET | `{"status": "healthy"}` |
| `/mcp` | POST | MCP JSON-RPC protocol for AI tool integrations |
| `/ws` | WebSocket | Persistent session — stateful multi-step episodes |
| `/openapi.json` | GET | Auto-generated OpenAPI spec |
| `/docs` | GET | Auto-generated Swagger UI |
| `/` | GET | Our custom HTML landing page |

We added the `/` route manually after `create_app()` because without it,
visiting the Space URL showed `{"detail":"Not Found"}` — confusing for users.

### Why HTTP `/step` Is Stateless

Looking at the step handler in the source:

```python
async def step_handler(request: StepRequest) -> StepResponse:
    _env = self._env_factory()    # ← FRESH environment, no prior state
    observation = await _env.step(action)
    return StepResponse(...)
    # _env is garbage-collected — state is lost
```

A new `SupportTriageEnvironment()` is created for every HTTP `/step` call.
It has no memory of any previous `/reset`. This is by design — HTTP is stateless.

We added an auto-initialise guard so `/step` works even without a prior `/reset`:

```python
def step(self, action):
    if self._ticket is None:
        self._ticket = self._rng.choice(TICKETS)   # pick a random ticket
```

For proper connected episodes (reset → step on the same ticket), use WebSocket.

---

## 7. WebSocket vs HTTP

| | HTTP | WebSocket |
|---|---|---|
| State | None — fresh env per call | Persistent — same env for whole session |
| Latency | Higher (new connection + env init) | Lower (connection reused) |
| Best for | Health checks, validation, quick tests | Agent inference, multi-step episodes |
| Used by | `curl`, browser, validators | `EnvClient`, `GenericEnvClient` |

### WebSocket Message Protocol

```json
// Client → reset
{"type": "reset", "data": {"task": "triage", "seed": 42}}

// Client → step
{"type": "step", "data": {"category": "billing", "priority": "high", "assigned_team": "billing"}}

// Client → get state
{"type": "state"}

// Server → observation response
{"type": "observation", "data": {"observation": {...}, "reward": 0.7, "done": true}}
```

The `type` field is a discriminator. Pydantic uses `Annotated[..., Field(discriminator="type")]`
to automatically parse each message into the correct class without if/else branching.

### Session Lifecycle

```
Client connects to ws://host/ws
    Server: new SupportTriageEnvironment() created

Client: {"type":"reset","data":{"task":"triage"}}
    Server: env.reset(task="triage") called
    Server: observation sent back

Client: {"type":"step","data":{"category":"billing",...}}
    Server: env.step(action) called
    Server: graded observation + reward sent back

Client disconnects
    Server: env.close() called, instance garbage-collected
```

---

## 8. The Client

### `EnvClient` — Three Abstract Methods

To build a typed client, inherit from `EnvClient` and implement three methods:

```python
class EnvClient(ABC, Generic[ActT, ObsT, StateT]):

    @abstractmethod
    def _step_payload(self, action: ActT) -> Dict[str, Any]:
        # Convert your typed Action → plain dict for the WebSocket message

    @abstractmethod
    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ObsT]:
        # Convert server's JSON response → your typed Observation

    @abstractmethod
    def _parse_state(self, payload: Dict[str, Any]) -> StateT:
        # Convert server's state JSON → your State object
```

The base class handles everything else: WebSocket connection management,
reconnection on failure, message serialisation/deserialisation, timeout handling,
and the async/sync wrapper.

### Our Implementation (`client.py`)

```python
class SupportTriageEnv(
    EnvClient[SupportTriageAction, SupportTriageObservation, State]
):
    def _step_payload(self, action: SupportTriageAction) -> Dict:
        payload = {"category": action.category}
        if action.priority is not None:
            payload["priority"] = action.priority
        if action.assigned_team is not None:
            payload["assigned_team"] = action.assigned_team
        if action.response_text is not None:
            payload["response_text"] = action.response_text
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[SupportTriageObservation]:
        obs_data = payload.get("observation", {})
        observation = SupportTriageObservation(
            ticket_id=obs_data.get("ticket_id", ""),
            customer_name=obs_data.get("customer_name", ""),
            # ... all fields mapped
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(observation=observation, reward=payload.get("reward"), done=...)
```

### `GenericEnvClient` in `inference.py`

`inference.py` uses `GenericEnvClient` instead of our typed `SupportTriageEnv`.
`GenericEnvClient` works with raw `Dict[str, Any]` — no typed models, no imports
of environment-specific packages. This avoids import path complexity in the inference script:

```python
# inference.py — no package import needed
async with GenericEnvClient(base_url=base_url) as env:
    result = await env.reset(task="categorize")
    obs = result.observation          # Dict[str, Any]
    result = await env.step({"category": "billing"})
    print(result.reward)              # float
```

---

## 9. Reward Function

### Why Partial Credit Is Important

Binary rewards (0 or 1) are extremely hard for RL agents to learn from.
If the agent gets 0 for everything except a perfect answer, it gets no
gradient signal and cannot improve incrementally.

Our reward function is designed to give signal at every partial improvement:

```
categorize:
  correct category      → 1.0
  related alias         → 0.5   ← tells agent "you're in the right area"
  completely wrong      → 0.0

triage:
  all 3 fields correct  → 1.0
  2 fields correct      → 0.4–0.7 depending on which ones
  1 field correct       → 0.3–0.4
  everything wrong      → 0.0
```

### The `respond` Task Rubric

The respond task grades 8 separate criteria, each independently:

```python
_WEIGHTS_RESPOND = {
    "category":          0.10,   # did you classify correctly?
    "priority":          0.05,   # did you prioritise correctly?
    "team":              0.05,   # did you route to the right team?
    "has_greeting":      0.10,   # does the reply start with Dear/Hi/Hello?
    "has_customer_name": 0.10,   # is the customer's name used?
    "keyword_coverage":  0.40,   # does the reply address the actual issue?
    "proper_structure":  0.10,   # does it have both greeting AND closing?
    "length_ok":         0.10,   # is it 80–1000 chars? (not too short/long)
}
```

`keyword_coverage` at 0.40 weight is the dominant signal. It measures what fraction
of the ticket's required keywords appear in the response. For example, a double-charge
ticket requires `["refund", "duplicate", "apologize"]`. If the response contains
"refund" and "apologize" but not "duplicate", it gets 0.67 on this criterion.

**Why deterministic instead of LLM-as-judge?**
- LLM judges are non-deterministic — the same response gets different scores each run
- Every step would require an extra API call, making inference slow and expensive
- Our string-matching rubric is instant, free, and perfectly reproducible

The tradeoff is it can miss a high-quality response that uses synonyms (e.g. "sorry"
instead of "apologize"), but for a training/evaluation environment, reproducibility
matters more than catching every good response.

---

## 10. The Inference Script

### The Mandatory Log Format

The hackathon evaluators parse stdout for exactly these three line types:

```
[START] task=<name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
```

Rules enforced by our implementation:
- One `[START]` per episode, printed before the first step
- One `[STEP]` immediately after each `env.step()` returns
- One `[END]` after the episode ends — printed inside `finally:` so it always runs
- `reward` and `rewards` formatted to exactly 2 decimal places
- `done` and `success` are lowercase: `true` or `false` (not Python `True`/`False`)
- All fields on a single line — newlines in action strings are stripped

### Server Subprocess Pattern

Rather than requiring the user to start the server separately, `inference.py`
manages the server lifecycle itself:

```python
# Start the server
_SERVER_PROC = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "server.app:app",
     "--host", "127.0.0.1", "--port", "8000", "--log-level", "warning"],
    cwd=os.path.dirname(os.path.abspath(__file__))
)

# Poll until ready
for _ in range(30):
    time.sleep(1)
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
        return  # server is up
    except:
        pass

# Shut down when done
_SERVER_PROC.send_signal(signal.SIGTERM)
```

`sys.executable` instead of `"python"` ensures the subprocess uses the same
virtualenv Python as the main script — important when running inside a venv.

### LLM Integration

The inference script uses the OpenAI Python client pointed at HuggingFace's router:

```python
client = OpenAI(
    base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
    api_key=os.getenv("HF_TOKEN"),
)
```

Each task gets its own system prompt that instructs the model to output JSON:
- `categorize`: "output ONLY `{"category": "..."}` — one of the 5 valid categories"
- `triage`: "output JSON with exactly category, priority, assigned_team"
- `respond`: "output JSON with triage fields + response_text containing a full email"

`temperature=0.2` (low) makes the model more deterministic for reproducible scores.
The model outputs JSON directly — no regex extraction needed.

### Actual Baseline Results

Running `Qwen/Qwen2.5-72B-Instruct` against all 3 tasks:

```
[START] task=categorize env=support_triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"category": "account"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1.00

[START] task=triage env=support_triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"category": "account", "priority": "high", "assigned_team": "support"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1.00

[START] task=respond env=support_triage model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"category": "Feature Request", "priority": "Medium", "assigned_team": "Product Development", "response_text": "Hello Carlos,  Thank you..."} reward=0.87 done=true error=null
[END] success=true steps=1 score=0.870 rewards=0.87
```

The 0.87 on respond is interesting: the model wrote a high-quality customer reply
but used `"Feature Request"` (capitalised, with space) instead of `"feedback"` for the
category, and `"Product Development"` instead of `"support"` for the team. The rubric
correctly caught this — those criteria scored 0, but keyword coverage and response
structure scored well, giving 0.87 overall.

---

## 11. Packaging

### Why `pyproject.toml` Instead of `setup.py`

`pyproject.toml` is the modern Python packaging standard (PEP 517/518).
`openenv validate` specifically requires it — it will not accept `setup.py`.

### The `package-dir` Trick

Our code lives at the repo root, not inside a subdirectory named after the package.
To make it importable as `support_triage`, we tell setuptools to map the package name
to the current directory:

```toml
[tool.setuptools.package-dir]
"support_triage" = "."         # root dir IS the support_triage package
"support_triage.server" = "server"

[tool.setuptools]
packages = ["support_triage", "support_triage.server"]
```

After `pip install -e .`:
- `import support_triage` → reads `./models.py`, `./client.py`, `./__init__.py`
- `from support_triage.server.app import main` → reads `./server/app.py`

The `[project.scripts]` entry:
```toml
server = "support_triage.server.app:main"
```
Creates a `server` command that `openenv validate` checks for.

### `uv.lock`

`uv` is a fast Python package manager (written in Rust by Astral, the company
behind `ruff`). The lockfile pins every dependency to an exact version + hash.

`openenv validate` requires `uv.lock` to exist. Without it, validation fails.

Generated with:
```bash
/home/amw/snap/code/232/.local/bin/uv lock
```

(The uv binary location after installation — not yet on PATH, so full path needed.)

---

## 12. Docker and HuggingFace Spaces

### The Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir \
    "openenv-core>=0.2.2" fastapi uvicorn pydantic openai

RUN pip install --no-cache-dir -e .   # installs support_triage package

EXPOSE 7860

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

Key decisions:
- `python:3.11-slim` — smaller image (no dev tools, no docs)
- `pip install -e .` — installs the package so `import support_triage` resolves
- `--no-cache-dir` — avoids storing pip's download cache inside the image layer
- HEALTHCHECK — Docker and HF can detect when the app is genuinely ready
- Port **7860** — the only port HuggingFace Docker Spaces expose externally

### Why Port 7860 Specifically

HuggingFace Docker Spaces have a hardcoded proxy that forwards external HTTPS
traffic to port 7860 inside the container. If your app listens on any other port:
- The container starts and appears healthy to Docker
- The HF proxy tries to connect to port 7860 and gets "connection refused"
- The Space stays stuck at `APP_STARTING` indefinitely — no error message

We originally used port 8000 everywhere. The fix required changing three files:

```
Dockerfile    → CMD [..., "--port", "7860"]
openenv.yaml  → port: 7860
README.md     → app_port: 7860  (HF reads this from frontmatter)
```

### HuggingFace Space Architecture

```
Your browser / curl / evaluator
         │
         ▼  HTTPS port 443
  HuggingFace load balancer
         │
         ▼  HTTP port 7860
  Docker container (inside HF's infrastructure)
    └── uvicorn (Python ASGI server)
         └── FastAPI application
              └── SupportTriageEnvironment instances
```

HuggingFace terminates TLS at their load balancer. The container never sees
HTTPS — it only needs to handle plain HTTP on port 7860.

### Deployment Script (`push_to_hf.py`)

```python
from huggingface_hub import HfApi, create_repo

# Create the Space if it doesn't exist
create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

# Upload all files except secrets and build artifacts
api.upload_folder(
    folder_path=project_dir,
    repo_id=repo_id,
    repo_type="space",
    ignore_patterns=["venv/", "__pycache__/", "token.txt", "*.egg-info/", ".env"]
)
```

`upload_folder` computes a diff and only uploads changed files.
`ignore_patterns` is critical — HuggingFace's secret scanner scans every uploaded file
and blocks the push if it finds a valid API token. `token.txt` contained an HF token
and caused the first push to fail with a 400 error.

---

## 13. Problems We Hit and How We Fixed Them

These are the real issues encountered during development, in order.

### Problem 1 — `openenv validate` failed: missing `main()`

**Symptom:**
```
server/app.py main() function not callable (missing if __name__ == '__main__')
```

**Root cause:** The validator does a raw string search for `"main()"`.
Our `__main__` block called `main(host=args.host, port=args.port)` which does not
contain the string `"main()"`.

**Fix:** Changed to:
```python
if __name__ == "__main__":
    main()
```

**Lesson:** Read the validator source code, not just the error message. It is doing
simpler string matching than you might expect.

### Problem 2 — HTTP `/step` returned 500

**Symptom:** `curl /step` returned Internal Server Error.

**Root cause:** The HTTP server creates a fresh `SupportTriageEnvironment()` for every
call. The fresh instance has `self._ticket = None`. Our `step()` called
`assert ticket is not None` which raised `AssertionError`.

**Fix:** Auto-initialise the ticket when step() is called without a prior reset():
```python
def step(self, action):
    if self._ticket is None:
        self._ticket = self._rng.choice(TICKETS)
```

**Lesson:** HTTP endpoints are inherently stateless. If your environment has state,
either make step() handle the no-reset case, or document clearly that users must
use WebSocket for stateful episodes.

### Problem 3 — HF Space stuck at `APP_STARTING`

**Symptom:** Space deployed, but stayed at `APP_STARTING` for 20+ minutes and never became RUNNING.

**Root cause:** The app was listening on port 8000. HuggingFace proxies to port 7860.
The proxy kept getting "connection refused" and retrying indefinitely.

**Fix:** Changed port from 8000 to 7860 in:
- `Dockerfile` CMD
- `openenv.yaml`
- `README.md` frontmatter (`app_port: 7860`)

**Lesson:** HuggingFace Docker Spaces always require port 7860. This is documented
but easy to miss if you assume 8000 is the standard.

### Problem 4 — HF push blocked by secret scanner

**Symptom:**
```
Bad request for commit endpoint: ... Offending files: token.txt
```

**Root cause:** `token.txt` in the project directory contained a valid HF token.
HuggingFace's push API scans all uploaded files for tokens and blocks the commit.

**Fix:** Added `token.txt` to `ignore_patterns` in `push_to_hf.py` and to `.gitignore`.

**Lesson:** Never put tokens in files at the repo root. Use `.env` files (gitignored)
or environment variables directly.

### Problem 5 — Root URL showed `{"detail":"Not Found"}`

**Symptom:** Visiting the Space URL in a browser showed a JSON error instead of anything useful.

**Root cause:** FastAPI returns a 404 for any route not explicitly registered.
`create_app()` does not register a `/` route.

**Fix:** Added a custom HTML landing page after `create_app()`:
```python
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> HTMLResponse:
    return HTMLResponse(content="<html>... landing page ...</html>")
```

### Problem 6 — Docker permission denied

**Symptom:** `docker build` returned `permission denied on /var/run/docker.sock`.

**Root cause:** User `amw` was not in the `docker` group.

**Fix:**
```bash
sudo usermod -aG docker $USER   # add to docker group permanently
newgrp docker                   # activate in current shell
```

Then used `sg docker -c "..."` to run Docker commands from within Claude Code's
process (which was started before the group change).

---

## 14. Full Call Chain

Here is every function call traced from `python inference.py` to a graded reward:

```
python inference.py
│
├── start_local_server()
│     └── subprocess.Popen(["uvicorn", "server.app:app", "--port", "8000"])
│           └── server/app.py is imported
│                 └── create_app(SupportTriageEnvironment, ...) called
│                       └── FastAPI app created with all routes registered
│
├── GenericEnvClient(base_url="ws://127.0.0.1:8000") connected
│     └── WebSocket connects to /ws
│           └── server creates new SupportTriageEnvironment() instance
│
├── await env.reset(task="categorize")
│     └── client sends: {"type":"reset","data":{"task":"categorize"}}
│     └── server: ResetRequest parsed, kwargs={"task":"categorize"}
│     └── server: inspect.signature(env.reset) → valid_kwargs = {"task":"categorize"}
│     └── server: env.reset(task="categorize") called
│           └── self._task = "categorize"
│           └── self._ticket = random.choice(TICKETS)   e.g. TKT-007
│           └── returns SupportTriageObservation(ticket_id="TKT-007", done=False, reward=None)
│     └── server serialises observation → JSON sent over WebSocket
│     └── client receives JSON → _parse_result() → StepResult[Dict]
│
├── get_action_from_model(openai_client, "categorize", obs)
│     └── builds prompt: "Ticket: TKT-007\nSubject: Can't reset password\n..."
│     └── client.chat.completions.create(model="Qwen/...", messages=[...])
│     └── model responds: '{"category": "account"}'
│     └── json.loads(...) → {"category": "account"}
│
├── await env.step({"category": "account"})
│     └── client sends: {"type":"step","data":{"category":"account"}}
│     └── server: deserialize_action({"category":"account"}, SupportTriageAction)
│           └── Pydantic validates → SupportTriageAction(category="account", priority=None, ...)
│     └── server: env.step(action) called
│           └── self._state.step_count += 1
│           └── _grade(action, self._ticket) called
│                 └── _category_score("account", "account", aliases) → 1.0
│                 └── breakdown = {"category": 1.0}
│                 └── reward = 1.0 * 1.0 = 1.0
│           └── returns SupportTriageObservation(reward=1.0, done=True, feedback="...")
│     └── server serialises → JSON sent over WebSocket
│     └── client → StepResult(reward=1.0, done=True)
│
├── log_step(step=1, action='{"category": "account"}', reward=1.0, done=True, error=None)
│     └── prints: [STEP] step=1 action={"category": "account"} reward=1.00 done=true error=null
│
└── log_end(success=True, steps=1, score=1.0, rewards=[1.0])
      └── prints: [END] success=true steps=1 score=1.000 rewards=1.00
```

---

## 15. Key Lessons

If you build your own OpenEnv environment, these are the things that will trip you up:

1. **`openenv validate` does string matching** — it looks for the literal `"main()"` in
   `server/app.py`. `main(port=8000)` does NOT satisfy it.

2. **`reset()` can accept any kwargs** — declare them as parameters and they are
   automatically forwarded from the HTTP body or WebSocket data via `inspect.signature`.

3. **`extra="forbid"` on Action** — every field an agent might send must be declared,
   even if optional. Undeclared fields cause HTTP 422 errors.

4. **HTTP `/step` is stateless** — each call creates a fresh environment. Make step()
   handle the no-reset case, or document that users must use WebSocket.

5. **HuggingFace Docker Spaces require port 7860** — any other port and the Space stays
   stuck at `APP_STARTING`. No error message will tell you this directly.

6. **`uv.lock` must exist** — run `uv lock` after any change to `pyproject.toml`.

7. **Never put tokens in repo files** — HF's push API scans all uploaded files and
   blocks commits that contain valid tokens.

8. **Rewards must be in [0.0, 1.0]** — always clamp: `max(0.0, min(1.0, reward))`.

9. **Use `sys.executable` not `"python"` in subprocesses** — ensures the subprocess
   uses the same virtualenv as the parent script.

10. **Add a `/` route** — `create_app()` does not register one. Without it, visiting
    the Space URL in a browser shows `{"detail":"Not Found"}`.
