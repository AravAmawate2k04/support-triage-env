"""
FastAPI application for the Support Triage Environment.

Endpoints:
    POST /reset    — start a new episode (accepts JSON body with optional `task` field)
    POST /step     — execute one action
    GET  /state    — current environment state
    GET  /schema   — action/observation/state schemas
    GET  /health   — liveness probe
    WS   /ws       — persistent WebSocket session (used by EnvClient)

Usage (development):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

Usage (production):
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:
    raise ImportError(
        "openenv-core is required. Install it with: pip install openenv-core"
    ) from exc

try:
    from ..models import SupportTriageAction, SupportTriageObservation
    from .support_triage_environment import SupportTriageEnvironment
except ImportError:
    from models import SupportTriageAction, SupportTriageObservation
    from server.support_triage_environment import SupportTriageEnvironment


app = create_app(
    SupportTriageEnvironment,
    SupportTriageAction,
    SupportTriageObservation,
    env_name="support_triage",
    max_concurrent_envs=4,
)

from fastapi.responses import HTMLResponse  # noqa: E402


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root() -> HTMLResponse:
    """Landing page with links to interactive docs and key endpoints."""
    html = """<!DOCTYPE html>
<html>
<head>
  <title>Support Triage Environment</title>
  <style>
    body { font-family: sans-serif; max-width: 700px; margin: 60px auto; padding: 0 20px; color: #222; }
    h1 { color: #1a56db; }
    code { background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
    pre { background: #f3f4f6; padding: 14px; border-radius: 8px; overflow-x: auto; }
    a { color: #1a56db; }
    .badge { display:inline-block; background:#1a56db; color:#fff; padding:2px 10px; border-radius:12px; font-size:0.8em; }
    table { border-collapse: collapse; width: 100%; }
    td, th { border: 1px solid #e5e7eb; padding: 8px 12px; text-align: left; }
    th { background: #f9fafb; }
  </style>
</head>
<body>
  <h1>🎫 Support Triage Environment <span class="badge">OpenEnv</span></h1>
  <p>An AI agent triages customer support tickets across three difficulty levels.</p>

  <h2>Tasks</h2>
  <table>
    <tr><th>Task</th><th>Difficulty</th><th>Agent must…</th></tr>
    <tr><td><code>categorize</code></td><td>Easy</td><td>Classify the ticket category</td></tr>
    <tr><td><code>triage</code></td><td>Medium</td><td>Category + priority + team routing</td></tr>
    <tr><td><code>respond</code></td><td>Hard</td><td>Full triage + write a customer reply</td></tr>
  </table>

  <h2>Quick Start</h2>
  <pre>curl -X POST {base_url}/reset \\
  -H "Content-Type: application/json" \\
  -d '{{"task": "triage", "seed": 42}}'</pre>

  <h2>Endpoints</h2>
  <ul>
    <li><a href="/docs">📖 Interactive API Docs (Swagger)</a></li>
    <li><a href="/health"><code>GET /health</code></a> — liveness probe</li>
    <li><a href="/schema"><code>GET /schema</code></a> — action / observation schemas</li>
    <li><a href="/metadata"><code>GET /metadata</code></a> — environment metadata</li>
    <li><code>POST /reset</code> — start episode (body: <code>{{"task":"categorize|triage|respond","seed":42}}</code>)</li>
    <li><code>POST /step</code> — submit action</li>
  </ul>

  <p><a href="https://huggingface.co/spaces/AravAmawate/support-triage-env">View on HuggingFace Spaces</a></p>
</body>
</html>"""
    return HTMLResponse(content=html)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """
    Entry point for direct execution.

    Examples::

        # Run via uv:
        uv run --project . server

        # Run directly:
        python -m server.app
        python server/app.py --port 8001
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
