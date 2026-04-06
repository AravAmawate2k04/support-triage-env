# Support Triage Environment — Docker image
# Build:  docker build -t support-triage-env .
# Run:    docker run -p 8000:8000 support-triage-env

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir \
    "openenv-core>=0.2.2" \
    "fastapi>=0.100.0" \
    "uvicorn>=0.20.0" \
    "pydantic>=2.0.0" \
    "openai>=1.0.0"

# Install the environment package in editable mode so imports resolve
RUN pip install --no-cache-dir -e .

EXPOSE 7860

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the server — HuggingFace Spaces require port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
