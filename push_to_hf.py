"""
Push the environment to a HuggingFace Space.

Usage:
    python push_to_hf.py --repo YOUR_HF_USERNAME/support-triage-env

The Space will be created if it doesn't exist.
Requires: huggingface_hub, HF_TOKEN env var or prior `hf auth login`.
"""

import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo


IGNORE_PATTERNS = [
    "venv/",
    ".git/",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "*.egg-info/",
    ".claude/",
    "push_to_hf.py",
    "token.txt",
    "*.token",
    "secrets.*",
    ".env",
]


def push(repo_id: str, token: str | None = None) -> str:
    api = HfApi(token=token)

    # Create the Space (Docker SDK) if it doesn't exist
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        token=token,
    )
    print(f"Space ready: https://huggingface.co/spaces/{repo_id}")

    # Upload all project files
    api.upload_folder(
        folder_path=os.path.dirname(os.path.abspath(__file__)),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Update support-triage environment",
    )
    url = f"https://{repo_id.replace('/', '-')}.hf.space"
    print(f"Deployed: {url}")
    return url


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace repo ID, e.g. myuser/support-triage-env",
    )
    args = parser.parse_args()
    token = os.getenv("HF_TOKEN")
    push(args.repo, token=token)
