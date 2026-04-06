"""
Quick interactive test — runs one episode per task using the live Space.
Shows the ticket, asks for your answer, then shows the score.

Usage:
    python test_env.py
    python test_env.py --url http://localhost:8000   # local server
"""

import asyncio
import argparse
from openenv.core.generic_client import GenericEnvClient

SPACE_URL = "https://aravamawate-support-triage-env.hf.space"

TASKS = {
    "1": ("categorize", "Enter the category (billing/technical/account/feedback/shipping): "),
    "2": ("triage",     "Enter category,priority,team (e.g. billing,high,billing): "),
    "3": ("respond",    "Enter category,priority,team,response (e.g. billing,high,billing,Dear ...): "),
}

async def run(base_url: str, task: str) -> None:
    async with GenericEnvClient(base_url=base_url) as env:
        result = await env.reset(task=task, seed=42)
        obs = result.observation

        print(f"\n{'='*60}")
        print(f"TASK: {task.upper()}")
        print(f"Ticket:   {obs['ticket_id']} — {obs['subject']}")
        print(f"Customer: {obs['customer_name']} ({obs['customer_tier']})")
        print(f"\n{obs['body']}\n")
        print(f"Instructions: {obs['task_description']}\n")

        if task == "categorize":
            cat = input("Your category: ").strip()
            action = {"category": cat}

        elif task == "triage":
            raw = input("category,priority,team: ").strip().split(",")
            action = {
                "category": raw[0].strip(),
                "priority": raw[1].strip() if len(raw) > 1 else "low",
                "assigned_team": raw[2].strip() if len(raw) > 2 else "support",
            }

        else:  # respond
            cat    = input("category: ").strip()
            pri    = input("priority: ").strip()
            team   = input("assigned_team: ").strip()
            print("response_text (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
            action = {
                "category": cat, "priority": pri,
                "assigned_team": team,
                "response_text": "\n".join(lines).strip(),
            }

        result = await env.step(action)
        obs = result.observation

        print(f"\n{'='*60}")
        print(f"REWARD:  {result.reward:.3f} / 1.000")
        print(f"\nFEEDBACK:\n{obs.get('feedback','')}")
        print(f"\nSCORE BREAKDOWN: {obs.get('score_breakdown',{})}")
        print(f"{'='*60}\n")


async def main(base_url: str) -> None:
    print("\nSupport Triage Environment — Interactive Test")
    print("1) categorize  (easy)")
    print("2) triage      (medium)")
    print("3) respond     (hard)")
    choice = input("\nPick a task [1/2/3]: ").strip()
    task, _ = TASKS.get(choice, ("categorize", ""))
    await run(base_url, task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=SPACE_URL)
    args = parser.parse_args()
    asyncio.run(main(args.url))
