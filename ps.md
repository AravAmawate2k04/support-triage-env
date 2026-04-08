Round 1 — Problem Statement
THE TASK
Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard  step() / reset() / state()  API.
KEY REQUIREMENTS AT A GLANCE
Must simulate a real-world task (not games or toys)
Implement full OpenEnv spec: typed models, step()/reset()/state(), openenv.yaml
Minimum 3 tasks with agent graders (easy → medium → hard, scores/reward 0.0–1.0)
Meaningful reward function with partial progress signals
Baseline inference script with reproducible scores
Deploy to Hugging Face Spaces + working Dockerfile
README with environment description, action/observation spaces, setup instructions


Real-world task simulation
The environment must simulate a task humans actually do. Not games, not toys. Examples: email triage, code review, data cleaning, scheduling, customer support, content moderation.
OpenEnv spec compliance
Implement the full OpenEnv interface: typed Observation, Action, and Reward Pydantic models. step(action) → returns observation, reward, done, info. reset() → returns initial observation. state() → returns current state. openenv.yaml with metadata. Tested via openenv validate.
Minimum 3 tasks with agent graders
Each task defines a concrete objective an agent must accomplish, with a programmatic grader that scores performance (0.0–1.0). Tasks should range: easy → medium → hard. Graders must have clear, deterministic success/failure criteria.
Meaningful reward function
Provides signal over the full trajectory (not just binary end-of-episode). Rewards partial progress toward task completion. Penalizes clearly undesirable behavior (e.g. infinite loops, destructive actions).
Baseline inference script
Uses the OpenAI API client to run a model against the environment. Reads API credentials from environment variables (OPENAI_API_KEY). Produces a reproducible baseline score on all 3 tasks.


Deploys to a Hugging Face Space
Environment must run as a containerized HF Space tagged with openenv.
Containerized execution
Must include a working Dockerfile. The environment should start cleanly with docker build + docker run.
Documentation
README must include: environment description and motivation, action and observation space definitions, task descriptions with expected difficulty, setup and usage instructions, baseline scores.


PARAMETER
WEIGHT
DESCRIPTION
Real-world utility
30%
Does the environment model a genuine task? Would someone actually use this to train or evaluate agents?
Task & grader quality
25%
Are tasks well-defined with clear objectives? Do graders accurately and fairly measure success? Meaningful difficulty progression?
Environment design
20%
Clean state management, sensible action/observation spaces, good reward shaping, proper episode boundaries.
Code quality & spec compliance
15%
Follows OpenEnv spec, clean project structure, typed models, documented, tested, Dockerfile works.
Creativity & novelty
10%
Novel problem domain, interesting mechanics, clever reward design, original approach.
SCORING BREAKDOWN
Real-world utility (30%)
•  0–5: Toy/artificial problem with no practical application
•  6–15: Valid domain but shallow modeling of the real task
•  16–25: Good domain modeling, would be useful for agent evaluation
•  26–30: Excellent — fills a real gap, immediate value for the RL/agent community
Task & grader quality (25%)
•  3+ tasks with difficulty range?
•  Graders produce scores between 0.0–1.0?
•  Graders deterministic and reproducible?
•  Hard task genuinely challenges frontier models?
Environment design (20%)
•  reset() produces clean state?
•  Action/observation types well-designed and documented?
•  Reward function provides useful varying signal (not just sparse)?
•  Episode boundaries sensible?
Code quality & spec compliance (15%)
•  openenv validate passes?
•  docker build && docker run works?
•  HF Space deploys and responds?
•  Baseline script runs and reproduces scores?
Creativity & novelty (10%)
•  Domain we haven’t seen in OpenEnv before?
•  Reward design has interesting properties?
•  Clever mechanics that make the environment engaging?
Evaluation Criteria



pre-eval checklist

HF Space deploys
Automated ping to the Space URL — must return 200 and respond to reset()
OpenEnv spec compliance
Validate openenv.yaml, typed models, step()/reset()/state() endpoints
Dockerfile builds
Automated docker build on the submitted repo
Baseline reproduces
Run the submitted inference script — must complete without error and produce scores
3+ tasks with graders
Enumerate tasks, run each grader, verify scores/reward in 0.0–1.0 range
Mandatory Additional Instructions
Before submitting, ensure the following variables are defined in your environment configuration:
API_BASE_URL   The API endpoint for the LLM.
MODEL_NAME     The model identifier to use for inference.
HF_TOKEN       Your Hugging Face / API key.
The inference script must be named `inference.py` and placed in the root directory of the project
Participants must use OpenAI Client for all LLM calls using above variables
Participants must emit structured stdout logs strictly following the [START], [STEP], and [END] format defined in the sample inference.py provided below. Any deviation in field names, ordering, or formatting will result in incorrect evaluation scoring. Refer to the Sample Inference Script for the complete format specification and examples.
Infra Restrictions
Runtime of inference script should be less than 20min 
Make sure your env and inference can run on a machine with vcpu=2, memory=8gb
Validator
Run the pre-submission validation script before submitting


Functional Requirements
1. Real-World Task Simulation
The environment must represent tasks that humans actually perform in real settings—no games or toy problems.
Examples include email triage, code review, data cleaning, scheduling, customer support, and content moderation.

2. OpenEnv Specification Compliance
The environment must fully implement the OpenEnv interface, including:
Typed Observation, Action, and Reward models using Pydantic
step(action) → returns (observation, reward, done, info)
reset() → returns the initial observation
state() → returns the current state
An openenv.yaml file containing metadata
The implementation must successfully pass validation via openenv validate.

3. Minimum of Three Tasks with Agent Graders
Provide at least three tasks, each with a clearly defined objective
Tasks should span increasing difficulty: easy → medium → hard
Each task must include a programmatic grader that assigns a score between 0.0 and 1.0
Grading criteria must be clear, deterministic, and reproducible

4. Meaningful Reward Function
The reward function must provide feedback throughout the task trajectory, not just at completion
It should reward incremental progress toward the objective
It must penalize undesirable behaviors such as infinite loops or destructive actions

5. Baseline Inference Script
Include an inference script that uses the OpenAI API client to evaluate a model within the environment
API credentials must be read from environment variables (HF_TOKEN)
The script should produce a reproducible baseline score across all tasks

Non-Functional Requirements
1. Deployment on Hugging Face Spaces
The environment must be deployable as a containerized Hugging Face Space
It should be tagged with openenv

2. Containerized Execution
Provide a working Dockerfile
The environment must build and run successfully using:
docker build
docker run

3. Documentation
The README must include:
Environment overview and motivation
Definitions of action and observation spaces
Task descriptions with expected difficulty levels
Setup and usage instructions
Baseline performance scores


Additional Guideline: Meta OpenEnv Hackathon: Guidelines