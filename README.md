# OpenEnv Multimodal Moderation Environment

This project implements a production-ready OpenEnv environment for multimodal content moderation. The agent moves through a fixed workflow:

1. `analyze`
2. `retrieve_policy`
3. `decide`
4. `review`
5. `finalize`

The environment follows the installed Meta PyTorch OpenEnv interfaces:

- `reset(...) -> Observation`
- `step(action) -> Observation`
- `state -> State`

It subclasses `openenv.core.env_server.interfaces.Environment` and exposes HTTP endpoints through `create_fastapi_app(...)`.

## Project Layout

```text
app/
  __init__.py
  env.py
  logic.py
  models.py
  rag/
    __init__.py
    policies.py
    retriever.py
server.py
inference.py
openenv.yaml
Dockerfile
requirements.txt
README.md
```

## Environment Design

Each episode contains a single moderation case with:

- user text
- image metadata
- expected moderation action
- reviewer recommendation

State persists in `state_data` across every step and stores:

- selected case
- retrieved policy chunks
- action history
- reviewer note
- cumulative rewards

### Moderation actions

- `allow`
- `flag`
- `remove`
- `escalate`

### Rule overrides

- text containing `kill` or `murder` forces a `remove` expectation
- text containing `nude` forces a `flag` expectation

### Dense rewards

- analysis step: `+0.2`
- retrieval grounding step: `+0.2`
- correct decision: `+1.0`
- reviewer agreement: `+0.2`
- unsafe allow on risky content: `-0.6`

## RAG

The retriever uses:

- `sentence-transformers/all-MiniLM-L6-v2`
- `FAISS IndexFlatIP`
- top-`k=3` retrieval over curated moderation policy chunks

The Docker build pre-downloads the encoder so there is no large model download at runtime.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the OpenEnv server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Additional helper endpoints:

- `/cases` lists built-in case ids
- `/state_full` returns the full typed state object
- `/episode_summary` returns final scores and reviewer output

## Inference Script

`inference.py` uses the OpenAI Python client against an OpenAI-compatible endpoint.

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional:

- `ENV_BASE_URL` default: `http://127.0.0.1:8000`

Run:

```bash
python inference.py
```

The script:

- resets the environment per case
- runs the full episode loop
- asks the model for a JSON action at each step
- currently th code uses llama-3.1-8b-instant model
- prints per-case and aggregate scores

## Docker

Build:

```bash
docker build -t openenv-moderation .
```

Run:

```bash
docker run --rm -p 8000:8000 openenv-moderation
```

## Resource Profile

- designed for `<= 2 vCPU`
- designed for `<= 8 GB RAM`
- small encoder only
- deterministic case selection when `seed` is provided
