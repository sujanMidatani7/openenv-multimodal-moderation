---
title: OpenEnv Multimodal Moderation Environment
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: apache-2.0
app_port: 8000
---

# OpenEnv Multimodal Moderation Environment

Production-ready OpenEnv environment for multimodal content moderation with a fixed episode flow:

1. `analyze`
2. `retrieve_policy`
3. `decide`
4. `review`
5. `finalize`

The environment follows the official Meta OpenEnv API:

- `reset(...) -> Observation`
- `step(action) -> Observation`
- `state -> State`

It subclasses `openenv.core.env_server.interfaces.Environment` and serves HTTP endpoints via `create_fastapi_app(...)`.

## Project Layout

```text
server/
  __init__.py
  app.py
  env.py
  logic.py
  models.py
  server_routes.py
  requirements.txt
  Dockerfile
  rag/
    __init__.py
    policies.py
    retriever.py
client.py
models.py
inference.py
openenv.yaml
pyproject.toml
requirements.txt
README.md
```

## Environment Design

Each episode uses one moderation case containing:

- text content
- image metadata
- expected moderation action
- reviewer recommendation

Persistent `state_data` stores:

- selected case
- retrieved policy chunks
- action history
- reviewer note
- reward breakdown
- final action

### Actions

- `allow`
- `flag`
- `remove`
- `escalate`

### Rule Overrides

- content containing `kill` or `murder` forces a `remove` expectation
- content containing `nude` forces a `flag` expectation

### Dense Rewards

- analysis step: `+0.2`
- retrieval grounding step: `+0.2`
- correct decision: `+1.0`
- reviewer agreement: `+0.2`
- unsafe allow on risky content: `-0.6`

## Policy Retrieval

The retriever loads curated moderation policies from `server/rag/policies.json` and ranks them with a lightweight keyword-overlap scorer. This keeps startup deterministic, avoids runtime model downloads, and eliminates noisy encoder logs during evaluation.

## Install

This repo is set up for `uv` and the official OpenEnv package source from the Meta repo.

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv sync
```

If you hit a stuck Windows cache rename error, remove the local cache and retry:

```powershell
Remove-Item -Recurse -Force .\.uv-cache -ErrorAction SilentlyContinue
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv sync
```

## Run Locally

Preferred:

```powershell
uv run server
```

Direct Python entrypoint:

```powershell
python server/app.py
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

## HTTP Endpoints

Core OpenEnv endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /docs`

Helper endpoints:

- `GET /cases` lists built-in case ids
- `GET /state_full` returns full typed state
- `GET /episode_summary` returns final reward and reviewer output

## Inference Script

`inference.py` uses the OpenAI Python client against an OpenAI-compatible API.

Environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional:

- `ENV_BASE_URL` default: `http://127.0.0.1:8000`

Run:

```powershell
python inference.py
```

Notes:

- the script runs one task per invocation and emits strict `[START]`, `[STEP]`, and `[END]` lines
- task selection and grading are aligned with the built-in moderation cases
- the current default `MODEL_NAME` in code is only used when your environment configuration provides one

## Docker

Docker uses `uv` for package installation.

Build:

```powershell
docker build -f server/Dockerfile -t openenv-moderation .
```

Run:

```powershell
docker run --rm -p 8000:8000 openenv-moderation
```

to run the deployed version:
```powershell
docker run -it -p 7860:7860 --platform=linux/amd64 \
	-e API_BASE_URL="YOUR_VALUE_HERE" \
	-e MODEL_NAME="YOUR_VALUE_HERE" \
	-e HF_TOKEN="YOUR_VALUE_HERE" \
    -e ENV_BASE_URL="YOUR_VALUE_HERE" \
	registry.hf.space/sujanmidatani-openenv-multimodal-moderation:latest 
```

## Resource Profile

- designed for `<= 2 vCPU`
- designed for `<= 8 GB RAM`
- lightweight JSON-backed retrieval only
- deterministic case selection when `seed` is provided
