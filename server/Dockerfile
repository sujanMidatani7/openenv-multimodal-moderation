
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SENTENCE_TRANSFORMERS_HOME=/opt/models

WORKDIR /workspace

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/opt/models')"

COPY app ./app
COPY server.py ./server.py
COPY inference.py ./inference.py
COPY openenv.yaml ./openenv.yaml
COPY README.md ./README.md

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
