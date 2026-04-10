
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY server/requirements.txt ./requirements.txt
RUN uv pip install --system --no-cache -r requirements.txt

COPY server ./server
COPY client.py ./client.py
COPY models.py ./models.py
COPY inference.py ./inference.py
COPY openenv.yaml ./openenv.yaml
COPY README.md ./README.md
# COPY .env ./.env
COPY __init__.py ./__init__.py

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
