# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install uv==0.6.14

COPY pyproject.toml .
RUN uv sync --no-dev

COPY app.py .

RUN useradd -m -u 1001 appuser
USER appuser

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
