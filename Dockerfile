FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-ind \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONPATH="/app/src"

COPY pyproject.toml poetry.lock* ./
COPY src ./src
COPY dataset ./dataset
COPY script ./script
COPY README.md ./
COPY deployment_gcp.md ./
COPY flow.png ./

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-interaction --no-ansi

EXPOSE 8080

CMD ["uvicorn", "karierai.server:app", "--host", "0.0.0.0", "--port", "8080"]
