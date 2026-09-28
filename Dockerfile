FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install uv

RUN uv sync --frozen

RUN uv run python -m spacy download en_core_web_lg

EXPOSE 8000

CMD ["uv", "run", "fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]