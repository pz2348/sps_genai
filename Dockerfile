FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY . .

# Install dependencies
RUN uv sync --frozen --no-dev

# Download spaCy language model
RUN uv run python -m spacy download en_core_web_lg

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uv", "run", "fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]