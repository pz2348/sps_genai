FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY . .

# Install dependencies
RUN uv sync --locked --no-dev --no-cache

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uv", "run", "--no-sync", "fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
