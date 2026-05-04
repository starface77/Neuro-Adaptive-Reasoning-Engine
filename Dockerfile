FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy VARE code
COPY nare/ ./nare/
COPY benchmarks/ ./benchmarks/
COPY generate_predictions.py .
COPY setup.py .
COPY pyproject.toml .

# Install VARE
RUN pip install -e .

# Create directories
RUN mkdir -p /app/memory_swe_submission /app/logs_submission

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GEMINI_API_KEY=""

# Default command
CMD ["python", "generate_predictions.py", \
     "--tasks", "benchmarks/swebench_real.json", \
     "--persist-dir", "memory_swe_submission", \
     "--output", "predictions.json", \
     "--log-dir", "logs_submission"]
