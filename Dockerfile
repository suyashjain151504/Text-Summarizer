# Use official slim Python 3.13 image (Debian Bookworm based)
FROM python:3.13-slim-bookworm

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (awscli + common build tools if needed)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        awscli \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first (better layer caching)
COPY requirements.txt .

# Install Python dependencies
# The old "upgrade → uninstall → reinstall" pattern for transformers/accelerate
# is no longer needed in 2025/2026. Modern versions resolve cleanly together.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install --upgrade transformers accelerate

# Copy the rest of the application code
COPY . .

# Default command
CMD ["python3", "app.py"]