# -----------------------------
# Base image
# -----------------------------
FROM python:3.11-slim

# -----------------------------
# Prevent prompts
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -----------------------------
# Set work directory
# -----------------------------
WORKDIR /code

# -----------------------------
# Install system packages required for transformers + torch
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy files
# -----------------------------
COPY . /code/

# -----------------------------
# Install Python dependencies
# -----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Expose port (HuggingFace uses 7860)
# -----------------------------
EXPOSE 7860

# -----------------------------
# Start the server
# -----------------------------
CMD ["python", "app.py"]
