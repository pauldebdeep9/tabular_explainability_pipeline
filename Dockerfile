# Use Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files
COPY . /app

# Install OS dependencies for sklearn, pandas, matplotlib, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Optional: run the pipeline by default
CMD ["python", "run_pipeline.py", "--feature_selection_method", "lasso"]
