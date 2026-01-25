# Hugging Face Spaces - Docker SDK
# Schema-Agnostic Database Chatbot with RAG

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create a non-root user for security
RUN useradd -m -u 1000 appuser

# Create cache directories with proper permissions
RUN mkdir -p /app/.cache/sentence_transformers /app/.cache/transformers /app/faiss_index \
    && chown -R appuser:appuser /app

# Copy requirements first for better caching
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose Streamlit port (HF Spaces expects 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--server.enableCORS=true", \
    "--server.enableXsrfProtection=false", \
    "--browser.gatherUsageStats=false", \
    "--server.fileWatcherType=none"]
