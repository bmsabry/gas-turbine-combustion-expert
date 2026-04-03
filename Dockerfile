FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and Node.js
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY embeddings/ ./embeddings/
COPY knowledge_graph/ ./knowledge_graph/
COPY chunks/ ./chunks/
COPY papers/metadata/ ./papers/metadata/
COPY admin_settings.json ./admin_settings.json

# Build frontend - copy entire frontend folder first
COPY frontend/ ./frontend/
WORKDIR /app/frontend
RUN npm install && npm run build

# Move built frontend to static directory
WORKDIR /app
RUN mkdir -p /app/static && cp -r /app/frontend/dist/* /app/static/ && rm -rf /app/frontend

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.backend:app", "--host", "0.0.0.0", "--port", "8000"]
