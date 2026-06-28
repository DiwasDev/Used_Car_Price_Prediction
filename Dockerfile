FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging and requirements files
COPY pyproject.toml setup.py requirements.txt README.md ./
COPY src/ ./src/

# Install dependencies (including the editable package installation)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 5000

# Environment variables
ENV MONGODB_URL=""
ENV AZURE_STORAGE_CONNECTION_STRING=""

# Run application
CMD ["python", "app.py"]
