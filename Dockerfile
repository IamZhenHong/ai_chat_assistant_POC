# Use Python 3.12 as the base image
FROM python:3.12

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first (for caching layers)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose FastAPI's port (default: 8000)
EXPOSE 8000

# Default command to run FastAPI
CMD uvicorn fastapi_app.main:app --host 0.0.0.0 --port 8000 & celery -A fastapi_app.worker worker --loglevel=info
