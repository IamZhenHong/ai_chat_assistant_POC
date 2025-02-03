# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies needed for psycopg2 and general compilation
RUN apt-get update && apt-get install -y \
    libpq-dev gcc python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy and install dependencies separately to leverage Docker cache
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose FastAPI port
EXPOSE 8000

# Define environment variable (corrected format)
ENV PYTHONUNBUFFERED=1

# Run FastAPI with Uvicorn
CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
