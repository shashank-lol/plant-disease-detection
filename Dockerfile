# Use Python 3.10 slim as base image
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files and buffering output
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy all files to the working directory
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
