FROM python:3.11-slim

# Install OpenJDK runtime environment for PySpark execution
RUN mkdir -p /usr/share/man/man1 && \
    apt-get update && \
    apt-get install -y default-jdk procps curl && \
    apt-get clean;

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary architecture directories
RUN mkdir -p /app/data/shared /app/models /app/spark_checkpoints

ENV PYTHONUNBUFFERED=1