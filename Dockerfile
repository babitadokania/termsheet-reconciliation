# Reproducible runtime: Python 3.11 + Java 17 for PySpark.
#   docker build -t tsrecon .
#   docker run --rm -e GEMINI_API_KEY=... -v "$PWD/output:/app/output" tsrecon
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
 && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home recon && chown -R recon /app
USER recon

ENTRYPOINT ["python", "run_pipeline.py"]
