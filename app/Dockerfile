FROM python:3.10-slim

# libgomp1: OpenMP runtime required by lightgbm at import time
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py comment_classifier_pipeline.joblib ./

ENV GRADIO_SERVER_NAME=0.0.0.0
# Render (and most PaaS hosts) inject a dynamic $PORT at runtime; fall back to
# 7860 for local `docker run` / platforms that don't set it (e.g. plain HF Docker Spaces).
EXPOSE 7860

CMD ["sh", "-c", "GRADIO_SERVER_PORT=${PORT:-7860} python app.py"]
