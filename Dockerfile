FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE_DIR=/models

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Create the runtime user first so model files are written with the right owner.
# (A `chown -R` after the download would duplicate the ~1.2 GB model layer.)
RUN useradd --create-home appuser && mkdir -p /models && chown appuser /app /models
USER appuser

# Bake the embedding + reranker models into the image so containers start offline.
# Build with --build-arg PRELOAD_MODELS=false for a small image that downloads at first start.
ARG PRELOAD_MODELS=true
ARG HF_ENDPOINT=https://huggingface.co
ARG MODEL_SOURCE=huggingface
ENV HF_ENDPOINT=${HF_ENDPOINT} \
    MODEL_SOURCE=${MODEL_SOURCE}
COPY --chown=appuser app/config.py app/models.py app/__init__.py ./app/
RUN if [ "$PRELOAD_MODELS" = "true" ]; then \
      python -c "from app.config import get_settings; from app.models import build_embedder, build_reranker; s=get_settings(); build_embedder(s); build_reranker(s)"; \
    fi

COPY --chown=appuser app ./app
COPY --chown=appuser data ./data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
