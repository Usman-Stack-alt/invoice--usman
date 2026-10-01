FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/tmp OMP_THREAD_LIMIT=1

# Tesseract 5 (bookworm ships 5.3.x) for OCR; LibreOffice Writer only for DOCX -> PDF.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0 curl \
        libreoffice-writer fonts-dejavu-core fonts-liberation \
    && rm -rf /var/lib/apt/lists/* \
    && tesseract --version

WORKDIR /srv
COPY requirements.txt requirements-dev.txt ./
ARG INSTALL_DEV=0
RUN if [ "$INSTALL_DEV" = "1" ]; then pip install -r requirements-dev.txt; else pip install -r requirements.txt; fi

COPY app ./app
COPY tests ./tests
COPY scripts ./scripts

RUN useradd -r -u 10001 appuser && mkdir -p /data/llm && chown appuser /data/llm
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8000/healthz || exit 1
# Each worker is one process; OCR runs in threads, bounded by CONCURRENCY per worker.
CMD ["gunicorn", "app.main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8000", "--timeout", "120"]
