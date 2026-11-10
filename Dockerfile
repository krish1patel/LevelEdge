# ---- 1. build the web frontend ----------------------------------------------
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---- 2. Python API that also serves the built frontend ----------------------
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 LEVELEDGE_DISABLE_LOGGING=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-deps .

COPY app/ app/
COPY benchmarks/results/summary.json benchmarks/results/summary.json
COPY --from=web /web/dist web/dist

# HuggingFace Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user && chown -R user /app
USER user

EXPOSE 7860
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
