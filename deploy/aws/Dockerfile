# LevelEdge API on AWS Lambda (arm64), served by the unchanged FastAPI app.
# The AWS Lambda Web Adapter turns Lambda invocations into plain HTTP requests to uvicorn,
# so the same app runs locally, in Docker and on Lambda. The frontend lives on Vercel.
#
#   docker build --platform linux/arm64 --provenance=false -f deploy/aws/Dockerfile -t leveledge-api .

# ---- research summaries only (the results folder is 39 MB; the API reads <1 MB of it) ----
FROM public.ecr.aws/docker/library/python:3.13-slim AS results
COPY benchmarks/results /results
RUN find /results -type f ! -name '*.json' ! -name 'variants.csv' -delete && find /results -type d -empty -delete

# ---- API ------------------------------------------------------------------------------
FROM public.ecr.aws/docker/library/python:3.13-slim
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 LEVELEDGE_DISABLE_LOGGING=1 \
    PORT=8080 AWS_LWA_READINESS_CHECK_PATH=/api/health \
    HOME=/tmp XDG_CACHE_HOME=/tmp/.cache
WORKDIR /app

COPY deploy/aws/requirements-lambda.txt .
RUN pip install -r requirements-lambda.txt

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-deps . && rm -rf src

COPY app/ app/
COPY --from=results /results benchmarks/results

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
