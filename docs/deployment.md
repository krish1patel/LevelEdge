# Deployment

The public dashboard runs at **<https://leveledge.vercel.app>**. The React frontend is hosted on Vercel, and the FastAPI backend runs on AWS Lambda. This guide shows you how to deploy updates, how the setup stays inside free tiers, and how to pause it.

```text
Browser ─▶ Vercel (static React build) ─▶ /api/* rewrite ─▶ Lambda Function URL ─▶ FastAPI (uvicorn)
```

`web/vercel.json` rewrites `/api/*` to the Lambda Function URL, so the browser talks to one origin and needs no CORS setup.

## Deploying an update

### Prerequisites

- The AWS CLI v2, signed in to the AWS account (`aws login`, region `us-east-1`)
- Docker
- The Vercel CLI, signed in to the account that owns the `leveledge` project

### Steps

1. Sign in to AWS. The session lasts several hours.

   ```bash
   aws login
   ```

2. Build and deploy the API:

   ```bash
   bash deploy/aws/deploy.sh
   ```

   The script builds the arm64 image, pushes it to ECR, then creates or updates the `leveledge-api` function and its public URL. It smoke-tests `/api/health` and rewrites `web/vercel.json` with the function's URL.

3. Publish the frontend:

   ```bash
   cd web && vercel deploy --prod
   ```

4. Verify the deployment:

   ```bash
   curl -s https://leveledge.vercel.app/api/health
   ```

   The response is JSON that includes `"ok":true` and the supported intervals.

Deploys run from your machine and aren't tied to merges.

## Understanding the AWS setup

| Resource | Setting |
|---|---|
| Lambda function `leveledge-api` | arm64 container image, 1,536 MB memory, 120 s timeout, public Function URL |
| Image | `deploy/aws/Dockerfile`. It installs only the packages the API imports (`deploy/aws/requirements-lambda.txt`) and the research summaries, and uses the AWS Lambda Web Adapter so the unchanged app runs under uvicorn. |
| Secrets | `deploy.sh` copies `SUPABASE_*`, `ALPACA_*` and `POLYGON_API_KEY` from `.env` into the function's encrypted environment. It never prints them. |
| IAM role | `leveledge-api-role` with only `AWSLambdaBasicExecutionRole`. Set `ROLE_ARN=...` to use an existing role, such as an AWS Academy `LabRole`. |

## Controlling cost

The deployment stays inside AWS's always-free Lambda tier at this traffic.

| Guard | Effect |
|---|---|
| Lambda free tier | 1M requests and 400,000 GB-seconds a month are free. At 1.5 GB, that's about 266,000 seconds of compute. |
| Reserved concurrency of 5 | At most 5 copies run at once, so a flood of requests can't run up a bill |
| ECR lifecycle policy | Keeps the 2 newest images (about 650 MB, a few cents a month) |
| Log retention | CloudWatch keeps 7 days of logs |
| Budget `leveledge-10usd` | Emails the account owner when actual monthly spend passes $10, or the month's forecast does. It covers the whole account. |

AWS updates billing data a few times a day, so a budget email can arrive hours after the threshold.

### Setting up the market scan

```bash
bash deploy/aws/scan.sh            # create or update the scan pipeline
bash deploy/aws/scan.sh --run-now  # ... and start a scan now
```

See [Market scan](market-scan.md). `deploy.sh` keeps the scan wired up and updates the worker's image on every deploy.

### Pausing the API

To stop all API spending at once:

```bash
bash deploy/aws/pause.sh
```

The function then refuses every request (HTTP 429). The dashboard still loads, but its data panels show errors. To resume:

```bash
bash deploy/aws/pause.sh resume
```

Pausing leaves the stored image, which costs a few cents a month.

## HuggingFace Space (older deployment)

Every push to `main` syncs the repository to a HuggingFace Space (`.github/workflows/sync_to_hf.yml`). The root `Dockerfile` builds the frontend with Node, then serves the API and the static frontend with uvicorn on port 7860. HuggingFace reads the Space's settings from YAML at the top of `README.md`; `scripts/sync_space.sh` adds that header from `deploy/huggingface/space_header.yml` at sync time, so the GitHub README stays clean.

Space secrets:

- `SUPABASE_URL`, `SUPABASE_KEY`: enable the Track record tab.
- `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`: show the live paper account. (`ALPACA_API_KEY` and `ALPACA_SECRET_KEY` also work.)
