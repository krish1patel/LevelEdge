# deploy/aws

Runs the LevelEdge API on AWS Lambda and points the Vercel frontend at it.

## Contents

| File | Purpose |
|---|---|
| `Dockerfile` | The arm64 Lambda image: the unchanged FastAPI app under uvicorn, behind the AWS Lambda Web Adapter, plus the research summaries |
| `requirements-lambda.txt` | Only the packages the API imports. The root `requirements.txt` also carries Streamlit and other tools the API doesn't need. |
| `deploy.sh` | Builds and pushes the image, creates or updates the function, its role and its public URL, applies the cost guards, smoke-tests, and writes `web/vercel.json` |
| `scan.sh` | Creates or updates the daily market scan: DynamoDB table, SQS queue and dead-letter queue, worker function, queue mapping (at most 5 in parallel), and the 5:45 PM ET schedule. `--run-now` starts a scan. |
| `pause.sh` | Stops the API from serving (reserved concurrency 0), or resumes it |

## Deploying

```bash
aws login                         # region us-east-1
bash deploy/aws/deploy.sh
cd web && vercel deploy --prod
```

`deploy.sh` reads these optional variables:

| Variable | Default | Meaning |
|---|---|---|
| `AWS_REGION` | `us-east-1` | Region for every resource |
| `NAME` | `leveledge-api` | Function, repository and role name |
| `MEMORY` | `1536` | Function memory in MB; CPU scales with it |
| `TIMEOUT` | `120` | Seconds before a request is cut off |
| `CONCURRENCY` | `5` | The most copies that run at once |
| `ROLE_ARN` | (created) | Use an existing role, such as an AWS Academy `LabRole` |
| `BUDGET_EMAIL` | (none) | Create a monthly budget alert for this address |

## Pausing and resuming

```bash
bash deploy/aws/pause.sh          # refuse all requests; nothing runs, nothing is billed for compute
bash deploy/aws/pause.sh resume   # serve again
```

## Testing the image locally

Lambda's filesystem is read-only except `/tmp`. Run the image the same way:

```bash
docker build --platform linux/arm64 --provenance=false -f deploy/aws/Dockerfile -t leveledge-api:local .
docker run --rm --read-only --tmpfs /tmp -p 8089:8080 leveledge-api:local
curl -s http://127.0.0.1:8089/api/health
```
