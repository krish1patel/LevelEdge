# deploy

Deployment configuration for the LevelEdge dashboard.

| Folder | Target | Status |
|---|---|---|
| [`aws/`](aws/README.md) | The API on AWS Lambda; the frontend on Vercel | **Production:** <https://leveledge.vercel.app> |
| [`huggingface/`](huggingface/README.md) | The original HuggingFace Space (API and frontend in one container) | Synced from `main` on every push |

For the full process, cost controls and secrets, see [docs/deployment.md](../docs/deployment.md).
