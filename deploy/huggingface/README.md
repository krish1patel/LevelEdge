# deploy/huggingface

Configuration for the HuggingFace Space that `.github/workflows/sync_to_hf.yml` updates on every push to `main`.

| File | Purpose |
|---|---|
| `space_header.yml` | The Space's settings (Docker SDK, port 7860, title). HuggingFace reads them from YAML at the top of the Space's `README.md`. |

`scripts/sync_space.sh` prepends `space_header.yml` to `README.md` in the snapshot it pushes, so the GitHub README stays free of the YAML block. The Space builds the root `Dockerfile`, which serves the API and the built frontend from one container.

To change the Space's title, emoji or port, edit `space_header.yml`. The change reaches the Space on the next push to `main`.
