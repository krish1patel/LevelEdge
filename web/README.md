# web

The LevelEdge dashboard: a React 18 + TypeScript app built with Vite. Its charts are hand-built SVG with `d3-scale` and `d3-shape`. It's deployed to Vercel at <https://leveledge.vercel.app>.

## Developing

### Prerequisites

- Node.js 22
- The API running on port 7860 (`uvicorn app.api:app --port 7860` from the repository root)

### Running with hot reload

```bash
npm ci
npm run dev            # http://127.0.0.1:5173; /api is forwarded to port 7860
```

### Building

```bash
npm run build          # type-checks, then writes the static site to dist/
```

`app/api.py` serves `dist/` when it exists, so after a build the whole dashboard runs from `uvicorn` alone.

## Structure

| Path | Contents |
|---|---|
| `src/App.tsx` | The shell: the Portfolio, Forecaster and Research sections, and hash routing (`#research`, `#forecast`, `#diagnostics`, `#track`) |
| `src/api.ts` | Typed client for every API endpoint |
| `src/format.ts` | Number, percent, money and date formatting |
| `src/styles.css` | Design tokens and component styles, with light and dark themes |
| `src/components/Portfolio.tsx` | The book: targets, live account, backtest, trend flips |
| `src/components/Outlook.tsx` | The S&P 500 outlook card and its fan chart |
| `src/components/Research.tsx` | The research log |
| `src/components/FanChart.tsx` | The forecaster's signature chart; the x-axis is in trading bars |
| `src/components/LevelsPanel.tsx`, `TrustPanel.tsx`, `Diagnostics.tsx` | Forecast levels, how far to trust the forecast, and model diagnostics |
| `src/components/TrackRecord.tsx`, `LedgerCard.tsx` | The live record: the verifiable ledger, the forward test and the offline benchmark |
| `src/components/XYChart.tsx`, `useWidth.ts` | A shared line and dot chart, and a responsive-width hook |
| `vercel.json` | Vercel build settings and the `/api/*` rewrite to AWS Lambda (written by `deploy/aws/deploy.sh`) |

## Following the design system

[`DESIGN.md`](../DESIGN.md) defines the tokens and patterns, and [`PRODUCT.md`](../PRODUCT.md) describes the users. In short:

- Use the tokens in `styles.css` (`--accent`, `--fan-inner`, `--good`, `--bad`, the type scale) instead of literal colours or sizes.
- Every chart has an `aria-label` summary, and every state (loading, empty, error) has a designed view.
- Layouts work at 390 px with no horizontal page scroll; wide tables scroll inside `.table-scroll`.

## Deploying

```bash
vercel deploy --prod
```

See [docs/deployment.md](../docs/deployment.md) for the full process, including the API.
