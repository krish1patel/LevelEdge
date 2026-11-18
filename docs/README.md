# LevelEdge documentation

These guides explain how LevelEdge works and how to run it. Start with the architecture overview, then read the guide for the part you're working on.

| Guide | Read it to |
|---|---|
| [Architecture](architecture.md) | Understand the components, how data moves between them, and the design decisions behind them |
| [Forecaster](forecaster.md) | Make forecasts, understand the v2 model, and read its accuracy benchmark |
| [Portfolio](portfolio.md) | Understand the traded book, the paper-trading loop, the safety checks, and how to copy trades into a real account |
| [Market scan](market-scan.md) | Understand the automated daily scan of 100 stocks and funds, its AWS pipeline, and how to run it |
| [Forecast ledger](forecast-ledger.md) | Verify the public forecast record and understand what it does and doesn't prove |
| [Deployment](deployment.md) | Deploy the dashboard to Vercel and AWS Lambda, and keep costs inside the free tiers |
| [Operations](operations.md) | Know what runs on a schedule, where alerts go, and what to do when something fails |

## Other references

- [Research log](../benchmarks/README.md): every study, its question, its protocol and its verdict.
- [Contributor guide](../CLAUDE.md): the evidence that shaped the product and the invariants that must not break.
- [Product brief](../PRODUCT.md) and [design system](../DESIGN.md): who the dashboard is for and how it looks.
