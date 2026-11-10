export interface Backtest {
  brier_skill: number | null;
  base_rate: number;
  blend_brier: number;
  climatology_brier: number;
  baseline_brier: number;
  n: number;
}

export interface LevelRow {
  input: string;
  level: number;
  move_pct: number;
  p_above: number;
  p_below: number;
  p_baseline: number;
  p_model: number;
  k_sigma: number;
  backtest: Backtest | null;
}

export interface Score {
  log_loss: number;
  brier: number;
  auc: number | null;
  n: number;
}

export interface ForecastResponse {
  meta: {
    ticker: string;
    interval: string;
    spot: number;
    anchor_time: string;
    effective_target: string;
    requested_target: string;
    horizon_bars: number;
    sessions_crossed: number;
    sigma_pct: number;
    is_backtest: boolean;
    as_of: string;
    is_crypto: boolean;
    elapsed_s: number;
  };
  levels: LevelRow[];
  events: { kind: string; date: string }[];
  quantiles: Record<string, number>;
  history: { t: string; o: number; h: number; l: number; c: number }[];
  fan: { time: string; step: number; q10: number; q25: number; q50: number; q75: number; q90: number }[];
  curve: { level: number; p_above: number; p_baseline: number }[];
  trust: { verdict: "model_adds_skill" | "baseline" | "unvalidated"; detail: string; coverage80: number | null; xgb_weight: number; baseline_weight: number };
  diagnostics: {
    metrics: Record<string, Score>;
    regime_metrics: Record<string, Record<string, Score>>;
    weights: { xgb: number; baseline: number; intercept: number };
    vol_half_life_bars: number | null;
    vol_inflation: number;
    rows: number;
    history_start: string;
    history_end: string;
    reliability: { forecast: number; observed: number; n: number }[];
    importance: { name: string; share: number }[];
  };
}

export interface ForecastRequest {
  ticker: string;
  target: string;
  interval: string;
  levels: string[];
  as_of?: string | null;
}

export class ApiError extends Error {}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    throw new ApiError("Can't reach the LevelEdge server. Check your connection and try again.");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof body?.detail === "string" ? body.detail
      : Array.isArray(body?.detail) ? "Some inputs are invalid. Check the ticker, target and levels." : `Request failed (${res.status}).`;
    throw new ApiError(detail);
  }
  return body as T;
}

export const getForecast = (req: ForecastRequest, signal?: AbortSignal) =>
  call<ForecastResponse>("/api/forecast", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });

export interface TrackModel {
  n: number;
  brier: number;
  log_loss: number;
  brier_skill_vs_base_rate: number;
  ece: number;
  base_rate: number;
  reliability: { forecast: number; observed: number; n: number }[];
}

export type TrackRecord =
  | { status: "not_configured" | "error" | "empty"; message: string }
  | {
      status: "ok";
      rows: number;
      first: string;
      last: string;
      days: number;
      by_model: Record<string, TrackModel>;
      daily: { model_type: string; day: string; brier: number; n: number }[];
      by_interval: ({ model_type: string; interval: string } & Omit<TrackModel, "reliability">)[];
    };

export const getTrackRecord = (days: number) => call<TrackRecord>(`/api/track-record?days=${days}`);

export type Benchmark =
  | { status: "missing"; message: string }
  | {
      status: "ok";
      forecasts: number;
      days: [string, string];
      overall: Record<string, { log_loss: number; brier: number; auc: number; ece: number; brier_skill_vs_v1: number }>;
      [k: string]: unknown;
    };

export const getBenchmark = () => call<Benchmark>("/api/benchmark");

// ---------------------------------------------------------------------------
// paper portfolio
// ---------------------------------------------------------------------------
export interface PerfMetrics {
  start: string;
  end: string;
  years: number;
  cagr: number;
  vol: number;
  sharpe: number;
  sortino: number | null;
  max_drawdown: number;
  calmar: number | null;
  worst_month: number;
  pct_positive_months: number;
  total_return: number;
  turnover_per_year?: number;
}

export type SeriesKey = "book" | "sixty_forty" | "spy_buy_hold" | "trend_sleeve_alone";
export interface DiffCI { diff: number; ci95: [number, number] }

export interface PortfolioBacktest {
  strategy: string;
  period: [string, string];
  labels: Record<SeriesKey, string>;
  cost_bps: number;
  metrics: Record<SeriesKey, PerfMetrics>;
  dev: Record<SeriesKey, PerfMetrics>;
  holdout: Record<SeriesKey, PerfMetrics>;
  sharpe_vs_sixty_forty: { full: DiffCI; holdout: DiffCI };
  random_sleeve: { sims: number; p_long: number; sharpe_p5_median_p95: [number, number, number]; max_dd_median: number; p_value_book_sharpe: number };
  crisis_windows: ({ label: string; start: string; end: string } & Record<SeriesKey, number>)[];
  equity_weekly: ({ date: string } & Record<SeriesKey, number>)[];
  drawdown_weekly: ({ date: string } & Record<SeriesKey, number>)[];
}

export interface TargetAsset {
  symbol: string;
  name: string;
  weight: number;
  core_weight: number;
  trend_weight: number;
  trend_on: boolean;
  return_12m: number;
  vol_annual: number;
  last_close: number;
}

export interface Targets {
  strategy: string;
  as_of_close: string;
  rebalance_today: boolean;
  next_rebalance: string;
  weights: Record<string, number>;
  cash_weight: number;
  gross: number;
  leverage: number;
  assets: TargetAsset[];
  sleeve_gross: number;
}

export type LivePaper =
  | { status: "not_configured" | "error"; message: string }
  | {
      status: "ok";
      paper: boolean;
      account: { equity: number; cash: number; last_equity: number };
      positions: { symbol: string; qty: number; market_value: number; unrealized_pl: number; weight: number }[];
      orders: { symbol: string; side: string; status: string; notional: number | null; qty: number | null;
                filled_avg_price: number | null; submitted_at: string; filled_at: string | null }[];
      curve: { date: string; book: number; spy_buy_hold: number | null; sixty_forty: number | null }[];
    };

export interface PortfolioResponse {
  strategy: string;
  design: { core: Record<string, number>; core_share: number; trend_share: number };
  targets: Targets | null;
  targets_status: string;
  backtest: PortfolioBacktest | null;
  live: LivePaper;
}

export const getPortfolio = () => call<PortfolioResponse>("/api/portfolio");

// ---------------------------------------------------------------------------
// research log
// ---------------------------------------------------------------------------
export interface Cal { n: number; brier: number; log_loss: number; ece: number }
export interface TradeStats {
  trades: number;
  avg_pnl_cents?: number;
  avg_pnl_cents_ci95?: [number, number];
  random_picker?: { p_value_strategy_beats_random: number };
}

export interface ResearchResponse {
  forecaster?: { forecasts: number; overall: Record<string, { brier: number; log_loss: number }>;
    brier_diff_v2_minus_v1?: { point: number; ci95: [number, number] } };
  kalshi_taker?: { rows: number; events: number; period: [string, string];
    liquid: { rows: number; calibration: Record<string, Cal>; brier_v2_minus_market: DiffCI; primary: TradeStats } };
  kalshi_maker?: { variants: number; rows: number;
    results: { variant: string; model_avg_pnl_cents: number; model_ci95: [number, number]; model_fills: number; mid_avg_pnl_cents: number }[] };
  kalshi_horizon?: { rows: number; events: number; period: [string, string];
    liquid: { calibration: Record<string, Cal>; brier_v2_minus_market: DiffCI; primary: TradeStats } };
  etf_trend?: { period: [string, string]; metrics: Record<string, PerfMetrics>; vol_engine: { ewma: number; leveledge: number };
    random_picker: { sims: number; sharpe_mean: number; p_value_strategy_sharpe: number } };
  etf_search?: { period: { dev: [string, string]; holdout: [string, string] }; trials: number; selected: string;
    deflated_sharpe_dev: { sharpe_annual: number; expected_max_sharpe_of_noise_annual: number; deflated_sharpe_prob: number };
    dev: Record<string, PerfMetrics>; holdout: Record<string, PerfMetrics>;
    holdout_sharpe_vs: Record<string, DiffCI>;
    rank_stability: { spearman_dev_vs_holdout: number; top10_dev_holdout_ranks: number[] } };
  etf_search_points?: { variant: string; dev: number; hold: number }[];
}

export const getResearch = () => call<ResearchResponse>("/api/research");
