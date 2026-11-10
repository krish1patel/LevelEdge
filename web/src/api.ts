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
