"""Scrapes vLLM Prometheus /metrics endpoint and logs to shared W&B.

Intended to run as a background process alongside vLLM inference nodes.
Requires WANDB_SHARED_MODE=1 and WANDB_SHARED_RUN_ID to be set.

Usage:
    python -m prime_rl.utils.vllm_metrics_logger \
        --metrics-url http://localhost:8000/metrics \
        --project my-project \
        --interval 30
"""

import argparse
import os
import re
import time

import requests
import wandb

# Gauges (instantaneous values)
GAUGE_METRICS = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)

# Counters (monotonically increasing — we compute rates)
COUNTER_METRICS = (
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:request_success_total",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
)

# Histogram _sum/_count pairs (we compute per-request averages)
HISTOGRAM_METRICS = (
    "vllm:e2e_request_latency_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:request_queue_time_seconds",
)

ALL_PREFIXES = GAUGE_METRICS + COUNTER_METRICS + tuple(
    f"{h}_{suffix}" for h in HISTOGRAM_METRICS for suffix in ("sum", "count")
)

# Prometheus text format: metric_name{labels} value
_METRIC_LINE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)"  # metric name
    r"(?:\{[^}]*\})?"                # optional labels
    r"\s+"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",  # numeric value
    re.MULTILINE,
)


def parse_all(text: str) -> dict[str, float]:
    """Parse Prometheus text, keeping only metrics matching ALL_PREFIXES."""
    result: dict[str, float] = {}
    for match in _METRIC_LINE.finditer(text):
        name = match.group(1)
        if not name.startswith(ALL_PREFIXES):
            continue
        val = float(match.group(2))
        # For metrics with labels, take the first match (or max for gauges)
        if name in result:
            result[name] = max(result[name], val)
        else:
            result[name] = val
    return result


def main():
    parser = argparse.ArgumentParser(description="Scrape vLLM metrics and log to W&B")
    parser.add_argument("--metrics-url", required=True, help="vLLM /metrics endpoint URL")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--interval", type=int, default=30, help="Scrape interval in seconds")
    parser.add_argument("--node-rank", type=int, default=0, help="Inference node rank (for labeling)")
    args = parser.parse_args()

    run_id = os.environ.get("WANDB_SHARED_RUN_ID")
    wandb.init(
        id=run_id,
        project=args.project,
        settings=wandb.Settings(
            mode="shared",
            x_label=f"inference-{args.node_rank}",
            x_primary=False,
            x_update_finish_state=False,
        ),
    )
    wandb.define_metric("*", step_metric="step")

    prev_raw: dict[str, float] = {}
    prev_time = 0.0
    step = 0

    while True:
        time.sleep(args.interval)
        try:
            resp = requests.get(args.metrics_url, timeout=10)
            resp.raise_for_status()
        except requests.RequestException:
            continue

        now = time.monotonic()
        raw = parse_all(resp.text)
        if not raw:
            continue

        metrics: dict[str, float] = {}
        dt = now - prev_time if prev_time > 0 else args.interval

        # Gauges — log directly
        for name in GAUGE_METRICS:
            if name in raw:
                metrics[f"inference/{name}"] = raw[name]

        # Counters — compute rate per second
        if prev_raw:
            for name in COUNTER_METRICS:
                if name in raw and name in prev_raw:
                    delta = raw[name] - prev_raw[name]
                    rate_name = name.replace("_total", "_per_sec")
                    metrics[f"inference/{rate_name}"] = delta / dt

        # Histograms — compute per-request average from _sum/_count deltas
        if prev_raw:
            for name in HISTOGRAM_METRICS:
                sum_key = f"{name}_sum"
                count_key = f"{name}_count"
                if all(k in raw and k in prev_raw for k in (sum_key, count_key)):
                    d_sum = raw[sum_key] - prev_raw[sum_key]
                    d_count = raw[count_key] - prev_raw[count_key]
                    if d_count > 0:
                        metrics[f"inference/{name}_avg"] = d_sum / d_count
                    metrics[f"inference/{name}_rate"] = d_count / dt

        # Derived: prefix cache hit rate
        if prev_raw:
            hits_key = "vllm:prefix_cache_hits_total"
            queries_key = "vllm:prefix_cache_queries_total"
            if all(k in raw and k in prev_raw for k in (hits_key, queries_key)):
                d_hits = raw[hits_key] - prev_raw[hits_key]
                d_queries = raw[queries_key] - prev_raw[queries_key]
                if d_queries > 0:
                    metrics["inference/prefix_cache_hit_rate"] = d_hits / d_queries

        prev_raw = raw
        prev_time = now

        if metrics:
            step += 1
            metrics["step"] = step
            wandb.log(metrics)


if __name__ == "__main__":
    main()
