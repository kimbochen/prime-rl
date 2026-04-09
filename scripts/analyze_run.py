"""Analyze RL training run: timing breakdown, bottleneck identification, and pipeline visualization.

Usage:
    uv run python scripts/analyze_run.py --entity kimbochen --project qwen30b-swe --run-id 2e00109a358f43598ae40840171838de
"""

import argparse

import pandas as pd
import wandb


def fetch_run(entity: str, project: str, run_id: str):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    df = run.history(pandas=True)

    # Shared W&B mode puts different labels' metrics in separate rows.
    # Merge rows that share the same step by taking the first non-NaN value per column.
    if "step" in df.columns:
        df = df.groupby("step", as_index=False).first()

    return df, run.config


def print_config(config: dict):
    print("=" * 70)
    print("SYSTEM CONFIGURATION")
    print("=" * 70)

    def get(path: str, default=None):
        """Dot-separated path lookup into nested dicts."""
        obj = config
        for key in path.split("."):
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return default
        return obj

    ep = get("model.ep", 0)
    cp = get("model.cp", 1)
    fsdp_offload = get("model.fsdp_cpu_offload", False)
    optim_offload = get("model.optim_cpu_offload", False)


    attn = get("model.attn", "?")

    model_name = get("model.name", "?")

    print(f"\n  Trainer")
    print(f"    Model:              {model_name}")
    print(f"    Attention:          {attn}")
    print(f"    EP:                 {ep}")
    print(f"    CP:                 {cp}")
    print(f"    FSDP CPU offload:   {fsdp_offload}")
    print(f"    Optim CPU offload:  {optim_offload}")


    batch_size = get("batch_size", "?")
    oversample = get("oversampling_factor", 1)
    max_async = get("max_async_level", "?")
    max_offpolicy = get("max_off_policy_steps", "?")
    print(f"\n  Orchestrator")
    print(f"    Batch size:         {batch_size}")
    print(f"    Oversampling:       {oversample}x  ({int(batch_size * oversample)} rollouts started)")
    print(f"    Max async level:    {max_async}")
    print(f"    Max off-policy:     {max_offpolicy} steps")

    train_envs = get("train.env", [])
    if train_envs:
        env = train_envs[0]
        env_args = env.get("args", {})
        print(f"\n  Environment")
        print(f"    Name:               {env.get('name', '?')} ({env.get('id', '?')})")
        print(f"    Max turns:          {env_args.get('max_turns', '?')}")
        print(f"    Sandbox workers:    {env_args.get('sandbox_client_max_workers', '?')}")
        print(f"    Sandbox timeout:    {env_args.get('sandbox_command_timeout', '?')}s")
        print(f"    Sandbox resources:  {env_args.get('cpu_cores', '?')} CPU, {env_args.get('memory_gb', '?')}GB RAM, {env_args.get('disk_size_gb', '?')}GB disk")

    wb = get("weight_broadcast", {})
    if wb:
        print(f"\n  Weight broadcast")
        print(f"    Type:               {wb.get('type', '?')}")
        print(f"    Inference GPUs:     {wb.get('inference_world_size', '?')}")

    print()


def safe_mean(series):
    vals = series.dropna()
    return vals.mean() if len(vals) > 0 else 0


def safe_std(series):
    vals = series.dropna()
    return vals.std() if len(vals) > 1 else 0


def analyze(df: pd.DataFrame):
    print("=" * 70)
    print("PIPELINE TIMING ANALYSIS")
    print("=" * 70)

    # ── Trainer metrics ──
    has_trainer = "time/step" in df.columns and df["time/step"].notna().any()
    if has_trainer:
        t = df[df["time/step"].notna()]
        step_time = t["time/step"]
        wait_time = t["time/wait_for_batch"]
        fb_time = t.get("time/forward_backward", pd.Series(dtype=float))
        bcast_time = t.get("time/broadcast_weights", pd.Series(dtype=float))

        print(f"\n{'─' * 40}")
        print("TRAINER (per step)")
        print(f"{'─' * 40}")
        print(f"  Total step time:      {safe_mean(step_time):>8.1f}s  (±{safe_std(step_time):.1f}s)")
        print(f"  Wait for batch:       {safe_mean(wait_time):>8.1f}s  (±{safe_std(wait_time):.1f}s)")
        if fb_time.notna().any():
            print(f"  Forward/backward:     {safe_mean(fb_time):>8.1f}s  (±{safe_std(fb_time):.1f}s)")
        if bcast_time.notna().any():
            print(f"  Broadcast weights:    {safe_mean(bcast_time):>8.1f}s  (±{safe_std(bcast_time):.1f}s)")

        avg_step = safe_mean(step_time)
        avg_wait = safe_mean(wait_time)
        avg_bcast = safe_mean(bcast_time)
        avg_fb = safe_mean(fb_time)
        wait_pct = (avg_wait / avg_step * 100) if avg_step > 0 else 0
        compute_time = avg_step - avg_wait - avg_bcast
        compute_pct = (compute_time / avg_step * 100) if avg_step > 0 else 0
        bcast_pct = (avg_bcast / avg_step * 100) if avg_step > 0 else 0

        print(f"\n  Step breakdown:")
        print(f"    Waiting (idle):     {wait_pct:>5.1f}%  ({avg_wait:.0f}s)")
        print(f"    Compute:            {compute_pct:>5.1f}%  ({compute_time:.0f}s)")
        if avg_bcast > 0:
            print(f"    Weight broadcast:   {bcast_pct:>5.1f}%  ({avg_bcast:.0f}s)")

        throughput = t.get("perf/throughput", pd.Series(dtype=float))
        mfu = t.get("perf/mfu", pd.Series(dtype=float))
        if throughput.notna().any():
            print(f"\n  Throughput:           {safe_mean(throughput):>8.0f} tok/s  (±{safe_std(throughput):.0f})")
        if mfu.notna().any():
            print(f"  MFU:                  {safe_mean(mfu):>7.1f}%  (±{safe_std(mfu):.1f}%)")

    # ── Orchestrator / Rollout metrics ──
    has_orch = "generation_ms/all/mean" in df.columns and df["generation_ms/all/mean"].notna().any()
    if has_orch:
        o = df[df["generation_ms/all/mean"].notna()]
        gen_ms = o["generation_ms/all/mean"]
        score_ms = o["scoring_ms/all/mean"]
        num_turns = o["num_turns/swe/mean"]
        comp_len = o.get("decode_len/swe/mean", o.get("completion_len/swe/mean", pd.Series(dtype=float)))

        avg_gen_ms = safe_mean(gen_ms)
        avg_score_ms = safe_mean(score_ms)
        avg_turns = safe_mean(num_turns)
        avg_comp_len = safe_mean(comp_len)

        print(f"\n{'─' * 40}")
        print("ROLLOUTS (per sample, averaged)")
        print(f"{'─' * 40}")
        print(f"  Total rollout time:   {avg_gen_ms/1000:>8.1f}s  (±{safe_std(gen_ms)/1000:.1f}s)")
        print(f"  Scoring time:         {avg_score_ms/1000:>8.1f}s  (±{safe_std(score_ms)/1000:.1f}s)")
        print(f"  Agent interaction:    {(avg_gen_ms - avg_score_ms)/1000:>8.1f}s  (rollout - scoring)")
        print(f"  Num turns:            {avg_turns:>8.1f}   (max: {safe_mean(o['num_turns/swe/max']):.0f})")
        print(f"  Completion len:       {avg_comp_len:>8.0f} tokens")

        if avg_turns > 0:
            per_turn_ms = avg_gen_ms / avg_turns
            print(f"\n  Per-turn estimate:    {per_turn_ms:>8.0f}ms")

        # Sandbox-specific metrics
        sandbox_cmd = o.get("metrics/swe/sandbox_command_execution_time", pd.Series(dtype=float))
        sandbox_wait = o.get("metrics/swe/sandbox_ready_wait_time", pd.Series(dtype=float))
        sandbox_timeout = o.get("metrics/swe/sandbox_timeout", pd.Series(dtype=float))
        sandbox_oom = o.get("metrics/swe/sandbox_oom", pd.Series(dtype=float))
        tool_calls = o.get("metrics/swe/total_tool_calls", pd.Series(dtype=float))
        bash_calls = o.get("metrics/swe/execute_bash_calls", pd.Series(dtype=float))
        edit_calls = o.get("metrics/swe/edit_via_str_replace_calls", pd.Series(dtype=float))

        if sandbox_cmd.notna().any():
            print(f"\n  Sandbox details:")
            print(f"    Command exec time:  {safe_mean(sandbox_cmd):>8.1f}s")
            if sandbox_wait.notna().any():
                print(f"    Container wait:     {safe_mean(sandbox_wait):>8.1f}s")
            if tool_calls.notna().any():
                print(f"    Tool calls/rollout: {safe_mean(tool_calls):>8.1f}")
            if bash_calls.notna().any():
                print(f"    Bash calls:         {safe_mean(bash_calls):>8.1f}")
            if edit_calls.notna().any():
                print(f"    Edit calls:         {safe_mean(edit_calls):>8.1f}")
            if sandbox_timeout.notna().any() and safe_mean(sandbox_timeout) > 0:
                print(f"    Timeouts:           {safe_mean(sandbox_timeout):>8.2f}")
            if sandbox_oom.notna().any() and safe_mean(sandbox_oom) > 0:
                print(f"    OOMs:               {safe_mean(sandbox_oom):>8.2f}")

    # ── Inference worker metrics ──
    has_infer = "inference/vllm:num_requests_running" in df.columns and df["inference/vllm:num_requests_running"].notna().any()
    if has_infer:
        inf = df[df["inference/vllm:num_requests_running"].notna()]
        running = inf["inference/vllm:num_requests_running"]
        waiting = inf["inference/vllm:num_requests_waiting"]

        # New metric names (from updated scraper)
        cache = inf.get("inference/vllm:kv_cache_usage_perc", pd.Series(dtype=float))
        gen_tps = inf.get("inference/vllm:generation_tokens_per_sec", pd.Series(dtype=float))
        prompt_tps = inf.get("inference/vllm:prompt_tokens_per_sec", pd.Series(dtype=float))
        e2e_latency = inf.get("inference/vllm:e2e_request_latency_seconds_avg", pd.Series(dtype=float))
        ttft = inf.get("inference/vllm:time_to_first_token_seconds_avg", pd.Series(dtype=float))
        itl = inf.get("inference/vllm:inter_token_latency_seconds_avg", pd.Series(dtype=float))
        queue_time = inf.get("inference/vllm:request_queue_time_seconds_avg", pd.Series(dtype=float))
        cache_hit = inf.get("inference/prefix_cache_hit_rate", pd.Series(dtype=float))
        req_rate = inf.get("inference/vllm:request_success_per_sec", pd.Series(dtype=float))

        print(f"\n{'─' * 40}")
        print("INFERENCE WORKER (vLLM)")
        print(f"{'─' * 40}")

        # Throughput
        print(f"  Throughput:")
        if gen_tps.notna().any():
            print(f"    Generation:          {safe_mean(gen_tps):>7.0f} tok/s  (±{safe_std(gen_tps):.0f})")
        if prompt_tps.notna().any():
            print(f"    Prompt:              {safe_mean(prompt_tps):>7.0f} tok/s  (±{safe_std(prompt_tps):.0f})")
        if req_rate.notna().any():
            print(f"    Requests:            {safe_mean(req_rate):>7.1f} req/s  (±{safe_std(req_rate):.1f})")

        # Latency
        has_latency = any(s.notna().any() for s in [e2e_latency, ttft, itl, queue_time])
        if has_latency:
            print(f"  Latency (per request avg):")
            if e2e_latency.notna().any():
                print(f"    End-to-end:          {safe_mean(e2e_latency):>7.1f}s")
            if ttft.notna().any():
                print(f"    Time to first token: {safe_mean(ttft)*1000:>7.0f}ms")
            if itl.notna().any():
                print(f"    Inter-token:         {safe_mean(itl)*1000:>7.1f}ms")
            if queue_time.notna().any():
                print(f"    Queue wait:          {safe_mean(queue_time)*1000:>7.1f}ms")

        # Saturation
        print(f"  Saturation:")
        print(f"    Requests running:    {safe_mean(running):>7.1f}  (max: {running.max():.0f})")
        print(f"    Requests waiting:    {safe_mean(waiting):>7.1f}  (max: {waiting.max():.0f})")
        if cache.notna().any():
            print(f"    KV cache usage:      {safe_mean(cache)*100:>6.1f}%  (max: {cache.max()*100:.1f}%)")
        if cache_hit.notna().any():
            print(f"    Prefix cache hit:    {safe_mean(cache_hit)*100:>6.1f}%")

    # ── Off-policy / async metrics ──
    off_policy_mean = df.get("off_policy_level/all/mean", pd.Series(dtype=float))
    off_policy_max = df.get("off_policy_level/all/max", pd.Series(dtype=float))
    async_level = df.get("scheduler/async_level", pd.Series(dtype=float))
    cancelled = df.get("scheduler/cancelled_rollouts", pd.Series(dtype=float))
    inflight = df.get("scheduler/inflight_rollouts", pd.Series(dtype=float))
    inflight_samples = df.get("scheduler/inflight_samples", pd.Series(dtype=float))

    if off_policy_mean.notna().any():
        print(f"\n{'─' * 40}")
        print("OFF-POLICY & SCHEDULING")
        print(f"{'─' * 40}")
        print(f"  Off-policy level (mean): {safe_mean(off_policy_mean):>5.2f}  (avg across rollouts in batch)")
        print(f"  Off-policy level (max):  {safe_mean(off_policy_max):>5.2f}  (worst straggler, avg across steps)")
        print(f"  Off-policy peak:         {off_policy_max.max():>5.0f}     (worst single rollout ever)")
        if async_level.notna().any():
            print(f"  Async level:             {safe_mean(async_level):>5.2f}  (how far ahead orchestrator is)")
        if cancelled.notna().any() and safe_mean(cancelled) > 0:
            print(f"  Cancelled rollouts:      {safe_mean(cancelled):>5.1f}/step")
        if inflight.notna().any():
            print(f"  In-flight rollouts:      {safe_mean(inflight):>5.0f}  (avg concurrent)")
        if inflight_samples.notna().any():
            print(f"  In-flight samples:       {safe_mean(inflight_samples):>5.0f}  (avg concurrent)")

        # Staleness distribution
        fresh = off_policy_mean[off_policy_mean.notna()]
        stale_batches = (fresh > 1).sum()
        total_batches = len(fresh)
        if total_batches > 0:
            print(f"\n  Batch freshness:")
            print(f"    Mostly on-policy (≤1): {total_batches - stale_batches}/{total_batches} batches")
            print(f"    Partially stale (>1):  {stale_batches}/{total_batches} batches")
            if off_policy_max.max() > 0:
                headroom = 16 - off_policy_max.max()
                print(f"    Headroom to limit:     {headroom:.0f} steps  (max_off_policy=16, peak={off_policy_max.max():.0f})")

    # ── GPU vs Sandbox time estimation ──
    if has_orch:
        print(f"\n{'─' * 40}")
        print("GPU vs SANDBOX TIME (estimated)")
        print(f"{'─' * 40}")

        # Use sandbox_command_execution_time if available for direct measurement
        if sandbox_cmd.notna().any():
            avg_sandbox_s = safe_mean(sandbox_cmd)
            avg_sandbox_wait_s = safe_mean(sandbox_wait) if sandbox_wait.notna().any() else 0
            avg_rollout_s = avg_gen_ms / 1000
            avg_scoring_s = avg_score_ms / 1000
            sandbox_total_s = avg_sandbox_s + avg_sandbox_wait_s + avg_scoring_s
            inference_s = avg_rollout_s - sandbox_total_s

            print(f"  Avg rollout:          {avg_rollout_s:>8.1f}s")
            print(f"    ├─ Inference (GPU): {max(0, inference_s):>8.1f}s  ({max(0, inference_s)/avg_rollout_s*100:.0f}%)")
            print(f"    ├─ Sandbox cmds:    {avg_sandbox_s:>8.1f}s  ({avg_sandbox_s/avg_rollout_s*100:.0f}%)")
            if avg_sandbox_wait_s > 0:
                print(f"    ├─ Container wait:  {avg_sandbox_wait_s:>8.1f}s  ({avg_sandbox_wait_s/avg_rollout_s*100:.0f}%)")
            print(f"    └─ Scoring (tests): {avg_scoring_s:>8.1f}s  ({avg_scoring_s/avg_rollout_s*100:.0f}%)")
        elif has_infer and gen_tps.notna().any() and avg_turns > 0:
            # Fallback: estimate from vLLM throughput
            avg_gen_tps = safe_mean(gen_tps)
            tokens_per_turn = avg_comp_len / avg_turns if avg_turns > 0 else 0
            infer_per_turn_ms = (tokens_per_turn / avg_gen_tps * 1000) if avg_gen_tps > 0 else 0
            total_infer_ms = infer_per_turn_ms * avg_turns
            sandbox_ms = avg_gen_ms - avg_score_ms - total_infer_ms

            print(f"  Avg rollout:          {avg_gen_ms:>8.0f}ms")
            print(f"    ├─ Inference (GPU): {total_infer_ms:>8.0f}ms  ({total_infer_ms/avg_gen_ms*100:.0f}%)")
            print(f"    ├─ Sandbox (est):   {max(0, sandbox_ms):>8.0f}ms  ({max(0, sandbox_ms)/avg_gen_ms*100:.0f}%)")
            print(f"    └─ Scoring (tests): {avg_score_ms:>8.0f}ms  ({avg_score_ms/avg_gen_ms*100:.0f}%)")

    # ── Utilization & Bottleneck ──
    if has_trainer:
        print(f"\n{'─' * 40}")
        print("GPU UTILIZATION & BOTTLENECK")
        print(f"{'─' * 40}")

        avg_mfu = safe_mean(mfu) if mfu.notna().any() else 0
        trainer_util = compute_time / avg_step * 100 if avg_step > 0 else 0
        # Effective MFU = MFU × fraction of time doing compute
        effective_mfu = avg_mfu * (compute_time / avg_step) if avg_step > 0 else 0

        print(f"\n  Trainer GPU utilization:  {trainer_util:>5.1f}%  (compute time / step time)")
        print(f"  Trainer GPU idle:         {wait_pct:>5.1f}%  (waiting for rollouts)")
        print(f"  MFU (during compute):     {avg_mfu:>5.1f}%  (model FLOPs utilization)")
        print(f"  Effective MFU:            {effective_mfu:>5.1f}%  (MFU × utilization)")
        print(f"")
        print(f"  ┌────────────────────────────────────┐")
        print(f"  │  Wall-clock efficiency              │")
        print(f"  │                                    │")
        print(f"  │  Of every {avg_step:.0f}s step:              │")
        print(f"  │    {compute_time:.0f}s doing useful compute     │")
        print(f"  │    {avg_wait:.0f}s waiting for rollouts      │")
        print(f"  │    {avg_bcast:.0f}s broadcasting weights      │")
        print(f"  │                                    │")
        print(f"  │  Of the {compute_time:.0f}s compute:             │")
        print(f"  │    {avg_mfu:.1f}% of peak FLOPS utilized    │")
        print(f"  │                                    │")
        print(f"  │  Net: {effective_mfu:.1f}% of peak FLOPS over    │")
        print(f"  │       the full training wall-clock  │")
        print(f"  └────────────────────────────────────┘")

        # Bottleneck diagnosis
        print(f"\n  Diagnosis:")
        if wait_pct > 50:
            print(f"  ⚠ ROLLOUT-BOUND: Trainer idle {wait_pct:.0f}% of step time.")
            print(f"    Rollout pipeline (inference + sandbox) is the bottleneck.")
            print(f"    Adding inference nodes would improve effective MFU from {effective_mfu:.1f}%.")
            if has_orch and sandbox_cmd.notna().any():
                avg_rollout_s = avg_gen_ms / 1000
                sandbox_total_s = safe_mean(sandbox_cmd) + (safe_mean(sandbox_wait) if sandbox_wait.notna().any() else 0) + avg_score_ms / 1000
                if sandbox_total_s > avg_rollout_s * 0.5:
                    print(f"    → Sandbox is the dominant cost ({sandbox_total_s:.0f}s / {avg_rollout_s:.0f}s per rollout).")
                else:
                    print(f"    → Inference is the dominant cost within rollouts.")
        elif wait_pct < 10:
            print(f"  ⚠ TRAINER-BOUND: Trainer idle only {wait_pct:.0f}% of step time.")
            print(f"    Training compute is the bottleneck. Inference GPUs may sit idle between batches.")
            print(f"    Effective MFU ({effective_mfu:.1f}%) ≈ raw MFU ({avg_mfu:.1f}%) — pipeline overhead is minimal.")
        else:
            print(f"  Trainer idle {wait_pct:.0f}% → {trainer_util:.0f}% GPU utilization.")
            print(f"  Effective MFU {effective_mfu:.1f}% vs raw MFU {avg_mfu:.1f}% — pipeline costs {avg_mfu - effective_mfu:.1f}pp.")
            if wait_pct > 25:
                print(f"  Adding inference capacity could recover up to {avg_mfu - effective_mfu:.1f}pp of effective MFU.")

    # ── Pipeline timeline (ASCII) ──
    if has_trainer:
        print(f"\n{'─' * 40}")
        print("PIPELINE TIMELINE (1 step)")
        print(f"{'─' * 40}")

        bar_width = 60

        # Trainer bar
        print(f"\n  Trainer step ({avg_step:.0f}s):")
        w = max(1, int(avg_wait / avg_step * bar_width))
        c = max(1, int(compute_time / avg_step * bar_width))
        b = max(1, int(avg_bcast / avg_step * bar_width)) if avg_bcast > 0 else 0
        c = bar_width - w - b
        bar = "░" * w + "█" * c + "▓" * b
        print(f"  [{bar}]")
        legend = f"   {'░ wait':<20} {'█ compute':<20}"
        if b > 0:
            legend += "▓ broadcast"
        print(legend)

        # Rollout bar
        if has_orch:
            avg_rollout_s = avg_gen_ms / 1000
            avg_scoring_s = avg_score_ms / 1000
            if sandbox_cmd.notna().any():
                s_cmd = safe_mean(sandbox_cmd)
                s_wait_val = safe_mean(sandbox_wait) if sandbox_wait.notna().any() else 0
                infer_s = max(0, avg_rollout_s - s_cmd - s_wait_val - avg_scoring_s)
                fracs = [
                    (infer_s, "█", "inference"),
                    (s_cmd, "░", "sandbox"),
                    (avg_scoring_s, "▓", "scoring"),
                ]
                if s_wait_val > 1:
                    fracs.insert(2, (s_wait_val, "▒", "wait"))
            else:
                agent_s = avg_rollout_s - avg_scoring_s
                fracs = [
                    (agent_s, "█", "agent"),
                    (avg_scoring_s, "▓", "scoring"),
                ]

            print(f"\n  Avg rollout ({avg_rollout_s:.0f}s):")
            total_frac = sum(f[0] for f in fracs)
            chars = []
            for val, ch, _ in fracs:
                n = max(1, int(val / total_frac * bar_width)) if total_frac > 0 else 1
                chars.append((n, ch))
            # Adjust to fill bar
            used = sum(c[0] for c in chars)
            if used < bar_width:
                chars[0] = (chars[0][0] + bar_width - used, chars[0][1])
            bar2 = "".join(ch * n for n, ch in chars)
            print(f"  [{bar2}]")
            print(f"   " + "  ".join(f"{ch} {name}" for _, ch, name in fracs))

    # ── Dependency diagram ──
    print(f"\n{'─' * 40}")
    print("COMPONENT DEPENDENCIES")
    print(f"{'─' * 40}")
    print("""
  ┌─────────────┐    prompts     ┌─────────────┐   commands    ┌─────────────┐
  │             │ ──────────────→│             │──────────────→│             │
  │  Inference  │    tokens      │ Orchestrator│   rewards     │   Sandbox   │
  │   (vLLM)    │ ←──────────────│             │←──────────────│   (SWE)     │
  │             │                │             │               │             │
  └──────┬──────┘                └──────┬──────┘               └─────────────┘
         │                              │
         │ weights                      │ batch
         │                              ↓
  ┌──────┴──────┐                ┌─────────────┐
  │             │                │             │
  │   Trainer   │←───────────────│  Filesystem  │
  │   (FSDP)    │   rollouts     │   (buffer)  │
  │             │                │             │
  └─────────────┘                └─────────────┘

  Flow: Orchestrator sends prompts to Inference, gets tokens back.
        Each turn, agent tool calls go to Sandbox, results come back.
        After all turns, Sandbox scores the rollout.
        Orchestrator buffers completed rollouts and writes batch to disk.
        Trainer reads batch, does forward/backward, broadcasts new weights.
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    print("Fetching metrics from W&B...")
    df, config = fetch_run(args.entity, args.project, args.run_id)
    print_config(config)
    analyze(df)


if __name__ == "__main__":
    main()
