#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Tokens-per-second benchmark for the Atlas metal serving path.

Measures, over the OpenAI-compatible streaming endpoint, the same two
numbers `mlx_lm` prints for the reference Bonsai-demo run:

  prompt tps      = prompt_tokens / time-to-first-token
  generation tps  = (completion_tokens - 1) / (last-chunk - first-chunk)

Token counts come from the server's `usage` object (via
`stream_options.include_usage`), not from counting SSE chunks, so
multi-token chunks can't skew the result. Pure stdlib — no deps.

Usage (server already running):
  python3 bench/bonsai_metal_tps.py --base-url http://127.0.0.1:8899 \
      --model bonsai-27b-atlas --max-tokens 128 --runs 3

Or let scripts/bench-bonsai-metal.sh build/start the server first.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PROMPT = (
    "Write a detailed, flowing essay about the history of ocean navigation, "
    "covering ancient Polynesian wayfinding, the age of sail, chronometers, "
    "and modern satellite systems. Do not use lists; write continuous prose."
)

# ~60-token paragraph used to synthesize long prompts for prefill runs.
FILLER = (
    "The tide rolled in across the shallow bay while gulls wheeled over the "
    "breakwater, and the old lighthouse keeper noted the wind shift in his "
    "logbook before trimming the lamp for the evening watch. "
)


def build_prompt(target_prompt_tokens: int) -> str:
    if target_prompt_tokens <= 0:
        return DEFAULT_PROMPT
    # ~0.75 words/token heuristic is close enough for a bandwidth benchmark;
    # the measured prompt_tokens from `usage` is what gets reported.
    reps = max(1, target_prompt_tokens // 60)
    return FILLER * reps + "\n\nSummarize the scene above in one sentence."


def stream_once(base_url: str, model: str, prompt: str, max_tokens: int,
                timeout: float) -> dict:
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t_start = time.perf_counter()
    t_first = None
    t_last = None
    chunks = 0
    usage = None
    finish_reason = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            choices = d.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
            # Only count deltas that carry generated payload; the initial
            # role-only chunk (if any) is emitted before decoding starts.
            if delta.get("content") or delta.get("tool_calls") or delta.get(
                    "reasoning_content"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                chunks += 1
    if t_first is None or usage is None:
        raise RuntimeError(
            f"stream produced no content chunks or no usage "
            f"(chunks={chunks}, usage={usage})")

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    ttft = t_first - t_start
    decode_time = (t_last - t_first) if t_last else 0.0
    gen_tps = ((completion_tokens - 1) / decode_time
               if completion_tokens >= 2 and decode_time > 0 else 0.0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "decode_s": decode_time,
        "prompt_tps": prompt_tokens / ttft if ttft > 0 else 0.0,
        "generation_tps": gen_tps,
        "finish_reason": finish_reason,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8899")
    ap.add_argument("--model", default="bonsai-27b-atlas")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--prompt", default=None,
                    help="override the default essay prompt")
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="synthesize a prompt of roughly this many tokens "
                         "(prefill benchmark)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--json", action="store_true",
                    help="emit one JSON object with all runs + medians")
    args = ap.parse_args()

    prompt = args.prompt or build_prompt(args.prompt_tokens)

    for i in range(args.warmup):
        try:
            r = stream_once(args.base_url, args.model, prompt,
                            args.max_tokens, args.timeout)
            print(f"warmup {i + 1}/{args.warmup}: "
                  f"gen {r['generation_tps']:.2f} t/s", file=sys.stderr)
        except (urllib.error.URLError, RuntimeError) as e:
            print(f"warmup failed: {e}", file=sys.stderr)
            return 1

    runs = []
    for i in range(args.runs):
        r = stream_once(args.base_url, args.model, prompt,
                        args.max_tokens, args.timeout)
        runs.append(r)
        print(f"run {i + 1}/{args.runs}: prompt {r['prompt_tokens']} tok "
              f"@ {r['prompt_tps']:.2f} t/s (ttft {r['ttft_s']:.2f}s) | "
              f"generation {r['completion_tokens']} tok "
              f"@ {r['generation_tps']:.2f} t/s"
              f" | finish={r['finish_reason']}", file=sys.stderr)

    med = {
        "prompt_tps": statistics.median(r["prompt_tps"] for r in runs),
        "generation_tps": statistics.median(
            r["generation_tps"] for r in runs),
        "ttft_s": statistics.median(r["ttft_s"] for r in runs),
    }
    if args.json:
        print(json.dumps({"runs": runs, "median": med}, indent=2))
    else:
        print(f"| {'Metric':<12} | {'Tokens':>8} | {'Speed (t/s)':>12} |")
        print(f"|{'-' * 14}|{'-' * 10}|{'-' * 14}|")
        print(f"| {'Prompt':<12} | {runs[-1]['prompt_tokens']:>8} "
              f"| {med['prompt_tps']:>12.2f} |")
        print(f"| {'Generation':<12} | {runs[-1]['completion_tokens']:>8} "
              f"| {med['generation_tps']:>12.2f} |")
        print(f"\nmedian ttft: {med['ttft_s']:.2f}s over {args.runs} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
