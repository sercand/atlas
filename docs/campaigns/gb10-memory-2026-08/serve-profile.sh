#!/usr/bin/env bash
# The §3 serve profile, verbatim. $1 = binary path, $2.. = extra flags.
# Keeping this in one file is the one-variable rule made mechanical: the only
# thing that may differ between two runs is the argv this script is given.
BIN="$1"; shift
exec env LIBRARY_PATH=$HOME/nccl-shim LD_LIBRARY_PATH=$HOME/nccl-shim \
ATLAS_MTP_ACCEPT_DEBUG=1 \
ATLAS_PREFILL_CODISPATCH=1 ATLAS_FP8_ROWWISE=1 ATLAS_MTP_DCUT_RATIO=1.0 \
ATLAS_MTP_K_LADDER=1:3,2:1,4:2,8:2,16:1 \
"$BIN" serve unsloth/Qwen3.8-27B-NVFP4 \
  --host 127.0.0.1 --port 36200 --model-name Qwen3.8-27B \
  --max-seq-len 2048 --max-batch-size 2 --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 --enable-prefix-caching true \
  --ssm-cache-slots 8 --ssm-checkpoint-interval 32 \
  --vision-max-pixels 16384 \
  --speculative --num-drafts 3 --mtp-quantization bf16 \
  --scheduling-policy fifo --tool-call-parser qwen3_coder --disable-tool-grammar true \
  --request-timeout 0 --vision-allow-remote-images \
  --ssm-h-dtype f16-pool --gdn-fused-norm --ssm-batched-recurrent \
  --ssm-tail-midchunk false --mtp-gate force --prefill-varlen-batch --no-tui "$@"
