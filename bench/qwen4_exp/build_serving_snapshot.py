#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Build a serving snapshot for RadixArk qwen4_exp with the BF16-converted PLE table.

Symlinks every original snapshot file except model-plefp8-* and the index;
adds model-plebf16-00000.safetensors; writes a patched index.json whose PLE
shard entries point at the new file (weight_scale entry dropped — the BF16
path never reads it).
"""
import json, glob, os, sys

SNAP = os.environ.get('QWEN4EXP_SRC_SNAPSHOT') or glob.glob(os.path.expanduser(
    '~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/*/'))[0].rstrip('/')
OUTDIR = os.environ.get('QWEN4EXP_SERVE_SNAPSHOT', '/home/otsimo/work/qwen4exp-serve/snapshot')
PLEBF16 = os.environ.get('QWEN4EXP_PLEBF16', '/home/otsimo/work/qwen4exp-serve/model-plebf16-00000.safetensors')
os.makedirs(OUTDIR, exist_ok=True)

with open(SNAP + '/model.safetensors.index.json') as f:
    idx = json.load(f)

wm = idx['weight_map']
patched, dropped = 0, 0
for k in list(wm):
    if '.ngram_embedding.shard_' in k and k.endswith('.weight'):
        wm[k] = 'model-plebf16-00000.safetensors'
        patched += 1
    elif k.endswith('.ngram_embedding.weight_scale'):
        del wm[k]
        dropped += 1
print(f'patched {patched} shard entries, dropped {dropped} scale entries')
assert patched == 128

for f in sorted(os.listdir(SNAP)):
    if f.startswith('model-plefp8-') or f == 'model.safetensors.index.json':
        continue
    dst = os.path.join(OUTDIR, f)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(os.path.join(SNAP, f)), dst)

dst = os.path.join(OUTDIR, 'model-plebf16-00000.safetensors')
if os.path.islink(dst) or os.path.exists(dst):
    os.remove(dst)
os.symlink(PLEBF16, dst)

with open(os.path.join(OUTDIR, 'model.safetensors.index.json'), 'w') as f:
    json.dump(idx, f)
print('snapshot at', OUTDIR, '-', len(os.listdir(OUTDIR)), 'entries')
