#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Convert RadixArk qwen4_exp PLE table: FP8-E4M3 + scalar scale -> one BF16 safetensors file.

Output layout matches what crates/spark-model/src/weight_loader/qwen4_exp/ple.rs expects:
all 128 shard tensors in ONE file, BF16 rows, no scale tensor.
"""
import json, struct, glob, sys, os, time
import numpy as np

SNAP = os.environ.get('QWEN4EXP_SRC_SNAPSHOT') or glob.glob(os.path.expanduser(
    '~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/*/'))[0]
OUT = sys.argv[1] if len(sys.argv) > 1 else '/home/otsimo/work/qwen4exp-serve/model-plebf16-00000.safetensors'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def read_header(path):
    with open(path, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n

# ── collect shards + scale ──
shards = {}   # idx -> (path, abs_offset, nbytes, shape)
scale = None
for f in sorted(glob.glob(SNAP + 'model-plefp8-*.safetensors')):
    hdr, data_start = read_header(f)
    for k, v in hdr.items():
        if k == '__metadata__':
            continue
        if k.endswith('.weight_scale'):
            with open(f, 'rb') as fh:
                fh.seek(data_start + v['data_offsets'][0])
                raw = fh.read(2)
            scale = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            scale = (scale << 16).view(np.float32)[0]  # bf16 -> f32
            scale_name = k
        elif '.ngram_embedding.shard_' in k:
            idx = int(k.rsplit('shard_', 1)[1].split('.')[0])
            assert v['dtype'] == 'F8_E4M3', (k, v['dtype'])
            o0, o1 = v['data_offsets']
            shards[idx] = (f, data_start + o0, o1 - o0, v['shape'], k)

assert scale is not None, 'weight_scale not found'
assert len(shards) == 128, f'found {len(shards)} shards'
rows, dim = shards[0][3]
print(f'scale={scale}, shards=128 x [{rows},{dim}] fp8', flush=True)

# ── LUT: fp8 e4m3fn byte -> bf16 bits of (value * scale) ──
def e4m3fn_to_f32(b):
    b = int(b)
    s = -1.0 if (b & 0x80) else 1.0
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0xF and m == 0x7:
        return float('nan')
    if e == 0:
        return s * (m / 8.0) * 2.0 ** -6
    return s * (1.0 + m / 8.0) * 2.0 ** (e - 7)

lut_f32 = np.array([e4m3fn_to_f32(b) for b in range(256)], dtype=np.float32) * scale
bits = lut_f32.view(np.uint32)
rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
LUT = rounded  # u8 -> bf16 bits

# ── write safetensors: header then streamed data ──
names = [shards[i][4] for i in range(128)]
per_bytes = rows * dim * 2
header = {}
off = 0
for i in range(128):
    header[names[i]] = {'dtype': 'BF16', 'shape': [rows, dim],
                        'data_offsets': [off, off + per_bytes]}
    off += per_bytes
hdr_json = json.dumps(header, separators=(',', ':')).encode()
pad = (8 - (len(hdr_json) % 8)) % 8
hdr_json += b' ' * pad

CHUNK = 256 * 1024 * 1024
t0 = time.time()
with open(OUT, 'wb') as out:
    out.write(struct.pack('<Q', len(hdr_json)))
    out.write(hdr_json)
    for i in range(128):
        path, aoff, nbytes, shape, name = shards[i]
        with open(path, 'rb') as fh:
            fh.seek(aoff)
            left = nbytes
            while left:
                n = min(CHUNK, left)
                raw = np.frombuffer(fh.read(n), dtype=np.uint8)
                out.write(LUT[raw].tobytes())
                left -= n
        if i % 16 == 0:
            el = time.time() - t0
            print(f'shard {i}/128  {el:.0f}s', flush=True)
print(f'DONE {time.time()-t0:.0f}s -> {OUT} ({os.path.getsize(OUT)/1e9:.1f} GB)', flush=True)

# ── spot-check 3 random rows against direct dequant ──
rng = np.random.default_rng(0)
with open(OUT, 'rb') as fh:
    n = struct.unpack('<Q', fh.read(8))[0]
    ohdr = json.loads(fh.read(n))
    odata = 8 + n
    for _ in range(3):
        si = int(rng.integers(0, 128)); ri = int(rng.integers(0, rows))
        path, aoff, _, _, name = shards[si]
        with open(path, 'rb') as sf:
            sf.seek(aoff + ri * dim)
            src = np.frombuffer(sf.read(dim), dtype=np.uint8)
        want = LUT[src]
        fh.seek(odata + ohdr[name]['data_offsets'][0] + ri * dim * 2)
        got = np.frombuffer(fh.read(dim * 2), dtype=np.uint16)
        assert np.array_equal(want, got), f'mismatch shard {si} row {ri}'
print('spot-check OK', flush=True)
