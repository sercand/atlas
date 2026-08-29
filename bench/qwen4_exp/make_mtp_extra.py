#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Build extra_weights.safetensors for the qwen4_exp MTP head.

- Slices the fused BF16 MTP experts (gate_up_proj [512,1280,2560],
  down_proj [512,2560,640]) into per-expert gate/up/down and quantizes each
  to ModelOpt-style NVFP4 (packed U8 + per-16 E4M3 weight_scale + F32
  weight_scale_2 + F32 input_scale=1.0), replicating
  kernels/gb10/common/quantize_bf16_to_nvfp4.cu bit-for-bit in the scale and
  E2M1 rounding math so the result matches what Atlas's runtime requantizer
  would produce.
- Copies every other mtp.* tensor through verbatim (BF16).

Output: <snapshot>/extra_weights.safetensors (loaded by the extra-weights
hook in spark-runtime's weight loader, bypassing skip_mtp).
"""
import json, struct, glob, os, sys, time
import numpy as np

SNAP_SRC = os.environ.get('QWEN4EXP_SRC_SNAPSHOT') or glob.glob(os.path.expanduser(
    '~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/*/'))[0]
OUT = os.environ.get('QWEN4EXP_EXTRA_OUT', '/home/otsimo/work/qwen4exp-serve/snapshot/extra_weights.safetensors')

def read_header(path):
    with open(path, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n

# ── locate all mtp.* tensors ──
#
# Two releases spell the BF16 backbone files differently — RadixArk
# `model-bf16-*.safetensors`, primitive-ai's mixed build
# `carry-model-bf16-*.safetensors` — and the mtp.* tensors are BF16 and
# byte-identical in both. Scan every shard EXCEPT the n-gram table (102 GB of
# `ple-bf16-*` whose headers we have no reason to read) and pick by name.
SHARD_GLOB = os.environ.get('QWEN4EXP_SHARD_GLOB', '*model*bf16-*.safetensors')
loc = {}
for f in sorted(glob.glob(SNAP_SRC + SHARD_GLOB)):
    if os.path.basename(f).startswith('ple-'):
        continue
    hdr, ds = read_header(f)
    for k, v in hdr.items():
        if k.startswith('mtp.'):
            loc[k] = (f, ds + v['data_offsets'][0],
                      v['data_offsets'][1] - v['data_offsets'][0], v['dtype'], v['shape'])
assert len(loc) == 31, f'{len(loc)} mtp tensors found'

def load_bf16(name):
    f, off, nbytes, dt, shape = loc[name]
    assert dt == 'BF16', (name, dt)
    with open(f, 'rb') as fh:
        fh.seek(off)
        raw = np.frombuffer(fh.read(nbytes), dtype=np.uint16)
    return raw.reshape(shape)

def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def f32_to_e4m3(v):
    """Vectorized replica of float_to_fp8_e4m3 (v >= 0 here)."""
    v = np.minimum(v.astype(np.float32), 448.0)
    out = np.zeros(v.shape, dtype=np.uint8)
    bits = v.view(np.uint32)
    f32_exp = ((bits >> 23) & 0xFF).astype(np.int32) - 127
    man = bits & 0x7FFFFF
    nonzero = v > 0
    sub = nonzero & (f32_exp >= -9) & (f32_exp < -6)
    if sub.any():
        m = np.minimum((v[sub] * 512.0 + 0.5).astype(np.int32), 7)
        out[sub] = m.astype(np.uint8)
    norm = nonzero & (f32_exp >= -6)
    if norm.any():
        e = np.maximum(f32_exp[norm] + 7, 1)
        m3 = ((man[norm] + (1 << 19)) >> 20).astype(np.int32)
        carry = m3 > 7
        m3 = np.where(carry, 0, m3)
        e = e + carry.astype(np.int32)
        over = e > 15
        e = np.where(over, 15, e)
        m3 = np.where(over, 6, m3)
        out[norm] = ((e.astype(np.uint32) << 3) | m3.astype(np.uint32)).astype(np.uint8)
    return out

def e4m3_decode(b):
    b = b.astype(np.uint32)
    exp = (b >> 3) & 0xF
    man = b & 0x7
    dec = np.where(exp == 0, man.astype(np.float32) * 0.001953125,
                   (((exp + 120) << 23) | (man << 20)).view(np.float32))
    # NaN pattern exp=15,man=7 -> 0 (kernel behavior)
    dec = np.where((exp == 15) & (man == 7), 0.0, dec)
    return dec.astype(np.float32)

E2M1_THRESH = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)

def quantize_nvfp4(w_u16):
    """w: [N,K] BF16 bits -> (packed U8 [N,K/2], scales U8 [N,K/16], scale2 F32)."""
    n, k = w_u16.shape
    w = bf16_to_f32(w_u16).reshape(n, k // 16, 16)
    absw = np.abs(w)
    gmax = absw.max(axis=2)                       # [N, K/16]
    global_max = float(absw.max())
    scale2 = global_max / (6.0 * 448.0)
    inv_scale2 = 1.0 / scale2 if scale2 > 0 else 0.0
    fp8_float = np.where(gmax > 0, gmax * inv_scale2 / 6.0, 0.0).astype(np.float32)
    fp8 = f32_to_e4m3(fp8_float)                  # [N, K/16] u8
    eff = e4m3_decode(fp8) * scale2               # effective scale
    inv_eff = np.where(eff > 0, 1.0 / eff, 0.0).astype(np.float32)
    q = w * inv_eff[:, :, None]                   # [N, K/16, 16]
    sign = (q < 0).astype(np.uint8) << 3
    idx = np.searchsorted(E2M1_THRESH, np.abs(q).ravel(), side='left').astype(np.uint8)
    nib = (sign.ravel() | idx).reshape(n, k)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)   # low = even
    return packed, fp8, np.float32(scale2)

# ── build tensor dict (name -> (dtype, shape, bytes)) ──
tensors = {}
FUSED = {'mtp.layers.0.mlp.experts.gate_up_proj', 'mtp.layers.0.mlp.experts.down_proj'}
for name in sorted(loc):
    if name in FUSED:
        continue
    f, off, nbytes, dt, shape = loc[name]
    with open(f, 'rb') as fh:
        fh.seek(off)
        tensors[name] = ('BF16', shape, fh.read(nbytes))

gate_up = load_bf16('mtp.layers.0.mlp.experts.gate_up_proj')   # [512,1280,2560]
down = load_bf16('mtp.layers.0.mlp.experts.down_proj')         # [512,2560,640]
E, GU, H = gate_up.shape
I = GU // 2
t0 = time.time()
one = np.float32(1.0).tobytes()
for e in range(E):
    for proj, w in (('gate_proj', gate_up[e, :I]), ('up_proj', gate_up[e, I:]),
                    ('down_proj', down[e])):
        packed, scales, s2 = quantize_nvfp4(np.ascontiguousarray(w))
        base = f'mtp.layers.0.mlp.experts.{e}.{proj}'
        tensors[f'{base}.weight'] = ('U8', list(packed.shape), packed.tobytes())
        tensors[f'{base}.weight_scale'] = ('F8_E4M3', list(scales.shape), scales.tobytes())
        tensors[f'{base}.weight_scale_2'] = ('F32', [], s2.tobytes())
        tensors[f'{base}.input_scale'] = ('F32', [], one)
    if e % 64 == 0:
        print(f'expert {e}/{E} {time.time()-t0:.0f}s', flush=True)

# ── write safetensors ──
header, off = {}, 0
order = sorted(tensors)
for name in order:
    dt, shape, data = tensors[name]
    header[name] = {'dtype': dt, 'shape': shape, 'data_offsets': [off, off + len(data)]}
    off += len(data)
hj = json.dumps(header, separators=(',', ':')).encode()
hj += b' ' * ((8 - len(hj) % 8) % 8)
with open(OUT, 'wb') as out:
    out.write(struct.pack('<Q', len(hj)))
    out.write(hj)
    for name in order:
        out.write(tensors[name][2])
print(f'DONE {time.time()-t0:.0f}s -> {OUT} ({os.path.getsize(OUT)/1e9:.2f} GB, {len(order)} tensors)', flush=True)

# spot check: dequant one expert row and compare to source
e, proj = 3, 'gate_proj'
w = bf16_to_f32(np.ascontiguousarray(gate_up[e, :I]))
packed, scales, s2 = quantize_nvfp4(np.ascontiguousarray(gate_up[e, :I]))
LUT = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
nib = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
nib[:, 0::2] = packed & 0xF
nib[:, 1::2] = packed >> 4
deq = LUT[nib].reshape(w.shape[0], -1, 16) * e4m3_decode(scales)[:, :, None] * s2
deq = deq.reshape(w.shape)
err = np.abs(deq - w)
rel = err.max() / (np.abs(w).max() + 1e-9)
print(f'spot dequant: max_abs_err={err.max():.5f} rel_to_absmax={rel:.4f}', flush=True)
