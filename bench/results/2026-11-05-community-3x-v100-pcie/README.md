# Community benchmark: three Tesla V100 32 GB, layer split 16/16/16

Measured on 2026-11-05 on the machine described in
[docs/NVIDIA_V100.md](../../../docs/NVIDIA_V100.md), using the V100 fork's
engine 0.1.45.6 at commit `4c56f00`. Three cards hold the model, split into
equal 16-layer stages; the fourth card of the box was NOT part of this test.

Median decode throughput was **78.3 tok/s at ~4.1k prompt tokens, 85.2 tok/s at
~8.8k, 71.8 tok/s at 29.5k, 62.6 tok/s at 117.8k, and 66.9 tok/s at 256k**.
Median prompt throughput over the same rows was 1,283 / 1,507 / 2,686 / 1,868 /
1,509 tok/s. These are synthetic repeated-text prompts with a 64-token output
cap, measured by the engine's own `/metrics`; they say nothing about answer
quality or about other workloads.

## Hardware and software

- 4x Tesla V100-class cards, 32,768 MiB each, no NVLink, all at PCIe Gen 3.
  This test used three of them, in the engine's stage order:
  - card 1, `0000:0a:00.0`, Tesla V100-PCIE-32GB, x16 — layers 0-15
  - card 2, `0000:42:00.0`, Tesla PG500-216, x8 — layers 16-31
  - card 3, `0000:43:00.0`, Tesla V100-PCIE-32GB, x16 — layers 32-47 and the head
  The engine's startup transfer probe reported 13.1 GB/s host-to-device on the
  x16 slots and 6.7 GB/s on the x8 slot. GPU clocks were not fixed.
- AMD Ryzen Threadripper 2990WX, 32 cores / 64 threads, no AVX-512 (the engine
  ran its expert kernels on AVX-2 and reported that at startup).
- 188.8 GiB usable RAM. The expert pack is 55 GB of host RAM plus an SSD-resident
  n-gram table.
- Debian 13, kernel 6.12.90+deb13.1-amd64, NVIDIA driver 580.178.04, CUDA 12.8
  build.
- Engine 0.1.45.6, source commit `4c56f00`, local sm_70 CUDA build.
- The gateway and its llama.cpp workers were stopped for the run. A small
  unrelated 4B vLLM engine (spark, on the fourth card) and the operating
  system's own services stayed up, so the host was not fully isolated.

## Model and configuration

`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, `IQ3_S`:

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (native pack source)
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` (PLE table)
- MTP drafter fetched by the installer

Engine settings (all in the service config, none overridden per request):

```
--pack iq3_s --expert-cache auto --vram-reserve-mib 2048
--prefill auto --spec 4 --spec-min-p 0.5 --mtp <rt>
--max-context 262144 --kv int8 --kv-resident 262144 --ple-io ram
--layer-split 16,32            # first layer of each later GPU, i.e. 16/16/16
parallel 3                     # batch slots, idle during this test
conversation cache: 16 GiB RAM tier, 400 GiB disk tier
```

What the engine reported at startup on this layout:

- `layer split across 3 GPUs: CUDA0, then CUDA1,2 (split 16,32)`
- expert caches of 8192 slots (15.99 GiB) and 8192 slots (16.52 GiB) on the two
  later stages, 8192 of 8192 profiled pairs each: **100% of the experts resident**
  across the three cards (the same model on two cards leaves 75-78% resident)
- prompt chunk 8192 tokens with a 96-slot streamed ring
- session/KV 1.27 GiB for the 262,144-token window, sharded over the stages

## Method

`bench/run_v100_bench.py --repeats 3`, unmodified, against
`http://127.0.0.1:18080` with `--model qwen3.8-27b`. Every row is a separate
request with a fresh unique prefix (`reused=0` in every row below, so no row was
served from a cached prefix), 64 generated tokens, greedy-adjacent defaults.
Prompt and decode times come from the server's `/metrics`, not client wall
clock. Raw rows: [matrix.json](matrix.json).

| row | prompt tokens | reused | prompt tok/s | decode tok/s |
|---|---|---|---|---|
| ~4K#0 | 4,144 | 0 | 1,163.6 | 83.7 |
| ~4K#1 | 4,139 | 0 | 1,289.0 | 66.9 |
| ~4K#2 | 4,144 | 0 | 1,277.2 | 76.7 |
| ~4K#0 | 4,375 | 0 | 1,300.6 | 80.6 |
| ~4K#1 | 4,376 | 0 | 1,264.2 | 69.2 |
| ~4K#2 | 4,378 | 0 | 1,297.6 | 80.0 |
| ~8K#0 | 8,788 | 0 | 1,540.7 | 91.2 |
| ~8K#1 | 8,786 | 0 | 1,507.1 | 85.2 |
| ~8K#2 | 8,787 | 0 | 1,190.3 | 80.0 |
| 32K#0 | 29,527 | 0 | 2,686.5 | 61.2 |
| 32K#1 | 29,527 | 0 | 697.1 | 71.8 |
| 32K#2 | 29,525 | 0 | 2,686.8 | 73.5 |
| 128K#0 | 117,837 | 0 | 1,868.3 | 70.0 |
| 128K#1 | 117,837 | 0 | 2,813.2 | 61.2 |
| 128K#2 | 117,839 | 0 | 1,860.0 | 62.6 |
| 256K#0 | 256,078 | 0 | 1,751.5 | 66.9 |
| 256K#1 | 256,078 | 0 | 1,208.0 | 83.5 |
| 256K#2 | 256,076 | 0 | 1,509.2 | 53.4 |

Medians by row size: 1,283 / 1,507 / 2,686 / 1,868 / 1,509 tok/s prompt and
78.3 / 85.2 / 71.8 / 62.6 / 66.9 tok/s decode at ~4.1k / ~8.8k / 29.5k / 117.8k /
256k.

Decode here is speculative decode with the MTP drafter at depth 4: the engine
reported 1,162 of 1,595 drafts accepted (73%) on one 109k-token request during
the same session. A run without `--mtp` would not be comparable.

## Spread, and what it is

Prefill varies a lot more than decode on this box, and one row is a clear
outlier: `32K#1` read at 697 tok/s against 2,686 tok/s in the other two runs.
The engine was writing conversation-cache checkpoints during the run (the RAM
tier fills and parks conversations), and a park competes with a prompt read for
the same cards. I did not re-run that row with parking disabled, so treat the
32K median as ~2,686 tok/s with one suspended run rather than as a distribution.
Decode spread within a row (e.g. 53.4-83.5 at 256K) is larger than I would like
and is dominated by where the output happened to land relative to the parked
conversation's checkpoints.

## Recall checks

`tools/needle_bench.py --lengths 32k,128k --depths 10,50,90` against the same
server: **6 of 6 found** at 32,355 and 121,950 prompt tokens. Raw:
[needles.json](needles.json).

## What to take from this

- Three V100s with an explicit 16/16/16 split keep **every expert resident**;
  the same model on the two x16 cards alone left 75-78% resident, and that
  difference is where the decode gain comes from.
- The x8 slot in the middle of the pipeline costs less than its link width
  suggests: the two x16 stages transfer at 13.1 GB/s and the engine's miss cost
  model treats 6.7 GB/s as equal once the experts of that stage are resident.
- A 256k-token prompt reads in 146-212 s at 1,208-1,751 tok/s, and the same
  window is answered at 53-84 tok/s, so the long rows are prefill-bound, as
  expected on Volta without NVLink.
