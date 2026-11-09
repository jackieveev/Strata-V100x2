# Unsloth's K-quants and Q8_0: formats, kernels and quality

This page covers Unsloth's files of the model ([unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
revision `38bb39e`; the repository's head `766911a` has the same files, byte for byte): what each file stores in
each layer, which of Strata's kernels run those formats on each kind of hardware, and how close each file comes to
the Q8_0 reference. Setup and the per-file details: [UD-Q4_K_XL](UNSLOTH_Q4.md), [UD-Q6_K_XL](UNSLOTH_Q6.md),
[MODELS.md](MODELS.md).

## What each file stores

The routed experts (gate, up, down; 512 per layer, 10 used per token) are where the files differ:

| File | Download | Routed gate / up | Routed down | Routed experts | Per token (10 experts x 48 layers) |
|---|---:|---|---|---:|---:|
| UD-IQ4_XS | 93.7 GB | IQ3_S (IQ4_XS in layer 2) | IQ4_NL (Q8_0 in layers 2, 4, 30, 46, 47) | 55.4 GiB | 1.16 GB |
| UD-Q4_K_XL | 111.3 GB | Q4_K (Q5_K in layer 2) | Q5_1 (Q8_0 in layers 2, 4, 30, 46, 47) | 71.7 GiB | 1.50 GB |
| UD-Q5_K_XL | 158.3 GB | Q5_K (Q6_K in layer 2) | Q8_0 | 91.6 GiB | 1.92 GB |
| UD-Q6_K_XL | 169.2 GB | Q6_K (Q8_0 in layer 2) | Q8_0 | 101.7 GiB | 2.13 GB |
| Q8_0 | 188.2 GB | Q8_0 | Q8_0 | 119.5 GiB | 2.51 GB |

Everything else is the same in all five: the attention, SSM, hyper-connection and shared-expert projections, the
PLE key/value and the token embedding are Q8_0 (the small vectors, norms and the router F32; the QSA indexer BF16),
the output head is Q8_0 (Q6_K in UD-IQ4_XS). The PLE table (`per_layer_token_embd`) is IQ4_NL (26.8 GiB) in UD-IQ4_XS
and UD-Q4_K_XL and Q8_0 (50.7 GiB) in UD-Q5_K_XL, UD-Q6_K_XL and Q8_0. The dense side the GPU reads once per verify
window is about 4.8 GB in every file.

UD-Q5_K_XL, UD-Q6_K_XL and Q8_0 share their shards 2 (the output head) and 3 (the PLE table) byte for byte (the same
SHA-256), so a PC that has one of them can link those two shards instead of downloading them again.

### Unsloth's Dynamic v3.1 sets (#95)

The same repository republished these seven files on 2026-11-04 (commit `d9620119`, "Dynamic v3.1"). They store the
routed experts below 4 bits with the i-quant and Q2_0 formats the 2-3-bit packs above use, and four of them put Q5_0
weights in the down projection of some layers:

| File | Download | Routed gate / up (layers) | Routed down (layers) | Routed experts | Per token (10 experts x 48 layers) |
|---|---:|---|---|---:|---:|
| UD-IQ1_M | 68.0 GB | IQ2_XXS 23, IQ2_S 18, IQ2_XS 7 | Q2_0 46, IQ4_NL 2 | 32.4 GiB | 0.68 GB |
| UD-Q2_0 | 70.5 GB | IQ2_XXS 18, IQ2_S 30 | Q2_0 39, IQ4_NL 9 | 34.8 GiB | 0.73 GB |
| UD-IQ2_XS | 75.8 GB | IQ2_XXS 8, IQ2_S 40 | IQ4_NL 24, Q2_0 23, Q5_0 1 | 39.4 GiB | 0.83 GB |
| UD-Q2_K_XL | 79.6 GB | IQ2_XXS 5, IQ2_S 26, IQ3_XXS 17 | IQ4_NL 31, Q2_0 16, Q5_0 1 | 42.8 GiB | 0.90 GB |
| UD-IQ3_XXS | 83.6 GB | IQ2_S 26, IQ3_XXS 21, IQ4_XS 1 | IQ4_NL 45, Q2_0 3 | 46.8 GiB | 0.98 GB |
| UD-Q3_K_XL | 89.5 GB | IQ2_S 4, IQ3_XXS 38, IQ4_XS 6 | IQ4_NL 26, Q5_0 20, Q2_0 2 | 52.3 GiB | 1.10 GB |
| UD-IQ4_XS (the v3.1 layout) | 93.7 GB | IQ2_S 3, IQ3_XXS 28, IQ4_XS 17 | Q5_0 33, IQ4_NL 13, Q2_0 2 | 56.2 GiB | 1.18 GB |

The dense side differs more than in the five files above: the attention is Q6_K (the qkv projection Q5_K in UD-IQ1_M
and UD-Q2_0, Q8_0 in UD-IQ2_XS and UD-Q2_K_XL), the shared expert is Q8_0 below 4-bit and Q6_K above it, the SSM is
Q8_0 (Q6_K in UD-IQ3_XXS), the token embedding is IQ4_XS (Q6_K in UD-IQ3_XXS), the output head Q6_K/Q8_0 and the
PLE table IQ4_NL (26.8 GiB) in every one of them, so the dense side a verify window reads is about 4.8 GB again.

The Q5_0 down needs engine 0.1.45.7 or newer: that build applies the format's -16 per-weight offset to the quantized
activation sum, as the Q5_1 and Q4_0 paths and ggml-cpu do, instead of to the sum of the original activations. Measured
with the synthetic expert parity (`native_expert_parity --synthetic`), the whole-expert error against a float reference
drops from 2.1-3.1e-2 to 1.2e-2, the same as every other down format, and the GPU then agrees with the CPU path.
Before that build the layers with Q5_0 downs answer, but 2-3% less accurately per expert. The other three v3.1 sets
(UD-IQ1_M, UD-Q2_0, UD-IQ3_XXS) have no Q5_0 and ran before it.

## Kernels per format and backend

| Format (role) | CUDA decode (grouped experts) | CUDA prompt | CPU pool | AMD (HIP) decode | AMD prompt | Intel (SYCL) decode | Intel prompt |
|---|---|---|---|---|---|---|---|
| Q4_K (gate/up) | yes | FP16 path; MMQ with `STRATA_MMQ_KQ=1` | ggml AVX2 dot; multi-token AVX2 (`STRATA_KQ256=1`) | yes (CUDA layout; `STRATA_EXPERT_V2K=1` bitwise) | FP16 path; MMQ with `STRATA_MMQ_KQ=1`; gfx11 fused (`STRATA_PF_FUSED_KQ=1`) | kernels built, not run with these files | FP16 path |
| Q5_K (gate/up) | yes | as Q4_K | ggml AVX2 dot | as Q4_K | as Q4_K | kernels built, not run with these files | FP16 path |
| Q6_K (gate/up) | yes (since this change; was `-DSTRATA_Q6K_EXPERTS=ON`) | FP16 path; MMQ with `STRATA_MMQ_KQ=1` | ggml AVX2 dot | yes (CUDA layout) | FP16 path; MMQ with `STRATA_MMQ_KQ=1` | kernels built, not run with these files | FP16 path |
| Q5_1 (down) | yes | as Q4_K | ggml AVX2 dot; multi-token AVX2 (`STRATA_KQ256=1`) | yes (CUDA layout; V2K) | as Q4_K | kernels built, not run with these files | FP16 path |
| Q8_0 (gate/up, down) | yes | MMQ | ggml AVX2 dot; multi-token AVX2 down (`STRATA_KQ256=1`) | yes (CUDA layout; V2K down) | MMQ | kernels built, not run with these files | FP16 path |

- "FP16 path": the prompt's experts dequantized to FP16 and multiplied by the BLAS library. It works for every format
  above but reads each expert as ~10 MB of FP16.
- **Formats cost nothing unless used.** The engine loads every kernel of its own binary when it starts (so that no
  kernel needs VRAM later, mid-prompt), which is why kernels for formats a model does not have used to take expert
  slots from every model. The Q6_K expert kernels are now part of every build: one more instance per grouped kernel,
  +19 KB of sm_60 code, and the expert-cache slot counts measured unchanged (IQ3_XXS: 6,028 slots on a P100, 15,878
  and 1,980 on an R9700). The K-quant MMQ prompt kernels (Q4_K, Q5_K, Q5_1, Q6_K: ~3.8 MB of sm_60 code, 3 IQ3_XXS
  slots on the P100 when they were in the engine) are on CUDA a library of their own next to the engine
  (`libstrata_kq.so` / `strata_kq.dll`), opened only for a model with these experts and `STRATA_MMQ_KQ=1`; without the
  file those experts take the FP16 prompt path. On AMD a code object loads on first use, so they stay in the engine.
- `STRATA_MMQ_KQ=1` is opt-in because MMQ rounds the prompt's products differently from the FP16 path (the output
  differs at the rounding level); its speed and KL are measured before it becomes a default. One exception: setup
  writes it for `UD-Q4_K_XL` on the CUDA 12 engine (V100, sm_70), because the FP16 prompt path returns non-finite
  logits (#879) from layer 0 for this file ([NVIDIA_V100.md](NVIDIA_V100.md)).

## Speed, and the bound

Measured on the AMD box: one Radeon AI PRO R9700 (32 GB, capped at 225 W), EPYC 7452 (32 cores, 8 channels of
DDR4-2400: 106 GB/s read by `rdbw` at 32 threads), every expert in RAM (setup's rule below), `--expert-cache auto`,
40K context, 8-bit K/V, MTP drafts; the second of two requests; prompt speed varies between runs on
this box by up to 2x (the range is given).

| File | Short story (tok/s) | Code (tok/s) | After a 6K document | Prompt 32K (tok/s) | llama.cpp 8bb3cf0d2 (same card, `-ngl 99 --n-cpu-moe 38-48`): decode / prompt 4K |
|---|---:|---:|---:|---:|---:|
| IQ3_S (for comparison) | 75 | 95 | 77 | 1,555 | - |
| UD-Q4_K_XL | 63-64 | 85-88 | 70-72 | 1,110-1,320 | 18-20 / 198-257 |
| UD-Q5_K_XL | 57-60 | 81-83 | 62 | 810-1,260 | 17-18 / 226-270 |
| UD-Q6_K_XL | 54 | 76 | 61 | 1,020 | - |
| Q8_0 | 53-57 | 66-74 | 54-55 | 1,230-1,300 | 16-18 / 186-225 |

**Where a decode window goes** (`STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1`, UD-Q4_K_XL, the short story, 2.0
tokens per window, 34.3 ms per window): the bandwidth bound of what a window must read is ~12.5 ms - the routed
experts in VRAM (~2.7 GB at 644 GB/s: 4.2 ms bound, 6.0 measured), the dense Q8_0 projections (2.84 GB: 4.4 ms bound,
7.2 measured), the hyper-connections (0.67 GB: 1.0 ms bound, 4.0 measured), router and shared expert (0.5 GB: 0.8 ms
bound, 3.7 measured), the head (1.05 bound, 1.26 measured), the CPU pool's experts at RAM bandwidth in parallel
(1.2 ms), plus the MTP draft (5.8 ms measured). Decode runs at ~35-40% of that bound for every file, IQ3_S included:
the formats' own kernels are near their bandwidth (VRAM-resident experts ~70%, the CPU pool 84-97% of RAM bandwidth,
the head 83%); the rest is the per-window latency of the small kernels and the draft, the same for every model.

**Dense Q8_0 GEMVs** (the engine's shapes, 2-4 columns): one row per block where it was measured
faster, the same bits as two (R9700: 2560 -> 10240 / 6144 / 12288 at 76-86% of 644 GB/s instead of 51-80%; RTX A4000:
+1-4%, and 6144 -> 2560 at 6 columns 70% -> 76%). UD-Q4_K_XL decode +1.0-2.7% on the R9700, the same text.

**NVIDIA with the RAM budget** (RTX A4000 16 GB, Xeon E5-2643 v3, 62 GB: UD-Q4_K_XL with 39 GiB of its experts in RAM,
the rest read from an NVMe SSD at 2.1 GB/s): 14-20 tokens/s and a 4K prompt at 134-144 tokens/s, bound by the SSD;
llama.cpp on the same PC and file 14.8-16.5 / 117-123. `STRATA_PROMPT_DIRECT=1` (Linux, opt-in) reads the prompt
path's experts unbuffered while decode keeps the file cache: the same output (6 of 6 prompts identical with fixed
flags), the 4K prompt 152 -> 315 tokens/s, a 20K prompt 217 -> 500, decode +2-15% (5 interleaved pairs, page cache
dropped before each run). On a second PC with more file-cache room (the R9700 box held to 64 GB) it was mixed: a 6K
document's prompt +8%, a 4K prompt -15% and the answer after the document -38%, because the buffered reads had left
the document's experts in the file cache. So it stays opt-in. Reading everything unbuffered (`STRATA_UNBUFFERED_LOAD=1`) doubled the prompt too but cost
decode 9-13%; a larger or smaller RAM budget, more fetch threads, a deeper read-ahead or a wider look-ahead did not
help.

**Several cards** (R9700 box, every expert in RAM, layer split, 2 runs each): UD-Q4_K_XL on 1 / 2 / 3 / 4 cards writes
63 / 64 / 55 / 52 tokens/s on the short story and 88 / 96 / 123 / 96 on code, and reads a 32K prompt at 1,110 /
1,275 / 1,987 / 2,381 tokens/s; UD-Q5_K_XL 57 / 60 / 62 / 46 and 83 / 94 / 101 / 89, prompt 1,264 / 1,177 / 1,767 /
2,067. llama.cpp on 3 / 4 cards: 24.5 / 28.2 tokens/s and 497 / 800 (4K prompt) for UD-Q4_K_XL. More cards read
prompts faster; for short answers the split's per-stage cost outweighs the experts it keeps in VRAM, since one card with
every expert in RAM is already fast (`--pipeline-windows 0` helps short answers at 3-4 cards, x1.3, and costs code
6-20%).

**Every expert in RAM against the old RAM budget** (setup's new rule, R9700 box, UD-Q4_K_XL, 5 interleaved pairs, the
old config's `--resident-budget-gib 71` holding the same experts read from the GGUF in place): decode x1.34-1.40 (short
story 39.5 -> 53.3 tok/s, code 60.1 -> 80.4, after a 6K document 49.0 -> 68.8, 5/5 each), prompt 4K x2.07, 6K x1.58,
32K x1.03.

**The route tail skip** (`STRATA_ROUTE_TAIL_SKIP=7`, the default since 0.1.42: a missed expert every token of a window
routes at rank 7 or lower is skipped) costs these files more than the 2-3-bit packs: UD-Q4_K_XL's KL against
Q8_0 goes from 0.056 to 0.086 with every expert in RAM (R9700, the auto cache) and from 0.053 to 0.115 in the RAM
budget mode (A4000, 1,500 GPU slots), for +4-10% decode in RAM and +26-67% in the budget mode (5 pairs each).
`STRATA_ROUTE_TAIL_SKIP=0` in a config's `"env"` turns it off.

**The hyper-connections on NVIDIA:** the pack's `--compat-bf16` copy of the Q8_0 HC projections rounds them to BF16
(and doubles their bytes). `STRATA_HC_Q8=1` reads the file's own Q8_0 instead (on AMD gfx12 it is the default): on the
A4000 UD-Q4_K_XL then comes closer to Q8_0 (KL 0.0562 -> 0.0526, p99 0.925 -> 0.716) and to the same file on
AMD (KL 0.038 -> 0.033). Opt-in until decided.

## Images

Every Unsloth file reads pictures with the model's own encoder: setup's `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (from
ISTA-DASLab's repository) has the same 334 tensors, byte for byte, as Unsloth's `mmproj-BF16.gguf`. Run end to end
with a test picture (a line of text and three coloured shapes), two questions through `/v1/chat/completions`: on an RTX
A4000 with the GPU encoder (UD-Q4_K_XL, IQ3_XXS) and on an R9700 with the CPU encoder (UD-Q4_K_XL, UD-Q5_K_XL,
IQ3_XXS), every answer had the text and the three shapes right, as llama.cpp's `mtmd-cli` did on the same GGUF and
mmproj. The GPU encoder keeps ~1.3 GB of VRAM free (UD-Q4_K_XL on 16 GB: 2,331 expert slots instead of 2,741); the CPU
encoder takes none. Setup turns images on by default for these files on an NVIDIA card of 16 GB or more, and on AMD
(Linux, the CPU encoder) where every expert stays in RAM.
