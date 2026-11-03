# Muse Spark decode monokernels

These kernels run one decoder layer of Muse Spark Open (`muse_spark`) as one
persistent launch per rank of a TP8 group (attention TP8, MoE TP8 / EP1), for
the rows of one request: one decode row, or the 2 to 8 consecutive tokens of
a speculative verify (`Plan.chain`). On a TP8 / EP8 server (MoE TP1 / EP8) they run on gathered
expert slices (see [Weights](#weights)). They use the building blocks in [`../README.md`](../README.md).
That file also gives the launch contract and the rules for writing a stage.

## Kernels

| Kernel | Layer |
|---|---|
| `gluon_muse_spark_layer_gfx950` | A decoder layer with MXFP4 routed experts: sliding-window (RoPE) or full (NoPE) attention, by the `full` flag. |

`layer.py` holds the kernel, its plans (`layer_plan(full)`) and its launcher
`launch_gluon_muse_spark_layer_gfx950`. `tokenspeed_kernel/ops/monokernel/`
registers them as `prepare_muse_spark_layer` and `muse_spark_layer_decode`.

### Contract

- Rank `r` owns hidden columns `1024 r ..`. The launch reads each row's
  FP32 residual of the owned columns (`[rows, 1024]`) and whole BF16
  attention input of the layer (`[rows, 8192]`). It writes each row's owned
  FP32 residual after the layer and whole BF16 attention input of the next
  layer, normed with that layer's input gain.
- Row `t` is the token at position `seq_len[0] - 1 + t`. `layer_plan(full,
  rows)` builds the plan of a row count; each row count compiles once.
- The regular path hands over the whole residual row. Thus the first launch
  of a run of fused layers reads its slice of that row, and after the last
  one the caller gathers the 8 slices back into a row (32 KiB).
- The launch reads and writes the regular path's paged KV cache in place:
  BF16 `[slots, 2, 64]` per rank, 64 tokens a page, the page table indexed
  by absolute position on sliding layers too. A sliding layer attends over
  the last 2048 positions. Each row writes its token's slot; the rows read
  the launch's own keys from the projection, not from the cache.
- The step counter, the sequence lengths, the write slots and the page
  table are device tensors. The layer index is a runtime argument. Thus one binary
  serves every layer of a kind, and the decode graph captures the launches.
- The mailbox tag is `step * 64 + layer + 1`. The peer regions' copy is
  `layer & 1`: it alternates from one launch to the next, across steps too,
  since layers 1 to 60 are an even count. A rank one launch ahead thus puts
  only into the copy no slower rank still reads. That needs one region
  layout for every launch in the scratch: the sliding and full plans size
  the attention regions for the full layer's splits, so they lay the scratch
  out alike (a layout per kind let a sliding and a full launch overlap, and
  the group deadlocked under serving).
- The launcher's docstring lists every operand.

### Serving

`python/tokenspeed/runtime/models/muse_spark_monokernel.py` is the runtime
seam. The switch `--decode-monokernel` enables it.

- Layers 1 to 60 are eligible. Layers 0 and 61 hand over to the embedding
  and the LM head, and keep the regular path.
- A model is eligible at attention TP8 and MoE TP8 / EP1 or TP1 / EP8, with Muse Spark
  Open's dimensions and the constants the kernel bakes in (`KERNEL_CONSTANTS`),
  BF16 dense projections and norms, and routed experts on a gluon MXFP4 MoE
  kernel. The seam decides this once, at load, and logs every other layer
  with its reasons. Startup fails when no layer is eligible.
- The seam allocates the mailbox scratch on the Iris symmetric heap before
  graph capture.
- A launch traps when its peers' values do not come within a few seconds,
  and a cold compile can take longer on one rank than on the others. Thus
  the first launch of each layer and row count only compiles
  (`compile_only`), the ranks meet at a CPU barrier, and then they launch.
- A forward uses the kernels when it is a decode step of one request: one
  token, or a speculative verify of up to 8 tokens (row `t` has length
  `seq - rows + t + 1` for the step's `seq_lens`). Other forwards use the
  regular path.
- A draft model that taps the residual entering a layer (DFlash2's
  `target_layer_ids`) gets whole rows there: the fused run before that layer
  gathers the 8 owned slices, as before a regular layer.

## Stages

One launch has 256 CTAs of 8 warps. The stages, in plan order (each module's
docstring gives its math and task split):

| Module | Stages | Work |
|---|---|---|
| `qkvg.py` | qkvg, gate | The fused q / k / v / output-gate projection, in two K halves. |
| `attention.py` | head, attn, merge | k norm, RoPE and the KV write; split attention with an online softmax; the LSE merge and the gated head norm. |
| `attn_tail.py` | oproj, tail1, tail2 | `o_proj` reduce-scattered to the owners; the post-attention norm, residual gate and pre-FFN norm. |
| `moe_route.py` | router, topk | Partial router logits of the owned columns; the sum, sigmoid scores and top 8. |
| `moe_pool.py` | pool | The pre-expert projection, gathered on every rank. |
| `moe_experts.py` | ex1, ex2 | The MXFP4 experts, each rank's 512 of 4096 intermediate values. The rows' distinct experts are slots: each streams once for every row that picked it. |
| `moe_tail.py` | comb1, comb2, unpool, fin, out | Combine the routes, the post-expert projection, the post-FFN norm, residual gate and the next layer's input norm. |
| `draft_mlp.py` | scat, sum1, norm, coef, conv0, gu, act, down, fin | The MLP half of a DFlash2 draft layer (Muse Spark's draft, TP8): the ranks' sum of the attention partials, the residual add and post-attention norm, the convolution coefficients, the convolutions around the MLP, gate/up, `silu(g) u` and down, ending in this rank's partial. Not served yet. |

## Weights

`weights.py` packs each layer once, after the weights load
(`pack_layer` -> `LayerWeights`).

- The BF16 QKVG, `o_proj`, pooler and post-expert weights are copied once
  into the GEMV's tile order (`bf16_tiles`). The pooler and post-expert
  weights are replicated on every rank; a rank copies only the rows it
  computes. The copies add about 4.3 GB per GPU for 60 layers. The regular
  path keeps its weights for prefill.
- The routed MXFP4 experts are read in place, in the layout of the regular
  path's gluon MoE (`preprocess_gluon_mxfp4_gfx950_moe_weights`).
- On an EP8 server a rank holds 32 whole experts, and the kernel reads the
  rank's 1/8 slice of all 256. While the weights load, before the gluon MoE
  prepares them, the seam (`prepare_ep8_experts`) sends each rank its slices
  of the local experts (one all-to-all per tensor), and `prepare_experts`
  prepares them with the same gluon preprocessing as an EP1 server's. The
  copy is about 0.8 GB per layer and rank, 48 GB for 60 layers, taken from
  the KV cache. The EP8 weights stay for the regular path. A kernel that runs
  whole experts on their owners would need no copy, but one token's 8 picks
  then land on the ranks unevenly, and the busiest rank sets the layer's
  time.
- The router takes the owned columns of the FP32 router weight. The norm
  gains are the FP32 `bf16(w + offset)`. The residual gate coefficients stay
  views of the regular path's tensors.
- `pack_layer` checks every tensor's shape and dtype and reports all
  mismatches in one error. `LayerWeights.kernel_args()` gives the launch's
  operand tuples in order.

## Files

| File | Contents |
|---|---|
| `layer.py` | The plans, the layer kernel and its launcher. |
| `dims.py` | The dimensions that several modules use. |
| `qkvg.py`, `attention.py`, `attn_tail.py` | The attention stages. |
| `moe_route.py`, `moe_pool.py`, `moe_experts.py`, `moe_tail.py` | The MoE stages and the layer boundary. |
| `draft_mlp.py` | The DFlash2 draft layer's MLP half. |
| `weights.py` | Host packing of the layer weights. |

## Tests

Tests are in `tokenspeed-kernel/test/amd/gfx950/ops/monokernel/`. All but the
QKVG and attention tests need 8 GPUs.

| Test | Covers |
|---|---|
| `test_muse_spark_qkvg_gfx950.py` | The QKVG and gate stages. |
| `test_muse_spark_attention_gfx950.py` | The attention stages, sliding and full. |
| `test_muse_spark_attn_tail_gfx950.py` | The output projection and attention tail. |
| `test_muse_spark_route_gfx950.py` | The router and top 8. |
| `test_muse_spark_pool_gfx950.py` | The pooler. |
| `test_muse_spark_experts_gfx950.py` | The experts, on the regular path's preprocessed weights. |
| `test_muse_spark_moe_tail_gfx950.py` | The combine, post-expert projection and boundary. |
| `test_muse_spark_draft_mlp_gfx950.py` | The DFlash2 draft's MLP half in one launch across the 8 ranks, each stage on the kernel's own input against `dflash2_draft_reference.py` (the regular path's rounding points). |
| `test_muse_spark_layer_gfx950.py` | The whole layer through the packer and the launcher, a compile-only call that runs nothing, no recompiles across layers, steps and lengths, CUDA graph replays against eager launches, and the packer's errors. |

Each stage test runs a slice of the real plan, at 1 row and at 8 chained
rows, and compares it with a torch reference of TokenSpeed's math
(`muse_spark_reference.py`).
`test/runtime/test_muse_spark_monokernel_seam.py` covers which forwards and
models the seam serves, and that an EP8 server's expert slices are the shards
of an EP1 server's loader.
