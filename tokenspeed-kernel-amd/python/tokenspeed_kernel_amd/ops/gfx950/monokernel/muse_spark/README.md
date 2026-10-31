# Muse Spark decode monokernels

These kernels run one decoder layer of Muse Spark Open (`muse_spark`) as one
persistent launch per rank of a TP8 group (attention TP8, MoE TP8 / EP1), for
one token row. They use the building blocks in [`../README.md`](../README.md).
That file also gives the launch contract and the rules for writing a stage.

## Kernels

| Kernel | Layer |
|---|---|
| `gluon_muse_spark_layer_gfx950` | A decoder layer with MXFP4 routed experts: sliding-window (RoPE) or full (NoPE) attention, by the `full` flag. |

`layer.py` holds the kernel, its plans (`layer_plan(full)`) and its launcher
`launch_gluon_muse_spark_layer_gfx950`. `tokenspeed_kernel/ops/monokernel/`
registers them as `prepare_muse_spark_layer` and `muse_spark_layer_decode`.

### Contract

- Rank `r` owns hidden columns `1024 r ..`. The launch reads the FP32
  residual of the owned columns and the whole BF16 attention input of the
  layer. It writes the owned FP32 residual after the layer and the whole
  BF16 attention input of the next layer, normed with that layer's input
  gain.
- The regular path hands over the whole residual row. Thus the first launch
  of a run of fused layers reads its slice of that row, and after the last
  one the caller gathers the 8 slices back into a row (32 KiB).
- The launch reads and writes the regular path's paged KV cache in place:
  BF16 `[slots, 2, 64]` per rank, 64 tokens a page, the page table indexed
  by absolute position on sliding layers too. A sliding layer attends over
  the last 2048 positions.
- The step counter, the sequence length, the write slot and the page table
  are device tensors. The layer index is a runtime argument. Thus one binary
  serves every layer of a kind, and the decode graph captures the launches.
- The mailbox tag is `step * 64 + layer + 1`. The peer regions' copy is
  `layer & 1`: it alternates from one launch to the next, across steps too,
  since layers 1 to 60 are an even count.
- The launcher's docstring lists every operand.

### Serving

`python/tokenspeed/runtime/models/muse_spark_monokernel.py` is the runtime
seam. The switch `--decode-monokernel` enables it.

- Layers 1 to 60 are eligible. Layers 0 and 61 hand over to the embedding
  and the LM head, and keep the regular path.
- A model is eligible at attention TP8 and MoE TP8 / EP1, with Muse Spark
  Open's dimensions and the constants the kernel bakes in (`KERNEL_CONSTANTS`),
  BF16 dense projections and norms, and routed experts on a gluon MXFP4 MoE
  kernel. The seam decides this once, at load, and logs every other layer
  with its reasons. Startup fails when no layer is eligible.
- The seam allocates the mailbox scratch on the Iris symmetric heap before
  graph capture.
- A forward uses the kernels when it is a decode step of one request with
  one token. Other forwards use the regular path.

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
| `moe_experts.py` | ex1, ex2 | The MXFP4 experts, each rank's 512 of 4096 intermediate values. |
| `moe_tail.py` | comb1, comb2, unpool, fin, out | Combine the routes, the post-expert projection, the post-FFN norm, residual gate and the next layer's input norm. |

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
| `test_muse_spark_layer_gfx950.py` | The whole layer through the packer and the launcher, no recompiles across layers, steps and lengths, and the packer's errors. |

Each stage test runs a slice of the real plan and compares it with a torch
reference of TokenSpeed's math (`muse_spark_reference.py`).
`test/runtime/test_muse_spark_monokernel_seam.py` covers which forwards and
models the seam serves.
