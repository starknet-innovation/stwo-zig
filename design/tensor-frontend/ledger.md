# Witness ledger: committed bits per token per layer

> **Port status (issue #438).** Ported with the three witness-volume levers
> into `src/frontends/int_inference` (protocol 0.2.0). In the package the
> ledger is driven by setting `ledger.sink` around a prove call (or
> `protocol.witnessLedger`); the `TENSOR_LEDGER` bench switches and
> `ledger_project.py` below are archived with `tensor-bench` and are not
> ported. Measurements are historical (November 2026, partly on battery).

The tensor frontend's equivalent of the per-opcode bits/gas table of the RV64
guest-efficiency work. The serving projection (serving.md section 4) found
that the Maverick gap is the **volume of committed witness**
(n_w ≈ 1e10 to 3e10 elements per request), not kernel speed. This ledger
breaks n_w down by committed tensor and op class, so that cuts can be ranked
the way bits/gas cuts were (virtualise derivable columns, pack the rest).

**Metric** (lower is better, soundness unchanged):
* committed bits per token per layer = 31 × committed elements / T / layers,
  separately for dense and MoE layers;
* prover ns per token per layer;
* projected Maverick committed elements and prove time per request.

## 1. Running it

```
TENSOR_LEDGER=1       tensor-bench layer ... | model ...   # ledger of the first proof
TENSOR_LEDGER=noprove tensor-bench layer ...               # forward pass + ledger, no proof
python3 -I src/frontends/int_inference/ledger_project.py run1.jsonl [run2.jsonl ...]
```

The bench prints one JSON line per committed tensor (`"ledger":"tensor"`),
per LogUp key (`"key"`), per class (`"class"`) and a `"summary"` after the
bench line. `ledger_project.py` checks a closed-form element count for every
class against the measured one (they must match exactly) and evaluates the
same formulas at the Maverick shape.

* Labels: `transformer.Builder` records a label per witness tensor: scope
  (block: `rms1`, `attn.proj`, `rope`, `attn`, `residual`, `rms2`, `mlp`,
  `moe.route`, `moe.expert`, `embed`, `head`, `head.greedy`, `rms.head`) and
  name (`scores`, `sel`, `d`, `e`, `den`, `acc`, `o`, `od`, `div.slack_lo`,
  `gate`, `disp`, ...), plus layer index and dense/MoE. Labels never enter
  the statement or the transcript.
* Capture: `protocol.proveCore` hands the committed tensors (witness +
  multiplicity columns), the derived LogUp relations and the witness-table
  layout to `ledger.sink` after the witness opening. It reads only. The
  `"ledger: every witness tensor labelled, proofs unchanged with the ledger
  on"` test proves an MoE layer with and without the sink and checks that the
  full proof digests are equal, that every tensor has a block label, and that
  `protocol.witnessLedger` (the no-prove path) sees the same tensors.
* Columns of the class line: `elems`; `bits31` = elems × 31 (what the M31
  commitment encodes); `used_bits` = elems × ceil(log2(max − min + 1)) (a
  packing lower bound, data-dependent); `gkr_leaves_direct` (key leaf = a
  committed column, possibly shifted by a constant) and `gkr_leaves_virtual`
  (key leaf = an affine combination of committed columns or accumulators
  that is never committed, e.g. requant remainders); `opening_bytes` = the
  witness opening attributed by element share (each queried column of the
  witness table holds every committed row).

## 2. Measured (this Mac, 12 threads, synthetic weights)

| run | command | committed elems | table (padding) | used / 31 bits |
|---|---|---:|---:|---:|
| dense layer D 2048, F 4096, 16q/4kv, 1024 tok | `TENSOR_LEDGER=noprove tensor-bench layer 10 11 12 4 12 54 2 20 2 1 0 10 1 2 54` | 124,371,176 | 124,780,544 (409,368) | 24.9 % |
| same layer, 512 tok, proved | `TENSOR_LEDGER=1 tensor-bench layer 9 11 12 4 12 54 2 20 2 1 0 9 1 2 54` | 45,444,324 | 45,613,056 (168,732) | 25.4 % |
| LM: embed + dense + MoE (64 experts top-1, capacity fitted = 32) + head, 512 tok | `TENSOR_LEDGER=1 tensor-bench model t=9 d=8 f=10 fe=9 h=3 kv=1 layers=2 e=6 v=14 key=0 r=1` | 39,253,713 | 39,321,600 (67,887) | 14.3 % |

The 1024-token layer is ledgered without a proof: its proof needs ~12 GB
(speed.md), above this host's 8 GB per-job cap. The 512-token proof gives
the timings and opening bytes. The MoE stack keeps Maverick's tokens per
expert (T / E = 8; 64 experts is the frontend's limit, `LayerCfg.validate`);
the fitted capacity was 32 slots (max expert load 20). The formula check
matches every class of all three runs.

GKR leaves (dense 1024-token layer): 80.2M direct, 48.8M virtual, 72K table
rows. Witness opening (512-token proof): 175,400 bytes of a 469,104-byte
proof.

### Dense layer, per token per layer (T = 1024, D = 2048, F = 4096, H = 16, Hkv = 4)

| class | elems/token/layer | bits31/token/layer | max value bits | share | GKR leaves direct / virtual | opening bytes (T=512 proof) |
|---|---:|---:|---:|---:|---|---:|
| attn/scores | 16,384 | 507,904 | 8 | 13.49% | 16,777,216 / 16,777,216 | 16,129 |
| attn/sel | 16,384 | 507,904 | 1 | 13.49% | 0 / 0 | 16,129 |
| attn/d | 16,384 | 507,904 | 10 | 13.49% | 16,777,216 / 0 | 16,129 |
| attn/e | 16,384 | 507,904 | 9 | 13.49% | 0 / 0 | 16,129 |
| attn.proj/qkv | 4,096 | 126,976 | 8 | 3.37% | 4,194,304 / 4,194,304 | 8,064 |
| attn/div.slack_lo (2 limbs) | 4,096 | 126,976 | 16 | 3.37% | 4,194,304 / 0 | 8,064 |
| attn/div.slack_hi (2 limbs) | 4,096 | 126,976 | 16 | 3.37% | 4,194,304 / 0 | 8,064 |
| residual/residual (h, out) | 4,096 | 126,976 | 9 | 3.37% | 2,097,152 / 0 | 8,064 |
| mlp/gate | 4,096 | 126,976 | 8 | 3.37% | 4,194,304 / 4,194,304 | 8,064 |
| mlp/up | 4,096 | 126,976 | 8 | 3.37% | 4,194,304 / 4,194,304 | 8,064 |
| mlp/silu | 4,096 | 126,976 | 7 | 3.37% | 0 / 0 | 8,064 |
| mlp/glu | 4,096 | 126,976 | 8 | 3.37% | 4,194,304 / 4,194,304 | 8,064 |
| input/x | 2,048 | 63,488 | 7 | 1.69% | 2,097,152 / 0 | 4,032 |
| rms1/u, rms1/y, rms2/u, rms2/y | 2,048 each | 63,488 each | 7-8 | 1.69% each | 2,097,152 / 2,097,152 each | 4,032 each |
| rope/q_rot | 2,048 | 63,488 | 8 | 1.69% | 2,097,152 / 2,097,152 | 4,032 |
| attn/acc, attn/od | 2,048 each | 63,488 each | 18 | 1.69% each | 0 / 0 | 4,032 each |
| attn/o | 2,048 | 63,488 | 7 | 1.69% | 2,097,152 / 0 | 4,032 |
| attn.proj/wo, mlp/down | 2,048 each | 63,488 each | 7-8 | 1.69% each | 2,097,152 / 2,097,152 each | 4,032 each |
| rope/k_rot | 512 | 15,872 | 7 | 0.42% | 524,288 / 524,288 | 1,008 |
| logup multiplicities (all tables) | 70 | 2,170 | 7-22 | 0.06% | - | ~280 |
| attn/m, attn/den | 16 each | 496 each | 7, 16 | 0.01% each | 0 / 0 | 32 each |
| rms ss / rt / rt2 / inv / inv_rt | 1 each | 31 each | 3-21 | 0.00% | small | 2 each |
| **total** | **121,456** | **3,765,136** | | | | 175,400 |

The four H·T² attention tensors are 54 % of the dense layer at T = 1024. One
of them (`sel`) uses 1 of its 31 bits. Value bits come from synthetic
weights; the MoE run's synthetic activations partly collapse (some 0-bit
columns), so its used-bit figures are not representative.

### MoE layer, per token per layer (T = 512, D = 256, Fe = 512, E = 64, C = 32)

| class | elems/token/layer | formula (per request) |
|---|---:|---|
| attn/scores, sel, d, e | 4,096 each | H T² each |
| moe.route/disp | 2,048 | E C T |
| moe.expert/gate, up, silu, glu | 2,048 each | E C Fe each |
| moe.expert/xe.gather, down | 1,024 each | E C D each |
| attention T·D classes, rms, rope, qkv/wo, residual | as dense | as dense |
| moe.route/logits | 256 | T Nr (router padded to the 2^8-column weight table) |
| moe.route/sel, dr, pre | 64 each | T E each |
| moe.route/occ, m | 4, 1 | E C, T |
| moe.expert/combine | 256 | T D |
| request: embed/onehot | 16,384 per token (whole request) | T V |
| request: head/logits, head.greedy/dr | 64, 32 per token | 2 V, V |

The routed experts commit E · C = 2048 slot rows for 512 tokens: 4 × the
active rows (capacity padding to the max load, then to a power of two).

## 3. Formulas and the Maverick projection

Per request, T tokens, dh = D/H, Dkv = Hkv dh, Q = pow2(D + 2 Dkv) (fused
QKV), slack limbs λ = 2 if a division slack (lt + 9 bits over n = lt +
log2 D key bits) takes the limb path, else 0 (section 6: since the wide
slack key, λ = 0 whenever 17 <= lt + 9 <= 20 and n >= lt + 11, i.e. D >= 2048
at T = 256 to 2048):

* attention block (dense and MoE layers):
  rms1 + rms2: 6 T D (xs, u, y; xs only when the input is wider than the
  activations) + 10 T (+ 8 T isqrt limbs when log2 D + 5 > 16);
  qkv T Q; rope T D + T Dkv; scores + sel + d + e 4 H T²; m + den 2 H T;
  acc + o + od 3 T D; division slacks 2 λ T D; wo T D.
* dense MLP: 4 T F (gate, up, silu, glu) + T D (down) + 2 T D (residuals).
* MoE: routing T Nr + T + 3 T E + E C T + E C; experts E C (2 D + 4 Fe);
  combine T D; residuals 2 T D. Shared expert (Maverick; **not implemented
  here**): 4 T Fs + 2 T D.
* request: one-hot T V, gather T D, head 2 D + 3·2 D + 10 + 2 V,
  greedy V + 1, multiplicity columns ~1e5 (table sizes).

n_w = 24 × dense + 24 × MoE + request. Maverick: T = 1024, D = 5120,
40 q / 8 kv heads (dh = 128), F = 16384, Fe = Fs = 8192, E = 128, C = 32
(same tokens per expert and fitted capacity as the measured MoE run),
V = 202048.

| | model dims (ideal) | power-of-two dims (as stwo-zig builds it: D 8192, 64 heads, V 2^18, Nr 2^13, Q 16384) |
|---|---:|---:|
| dense layer, elems / token | 329,826 | 492,690 |
| dense layer, bits31 / token | 10.2 M | 15.3 M |
| MoE layer, elems / token | 483,943 | 685,591 |
| MoE layer, bits31 / token | 15.0 M | 21.3 M |
| **n_w per request** | **2.02e10** | **2.92e10** |
| committed bytes at 31 bits | 72.9 GiB | 105.5 GiB |

This sits inside serving.md's 0.93e10 to 2.6e10 range (its central 2.0e10
agrees with the ideal-dims figure); the power-of-two padding alone adds 45 %.

### Ranking by share of committed bits (Maverick, model dims)

| family | elements | share |
|---|---:|---:|
| MoE: routed experts (E·C slot stacks) | 4.35e9 | 21.5 % |
| dense: attention T² (scores/sel/d/e) | 4.03e9 | 19.9 % |
| MoE: attention T² (scores/sel/d/e) | 4.03e9 | 19.9 % |
| dense: MLP (gate/up/silu/glu/down) | 1.74e9 | 8.6 % |
| MoE: shared expert (projected, not implemented) | 1.06e9 | 5.2 % |
| dense / MoE: attention T·D (acc/o/od/slack limbs, m/den) | 0.88e9 each | 4.4 % each |
| dense / MoE: RMSNorm parts | 0.76e9 each | 3.7 % each |
| dense / MoE: attention projections (qkv, wo outputs) | 0.30e9 each | 1.5 % each |
| dense / MoE: residual stream | 0.25e9 each | 1.2 % each |
| request: one-hot embedding | 0.21e9 | 1.0 % |
| dense / MoE: RoPE outputs | 0.15e9 each | 0.75 % each |
| MoE: routing (logits/sel/dr/pre/disp/occ) | 0.11e9 | 0.6 % |
| request: gather, LM head + greedy, head RMSNorm | 6e6 | 0.03 % |

With power-of-two dims, attention T² rises to 44 % (64 padded heads) and the
router logits (padded to 2^13 columns for 128 experts) to 1.1 %.

### Prover time

| run | prove | ns / committed element | ns / token / layer |
|---|---:|---:|---:|
| dense layer, 512 tok (proof) | 0.893 s | 19.6 | 1.74 M |
| LM, 2 layers + head, 512 tok (proof) | 1.067 s | 27.2 | 1.04 M (incl. head) |

**Both timings were taken on battery power** (pmset: "Battery Power"); they
are indicative only and must be re-run on AC before quoting. At 20 to 27
ns/element, Maverick's 2.02e10 elements are 400 to 550 s of proving per
request on this CPU, before the weight fold and the shared opening, and
assuming ns/element does not grow with size (it did between T = 256 and
512, serving.md). Per token per layer: dense 6.5 to 9.0 ms, MoE 9.5 to
13.2 ms.

## 4. What the ledger points at (candidates, not implemented)

Savings below are from the formulas at Maverick model dims (n_w 2.02e10).
None of them is implemented or measured. Each needs its own soundness
argument and adversarial tests before it lands.

| lever | removes | est. share |
|---|---|---:|
| capacity padding: expert stacks at T slots (sorted / ragged dispatch) instead of E · C = 4 T | 3/4 of the routed-expert rows | ~16 % |
| causal half: commit scores / sel / e for j ≤ t only (on the masked half, sel = 0 and e is the exp table at the masked bias; needs a triangular layout and a proof that the mask is honoured) | ~half of 3 H T² | ~15 % |
| `d` virtual: the exp key's input is the expression m − S − bias (already a lineq), not a committed column | H T² per layer | ~10 % (**implemented**, section 6) |
| fused glu (`Op.glu`, one triple lookup) in the transformer MLPs and expert stacks: drops the committed SiLU output (**implemented, section 4a**: -7.0 % measured by formula) | T F, E C Fe, T Fs | ~7 % |
| division slacks as virtual keys into one 2^(lt+9) range table (instead of 16-bit limb pairs). **Implemented, section 7** | 4 T D per layer | 5.0 % (measured formula) |
| residual stream as expressions (lineq sums) where consumers accept expressions | 2 T D per layer | ~2.5 % (needs checking) |
| one-hot embedding as a public input to the gather (verifier evaluates its sparse MLE) | T V | ~1 % |
| together | | ~56 % → n_w ≈ 8.8e9 |

Packing: the dense layer uses 25 % of its committed bits (`sel` and `disp`
1 bit of 31, activations 7 to 8 bits). Packing several narrow values per
field element is the second lever after virtualisation, as in the bits/gas
work. It changes the PCS encoding, so it is a protocol change, not a tweak.

## 4a. Implemented: fused SwiGLU gate (`fused-glu`)

Sections 2 and 3 are the baseline before this lever, measured at d3545530f.

**Change.** `transformer.Builder.gate(g, u, silu)` emits the SwiGLU gate as
one `proto.Op.glu` (the triple LogUp key g + βu + β²h into
{(g, u, clamp8(rq_s(silu(g)·u))) : g, u ∈ int8}, 2^16 rows per distinct
(table, shift)) whenever g has ≥ 2^16 elements (`GLU_MIN_NUMEL`).
Otherwise it keeps the committed `lookup(g, silu)` + `hadamard(act, u)` pair.
The pair stays below the threshold because there the glu table's 2^16
multiplicity rows would cost more than the SiLU column they remove. All
three int8 SwiGLU sites use it: `transformer.denseMlp`, the per-expert loop
in `transformer.moe`, and `lm.moeStacked` (g = [E, C, Fe]). Wide mode
(act_bits ≠ 8) is unchanged. `Builder.glu` computes silu(g)·u in memory and
never commits it. It takes the shift from the hadamard's plan slot, chosen by
the same `model.chooseShift` on the same accumulator, and keeps `checkInt8`.
So h is bit-identical to the old pair's output (tested), and the shift plan
length is unchanged. The verifier's shape-only rebuild (`lm.requestStatement`)
makes the same choice, because it depends on shapes only.

**Removed per hidden unit:** the committed SiLU tensor, its `lookup_pair` key,
the hadamard remainder key, the hadamard y_range key, the hadamard requant
sumcheck and the SiLU column's PCS claim. **Added:** one glu_triple key per
hidden unit, and the 2^16-row glu table per distinct (table, shift). The
up matmul's y_range key is also dropped (`rangeImplied` covers glu inputs
exactly as it covers lookup inputs; the gate's was already dropped as the
SiLU lookup's input). Net: lookup_pair, hadamard remainder, hadamard y_range
and up y_range out, one glu_triple in, so -3 GKR leaves per hidden unit.
Dropping the up y_range is sound because the up requant remainder is unique
over the integers (|acc| + 2^s 128 < p / 2), so the only field-level freedom
is to wrap u mod p, which leaves int8 and is rejected by the glu table.

**Soundness.** Op.glu is the existing, reviewed relation (zk.zig, model.zig
fused MLP, `security.zig` per-relation degree: arity 3 → degree 2, derived
automatically into `logup_degree_fractions`). Table membership makes g and u
int8, and makes h = clamp8(rq_s(silu(g)·u)) exactly. The one semantic
difference: the relation accepts the saturated value, where the old pair had
no valid witness (the hadamard's y_range would fail). The saturated value is
a deterministic function of the committed (g, u), so the prover gains no
freedom. The honest prover still refuses such inputs (`checkInt8`). `bounds.zig`
already treats glu outputs as int8, so the down matmul's accumulator bound is
unchanged.

**Tests** (transformer.zig, lm.zig):
* fused at 2^16 versus pair at 2^8: h equals rq_s(silu(g)·u) with
  s = chooseShift; proves and verifies; the honest prover refuses h + 1;
* on a statement where the glu triple is the only constraint on free g and u
  (test-only unchecked glu multiplicities), the verifier rejects h + 1, a
  moved g with u and h fixed, u = 200 with h its natural clamp extension, and
  g = 384;
* checker test: u wrapped mod p as u - t 2^(31 - s), so the up matmul's
  field remainder stays in [0, 2^s); the cheating prover counts remainders
  mod p (test-only `prep.test_field_remainders`) and the verifier rejects
  (LogUpSumNonZero: only the glu table's int8 domain on u catches it); a
  control with the hooks on and an honest witness verifies;
* model with a dense layer, stacked MoE and per-expert MoE all at 2^16: the
  verifier-built statement is byte-identical to the prover's, and the
  per-request proof verifies.

No pinned digest changes, because no golden shape reaches 2^16 gate elements.

**Measured** (this Mac, 12 threads; timings on battery power, A/B interleaved,
medians of 5 (layer) and 3 (model) pairs):

| run | before | after | Δ |
|---|---:|---:|---:|
| dense layer D 2048 F 4096 T 1024 (noprove): committed elems | 124,371,176 | 120,242,120 | −4,129,056 (−3.3 %) |
| same, per token per layer (incl. input x, multiplicities) | 121,456 | 117,424 | −4,032 |
| same, bits31 per token per layer | 3,765,136 | 3,640,144 | −124,992 |
| same, GKR leaves direct / virtual | 80.2M / 48.8M | 71.8M / 44.6M | −3 per hidden unit |
| same, multiplicity columns | 71,912 | 137,160 | +65,248 (glu table − act table) |
| dense layer T 512, proved: prove s | 0.915 | 0.830 | −9.4 % |
| same: witness commit / GKR stage s | 0.295 / 0.278 | 0.197 / 0.258 | |
| same: proof bytes | 469,104 | 462,848 | −1.3 % |
| same: peak RSS | 6.13 GB | 5.79 GB | −0.34 GB |
| LM 2 layers (dense + MoE E 64 C 32 Fe 512) + head, T 512: committed | 39,253,713 | 37,746,129 | −1,507,584 (−3.8 %) |
| same, dense / MoE elems per token per layer | 25,434 / 34,335 | 24,410 / 32,287 | −F / −E C Fe / T |
| same: prove s | 1.128 | 1.133 | flat (noise) |
| same: per-request proof bytes | 533,880 | 519,640 | −2.7 % |
| same: peak RSS | 5.42 GB | 5.08 GB | −0.34 GB |

`ledger_project.py` formulas match every measured class before and after.
Its `--no-fused-glu` flag evaluates the baseline.

**Maverick projection** (formulas, 24 dense + 24 MoE layers, shared expert
counted as projected):

| | before | after |
|---|---:|---:|
| dense elems / token / layer (model dims) | 329,826 (10.2 Mbit) | 313,442 (9.72 Mbit) |
| MoE elems / token / layer (model dims) | 483,943 (15.0 Mbit) | 442,983 (13.7 Mbit) |
| n_w per request, model dims | 2.021e10 | 1.880e10 (−1.41e9, −7.0 %) |
| n_w per request, power-of-two dims | 2.924e10 | 2.783e10 (−4.8 %) |
| GKR leaves per request | | −4.2e9 (3 × 1.41e9 hidden units) |
| prove, serving.md 40–55 ns/elem | 810–1110 s | 750–1030 s |
| prove, H100 projection 7.5–9 ns/elem | 152–182 s | 141–169 s |

Without the shared expert (not implemented), the routed experts and dense MLP
alone give −1.21e9. The time rows assume constant ns/elem. The measured T 512
dense layer dropped more than its element share (−9.4 % time for −4.5 %
elements, since GKR leaves fell 10 %), and the LM run was flat. The LM proof
is dominated by the one-hot embedding and attention, and the glu table adds
2^16 multiplicity rows there. Treat the time saving as the element share
(about 7 %) until it is re-measured on AC power.

## 5. Caveats

* The rms `xs` tensor is data-dependent: it is not committed when the input
  already fits the activation width (the formula counts it).
* Maverick's shared expert is not implemented. It is projected as a dense
  SwiGLU of width Fs plus one add.
* C = 32 assumes Maverick's max expert load fits 32 slots, as in the 64-expert
  run (max load 20 at 8 tokens per expert on average).
* Wide activations (accuracy.md: A = 12, R = 16) were not ledgered at D 2048.
  They cannot be built there, and MoE layers refuse A or R ≠ 8. Width changes
  used bits and the range-table sizes, not the element counts (except
  embedding limbs).
* Fix made while ledgering: `ext.gatherPacked` started its last chunks past
  the end when n_ch × chunk > len, which panicked in ReleaseSafe and crashed
  in ReleaseFast. The model bench at E = 64, T = 512 hit this. Chunk
  starts are now clamped. Outputs for every shape that ran before are
  identical, so no digests change.

## 6. Implemented levers

### 6.1 attn-virtual-d-sel (branch `codex/tensor-wv-attn-virtual-d-sel`)

Two commits, both in `transformer.attention` (statement and soundness:
transformer.md sections 2 and 3):

1. **D virtual.** `attn/d` (H T²) is no longer committed. The exp key's input
   is the expression m − S − Bias; the D lineq is gone. Attainment moves from
   Σ_j sel·D = 0 to z1 = Σ_j sel·S and z2 = Σ_j sel·Bias (committed, H T
   each) with m − z1 − z2 = 0, and m gets its own signed score_bits range key.
2. **Factored selector** (log_t ≥ `FACTORED_SEL_MIN_LOG_T` = 4; toy
   statements keep sel). sel[h,t,j] (H T²) becomes A[h,t,jh] and B[h,t,jl]
   (j = jl + 2^⌊lt/2⌋ jh), each boolean (zero-check) and one-hot (sum lineq),
   with partial sums PS = Σ_jl B·S and PB = Σ_jl B·Bias (committed,
   H T 2^⌈lt/2⌉ each) and z1 = Σ_jh A·PS, z2 = Σ_jh A·PB. bounds.zig's
   existing one-hot rule covers both einsum levels (no bounds.zig change).

Measured (this Mac, 12 threads, **battery power**, single runs; timings are
indicative only):

| run | before | after commit 1 | after commit 2 |
|---|---:|---:|---:|
| dense layer D 2048, F 4096, 16q/4kv, 1024 tok (noprove): committed elems | 124,371,176 | 107,626,728 | 92,946,664 |
| same, elems / token / layer (incl. the input) | 121,456 | 105,104 | 90,768 (−25.3 %) |
| same, bits31 / token / layer | 3.77 M | 3.26 M | 2.81 M |
| same, GKR leaves direct / virtual | 80.2 M / 48.8 M | 63.5 M / 65.5 M | 63.5 M / 65.5 M |
| dense layer, 512 tok (proof): committed elems | 45,444,324 | | 37,989,604 (−16.4 %) |
| same, prove (ns / committed elem) | 0.914 s (20.1) | | 0.845 s (22.2) |
| same, proof bytes (witness opening) | 469,104 (175,400) | | 466,216 (169,168) |
| LM embed + dense + MoE (E 64, C 32) + head, 512 tok: committed elems | 39,253,713 | | 31,798,993 (−19.0 %) |
| same, dense layer / MoE layer elems per token | 25,434 / 34,335 | | 18,154 / 27,055 |
| same, prove per request (ns / elem) | 1.300 s (33.1) | | 1.163 s (36.6) |
| same, request bytes (per-request total) | 532,792 | | 539,216 |

The formula check (`ledger_project.py`) matches every class of every run.
Commit-1 elements were measured on a separate build of the commit-1 tree.
Proof bytes barely move: the witness opening shrinks with the committed
columns (−6 KB), and the new einsum sumchecks and ext claims add about the
same. ns/element rises because the H T² exp-key work (now a 3-term virtual
expression), the sel·S einsum (now B·S at H T²) and the scores stay while
committed elements fall; the prove time still falls by 8 to 11 %.

Maverick projection (formulas, section 3):

| | before | after |
|---|---:|---:|
| model dims: dense / MoE elems per token per layer | 329,826 / 483,943 | 253,106 / 407,223 |
| model dims: bits31 per token per layer | 10.2 M / 15.0 M | 7.85 M / 12.6 M |
| model dims: n_w per request | 2.02e10 | 1.64e10 (−18.7 %) |
| power-of-two dims: n_w per request | 2.92e10 | 2.32e10 |

Per token per layer at model dims (H = 40): d −40,880 and sel −35,840 in
both layer kinds. At 22 to 37 ns per element (the battery-power figures
above), 1.64e10 elements are 365 to 600 s of proving per request on this
CPU, against 406 to 669 s for 2.02e10 at the "before" ns/element; to be
re-measured on AC (`TENSOR_LEDGER=1`) before quoting.

Soundness: the exp-table membership of the virtual D_j proves
m ≥ S_j + Bias_j as integers (|m| < 2^(sb−1) from m's range key,
|S| < 2^(sb−1), |Bias| ≤ 2^sb + 256, so the expression plus the 2^(sb+2)
domain is far below p: `bounds.exprCheck`); one-hot A ⊗ B (or sel) with the
exact einsums gives m = S(j*) + Bias(j*). security.zig accounts ext ops
generically (`accountExt`): commit 1 removes one lineq over n3 and adds two
einsum sumchecks over n2, one lineq over n2 and one range relation; commit 2
removes an n3 zero-check and an n2 lineq, adds two zero-checks over
n2 + lt/2, two lineqs and four einsum sumchecks, each term far below
2^−100 over QM31. Adversarial tests (unchecked prover, all rejected): plain
selector `m_lowered`, `m_raised`, `sel_moved`, `sel_nonbinary`; factored
selector `m_lowered` (A, B, PS, PB, z1, z2 and e consistent), `m_raised`,
`a_not_one_hot` (2 and −1, sum 1), `b_two_hot`, `a_moved` (z1, z2 following
PS, PB), `ps_forged`.

Digests repinned: golden decoder layer flat 837b19ac → 8424c4b6 (commit 1)
→ a2bcc5c7 (commit 2), pcs2_moe 0d99b4e3 → 79360fbd → 4dd28279, and the
CUDA-emu and Metal CPU-golden pins with flat.

## 7. Wide slack key (implemented)

`Builder.rangeWide(n, e, bits)` proves `e in [0, 2^bits)` over `{0,1}^n`.
Above 16 bits it used to commit two 16-bit limbs `lo`, `hi`, key both into
range(16) / range(bits - 16) and add the lineq `e = lo + 2^16 hi`. It now
emits **one key of the never-committed expression `e` into range(bits)**
when

    16 < bits <= MAX_TABLE_BITS (20)   and   n >= bits + 2

(`Builder.wideKeyDirect`). The second condition keeps the 2^bits multiplicity
column at most 1/4 of the 2 · 2^n limb elements it replaces. The rule depends
only on (n, bits), which the verifier rebuilds from the model description, so
the statement stays deterministic and no prover-supplied shape enters it. The
honest prover still refuses an out-of-range `e` (`ValueOutOfRange`).

Where it fires: the two attention division slacks `2 acc + den - 2 od` and
`den - 2 acc + 2 od - 1` (lt + 9 bits over n = lt + lh + ldh = log2(T D)),
whenever D >= 2048 and 8 <= lt <= 11. The RMSNorm isqrt slacks (n = lt) and
every toy shape keep the limb path, so **no pinned digest changes** (all
proof-digest tests pass unchanged). Proofs at the larger shapes (ledger
benches) change, as intended.

**Soundness.** The limb path proves exactly `{e : e = lo + 2^16 hi, lo in
[0, 2^16), hi in [0, 2^(bits-16))} = [0, 2^bits)` as integers, given the
lineq is an integer identity (|e| + 2^bits < p, checked by `bounds.zig`). The
direct key proves `e mod p in [0, 2^bits)` by LogUp membership; with
`|e| + 2^bits < p` (bounds.zig `exprCheck` on every non-covering key input,
the same magnitude the lineq check used) that is the same integer set. For
the slacks |e| <= 2 |acc| + |den| + 2 |od| < 2^(lt+19) <= 2^30 at lt <= 11
(acc, od < 2^(lt+16)), and 2^30 + 2^20 < p; bounds.zig re-derives this per
statement from the proved bounds and rejects the statement otherwise. LogUp counts: range(lt + 9) gets 2 T D keys per layer
(Maverick: 48 · 2 · 2^10 · 5120 ≈ 5.0e8 at model dims, 48 · 2 · 2^23 ≈ 8.1e8 at the power-of-two D 8192 stwo-zig builds; both < p, `checkLogUpCounts`), and the
range(16) relation loses 4 T D limb keys per layer (−1.0e9 at Maverick),
which helps the per-relation LogUp bound that a full-scale run needs.
Security accounting is derived from the statement (`security.zig`): one
lineq term (n/|F|) per slack disappears and the LogUp terms are re-derived
from the relations; the dense 512-token proof reports 149 terms (was 151)
and the same 56.59-bit compiled level.

**Tests** (`wide_slack_test.zig`): the rule table; a synthetic statement with
`3a - b` keyed into range(17) over n = 19 (no limb tensors, no lineq) that
proves honestly and rejects `3a - b = -1` and `3a - b = 2^17` with a, b
inside their own ranges (unchecked prover). A mutant that keys into
range(bits + 1) is caught by the 2^17 case. The honest prover refuses both
values. Attention at T = 256 with one 2048-wide head: no limb or slack
tensors, exactly two range(17) keys, validateStatement passes, it proves, and
EXT-1's attack (inflate o on the last row with od = o · den recomputed) is
rejected for o in {1, 64, 127, -1}.

`wide_slack_adv_test.zig` (independent check) requires every cheat to be
*produced* by the unchecked prover and then rejected by the verifier, so no
rejection comes from a prover-side check: `3a - b` in {-1, -65535, 2^17,
2^17 + 1, 3 (2^16 - 1)} rejected and the edges 0, 2^17 - 1 accepted; a proof
for the statement with the key retargeted to range(18) verifies against that
statement but is rejected against the honest one; attention at T 256 with a
2048-wide head rejects o ± 1 (od recomputed) on rows 0, 1, 37, 255; and a
virtual key whose expression can reach p (c a - b with c = 2^15) is refused
by validateStatement (StatementAccumulatorBound). Mutants that drop the key
or key into range(bits + 1) fail these tests.

### Measured (battery power; times indicative only)

| run | committed elems before → after | per token per layer | proof bytes | prove s (median of 3) |
|---|---:|---:|---:|---:|
| dense layer T 1024, D 2048, F 4096, 16q/4kv (noprove) | 124,371,176 → 116,506,848 (−6.3 %) | 121,456 → 113,776 (3.77 → 3.53 Mbit) | – | – |
| dense layer T 512, same widths (proved) | 45,444,324 → 41,512,160 (−8.7 %) | 88,758 → 81,078 | 469,104 → 463,624 | 0.972 → 0.963 |
| MoE layer T 512, D 2048, E 16, C 128, Fe 512 (`layer 9 11 9 4 12 54 2 20 2 1 4 7 1 2 54`) | 55,170,788 → 51,238,624 (−7.1 %) | 107,755 → 100,075 | 688,392 → 684,736 | 1.457 → 1.248 |
| LM, embed + dense + MoE, D 256 (section 2 run) | 39,253,713 → unchanged (n = 17 < 20: limbs kept) | – | – | – |

The dense T 1024 delta is −8,388,608 (4 T D limbs) + 524,288 (the range(19)
multiplicity column) − 8 (the range(3) table is gone). GKR leaves: direct
80.2M → 71.8M, virtual 48.8M → 53.0M, tables 72K → 596K (total −2.8 %).
`ledger_project.py` reproduces every class of both dense runs exactly.
(Its formula check does not model the MoE *layer* bench, whose classes are
the transformer.moe path, before or after; the MoE delta above is measured.)
The measured dense prove time moves by about 1 %: the limbs were cheap to
commit next to GKR and the H·T² work, so elements fall faster than time.

### Projection (Maverick, ledger formulas)

| | before | after |
|---|---:|---:|
| dense elems / token / layer (model dims) | 329,826 (10.2 Mbit) | 309,346 (9.59 Mbit) |
| MoE elems / token / layer (model dims) | 483,943 (15.0 Mbit) | 463,463 (14.4 Mbit) |
| n_w per request, model dims | 2.021e10 | 1.921e10 (−1.00e9, −5.0 %) |
| n_w per request, power-of-two dims (D 8192) | 2.924e10 | 2.763e10 (−1.61e9, −5.5 %) |

Time: at the per-element coefficients (19.6-27.2 ns/elem battery, 40-55 ns
serving.md) the −1.0e9 elements would be −20 to −55 s per request on this
CPU, but the measured 1 % prove-time change says the realistic gain is a few
seconds; the element count, the opening bytes and the LogUp count are the
real wins.
