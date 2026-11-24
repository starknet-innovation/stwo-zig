# evmone Ethereum client guest (RV64IM)

A C++ stateless Ethereum client for this repository's RV64IM guest ABI. It is
built on [ziskethone](https://github.com/0xPolygonHermez/ziskethone), the ZisK
team's C++ Ethereum block client, and on [evmone](https://github.com/ethereum/evmone)'s
baseline interpreter (v0.24). The guest runs ziskethone's block pipeline and
state database with evmone as the EVM. It reads the same canonical SSZ
`StatelessInput` as the [Rust guests](../rv64/README.md) and writes the
identical 43-byte output. It is exact on all 15 basket blocks under both the
admitted and the candidate profiles.

The design is ZisK's. Their measurements, quoted in
[`../abi/CONTEXT.md`](../abi/CONTEXT.md), put ziskethone at 2.02 steps per gas
against 6.21 for ZisK's Reth guest. Most of the work here adapts ziskethone and
evmone to a plain RV64IM machine without ZisK's DMA and bit-manipulation
instructions: aligned-word memory paths, a lazily walked SSZ witness, and the
admitted Keccak-f and secp256k1 services. Changes to upstream source are
carried as patches in `patches/` and summarised below. The work started as track A of the archived
[guest-efficiency program](https://github.com/starknet-innovation/stwo-zig/tree/archive/pre-restructure-20261108/autoresearch/guest_efficiency/PROGRAM.md).

## Licences and third-party sources

No third-party source is committed. `build.py` fetches pinned public sources
into a cache directory and applies the patches in `patches/`:

| Source | Commit | Licence |
|---|---|---|
| ziskethone `https://github.com/0xPolygonHermez/ziskethone` | `1c3e8d1737c0269a50af24a6fdb26091a6eaee27` | Apache-2.0 OR MIT |
| evmone `https://github.com/ethereum/evmone` | `2a4decd05b7e088b2da0e1f1924f8968e14e3586` (v0.24.0) | Apache-2.0 |
| intx `https://github.com/chfast/intx` | `0306b495820a5259563e5055f0a8b726b9ae4e0b` (v0.15.0) | Apache-2.0 |
| blst `https://github.com/supranational/blst` | `54e6e55674722fc2797ebb4bbb71b26d881eb4b8` (v0.3.17, evmone's pin) | Apache-2.0 |

ziskethone's own evmone patch stack (02-16, from the fetched tree) is applied
first; its `ZEG_ZISK`-gated hunks are inactive here. Licence review by legal is
still required before any of this code is vendored ([PROGRAM.md](https://github.com/starknet-innovation/stwo-zig/tree/archive/pre-restructure-20261108/autoresearch/guest_efficiency/PROGRAM.md)).

Everything under `glue/`, `tests/`, `adapter.py`, `build.py`, `patches/` and
`linker-wide-cxx.ld` is ours.

## Build

```sh
B=guests/ethereum/client_evmone
python3 -I $B/build.py --out OUT                         # OUT/client-evmone-admitted.elf
python3 -I $B/build.py --out OUT --profile candidate     # + candidate services (mem,sha,arith)
python3 -I $B/build.py --out OUT --profile candidate --candidate-set mem,sha   # subsets
python3 -I $B/build.py --out OUT --profile candidate --candidate-set mem,sha,arith,bn,bls,keccak  # + curve/sponge candidates
python3 -I $B/build.py --out OUT --host --tests          # native build + host tests
```

Toolchain: `zig c++` 0.15.2 only (bundled libc++ headers and compiler-rt,
`-target riscv64-linux-musl -mcpu=generic_rv64+m`, linked `-nostdlib` against
`linker-wide-cxx.ld`). `--jobs` defaults to 4. Writes `build-manifest-*.json`
(commits, applied patches, defines, ELF sha256).

The RV64IM v3 guest uses ThinLTO across translation units and RISC-V linker
relaxation to shorten calls. The manifest records these flags in
`guest_link_optimization`. Use the ELF digest, terminal PC and linker symbols
from the same build when preparing a proving request.

## Score

The scoring harness behind this README's numbers is archived as
[`bench.py`](https://github.com/starknet-innovation/stwo-zig/tree/archive/pre-restructure-20261108/autoresearch/guest_efficiency/bench.py); run it from a checkout of that tag:

```sh
python3 -I autoresearch/guest_efficiency/bench.py run --elf OUT/client-evmone-admitted.elf \
  --executor EXEC --cache CACHE --out RUN --set all --client evmone \
  --input-adapter "python3 -I $B/adapter.py --host-binary OUT/client-evmone-host"
```

The adapter is optional (the guest also accepts the plain canonical SSZ).

## Guest ABI mapping

| Item | Here |
|---|---|
| Entry | `glue/start.S`: `gp`, `sp = __stack_top`, run `.init_array`, `stwo_guest_main` |
| Input | `u32` length + canonical schema-prefixed SSZ at `__input_start` (or the adapter container, below) |
| Output | 43 bytes at `__output_data` (request root, 1, chain id LE, schema id LE), `__output_len = 43`, `__halt_flag = 1`, then spin |
| Failure | illegal instruction (`.word 0`), no output (the Rust guest panics the same way) |
| Linker | `linker-wide-cxx.ld` = `rv64/linker-wide.ld` plus `.init_array` inside `.data`; memory map and symbols unchanged |
| Patches | 0001 ziskethone (SSZ witness state, lazy rows, RV64 word paths), 0002 blst 64-bit limbs, 0003 intx byte-wise big-endian, 0004 candidate arith256_mod, 0005 evmc word compares and copy-free host trampolines; 0006-0017 interpreter; 0020/0021 candidate curves; 0023 v3 modexp; 0025 code digest in the ZEG0 contract record; 0026 blst constant-limb Montgomery product; 0027 ziskethone word copies; 0028/0029 JUMPDEST analysis on the jumpdest_bitmap service (admitted v3 default `jumpdest` group; `--candidate-jumpdest` for the execution-only candidate); 0030 deferred gas test for pure stack opcodes in `dispatch_cgoto` (the switch dispatch keeps per-opcode checks and is the fuzzer's reference) and PUSH1..7 with one store of limb 0; 0031 aligned EVM memory words; 0032 guarded stack and word-copy fusions; 0033/0034 shared prepared Miller loop for KZG; `wip/0024` translated segments (parked, not applied: measured slower) |
| Keccak-f | `0x0001_0001` in `glue/keccak_stwo.cpp` (replaces evmone's `keccak.c`) |
| secp256k1 | Complete RCB ADD word packets (`0x0002_0011`) cover additions and doublings in `glue/secp256k1_stwo.cpp`. |
| Candidate | `0x0003_0001-0003` bulk memory (runtime), `0x0003_0010` SHA-256 compress, `0x0003_0021` arith256_mod (MULMOD, ADDMOD, 256-bit `ModArith` in plain form), optional `0x0003_0020` arith256 for MUL |
| Candidate curves | groups `bn` (needs `arith`): `0x0003_0100/0101` BN254 affine add/double for ecAdd/ecMul, `0x0003_0102` Fq2 product in ecPairing; `bls`: `0x0003_0121` arith384_mont for blst's 384-bit Montgomery products and `0x0003_0112` BLS12-381 Fp2 product (KZG, EIP-2537); `keccak`: `0x0003_0130` Keccak-256 sponge for inputs over 64 bytes. Glue `glue/curves_stwo.cpp`, `glue/candidate_curves.hpp`, patches 0020/0021; ABI [`../abi/candidate-curves-abi.md`](../abi/candidate-curves-abi.md) |

## Design

- **Input.** The non-state ZEG0 sections ziskethone expects (consensus info,
  transactions, contracts, ancestor headers) are synthesized from the SSZ
  payload and witness (`glue/stwo_client.cpp`). The SSZ witness carries trie
  nodes without key preimages, so the state is not pre-walked: the account and
  storage tables are filled lazily (patch 0001 hooks) by walking the witness
  trie from the parent state root (`glue/witness_trie.cpp`). The parent header
  is bound by the payload's `parent_hash`.
- **Binding.** After execution every `ExecutionPayloadV3` field is compared
  with the re-executed header, the computed block hash with the payload's,
  versioned hashes with the blob transactions, and the request lists through
  `requests_hash`. The output root is the SSZ `hash_tree_root` of the input
  request (`NewPayloadRequestElectraFulu`, progressive containers and lists).
- **Witness trie.** Every witness node is hashed once into an open-addressing
  table; a node is decoded at most once, keeping the offset of each child item
  in the witness bytes (branch children scanned lazily, up to the child a
  lookup follows) and a per-child cache of the resolved node. Child digests
  are probed in place with funnel-shifted words; leaf paths stay hex-prefix
  encoded and are compared with the key 8 bytes at a time.
- **State root.** A mutable MPT over the decoded witness: unmodified children
  are tagged pointers (bit 63) to their RLP item in the witness bytes (no
  allocation), owned nodes are modified in place, only dirty paths are
  re-hashed. A branch owned from the witness records the positions assigned
  since and re-encodes by copying the runs of unchanged items from its
  original encoding. A full branch has sixteen digest references and a
  532-byte canonical encoding. The v3 sponge guest retains the Keccak state
  after its first two rate blocks (272 bytes). Re-encoding can resume there
  when the length is unchanged and all nine child items touching the prefix
  are unmodified. Other shapes and earlier edits hash the complete encoding.
- **State tables.** ziskethone's account and storage row indexes are
  open-addressing tables of key words (`zeg/stwo_flat_index.hpp`, patch 0001)
  with the same seeded hash; the seeded hash, fixed-size compares/copies and
  the evmc host trampolines and equality operators (patch 0005) use word
  accesses instead of the byte loops RV64IM gets for alignment-1 arrays.
- **Senders.** The SSZ public key of each transaction gives the sender at
  once; all signatures are checked in one batch (batched inversions, GLV
  endomorphism, a fixed generator comb and signed width-five digits). Verification with the
  y-parity check pins the unique recovered key, so it equals recovery.
- **Adapter (`adapter.py`).** Untrusted: runs the native host build and appends
  hints for every full recovery (key, `R.y`, `s^-1`). The guest checks
  `s*w = 1`, `R` on the curve with the right parity and `(zw)G + (rw)Q = R`
  projectively before using a hint, otherwise recovers in full. Container:
  `"STW1" | u64 ssz_len | SSZ | pad8 | u64 n | n x 232-byte records`.
- **RV64IM specifics.** Misaligned accesses trap, so memory functions use
  aligned words only (`glue/mem_functions.hpp`); SHA-256 uses duplicated-word
  rotations; intx big-endian conversions go byte-wise (patch 0003); blst uses
  64-bit limbs with `__int128` products (patch 0002); size-class free-list
  heap (`glue/runtime.cpp`).
- **Interpreter on RV64IM (patches 0005-0017).** 256-bit big-endian transfers
  at 8-aligned addresses use aligned words and a 13-instruction mask byte swap
  (masks in a mutable global so LLVM does not re-expand them); frameless
  MLOAD/MSTORE fast paths, funnel-shifted CALLDATALOAD, aligned-word context
  loads; column-wise 256-bit MUL, native DIV/MOD for 64-bit operands; gas
  tested on the decremented value and a constant on failed requirements (no
  per-opcode moves); host storage/log calls pass `msg->recipient` in place;
  sentinel JUMPDEST scan over the padded copy, unrolled by four; LT/GT/SLT/SGT
  join the fused `PUSH2 JUMPI` set. Measurements: [`notes/client-interpreter.md`](https://github.com/starknet-innovation/stwo-zig/tree/archive/pre-restructure-20261108/autoresearch/guest_efficiency/notes/client-interpreter.md) (archived).
- **Client pass 2 (glue, patches 0025-0027).** Memory functions: 32-byte
  funnel copy for misaligned sources, word heads/tails (one aligned RMW each)
  for memset and around the admitted service calls, service only from 64
  bytes. Contract codes are hashed while they are copied into ZEG0 (patch
  0025 carries the digest in the record: `u64 size | keccak256 | code | pad`).
  blst's portable Montgomery product runs with a constant limb count (patch
  0026). Hot evmc bytes32/address copies in ziskethone use `zisk_xmemcpy` (patch 0027); the secp glue's 96-byte
  point copies, the SHA-256 block copy and the Keccak first-block absorb are
  forced inline instead of memcpy calls. Measurements:
  [`notes/client-pass2-20261102.md`](https://github.com/starknet-innovation/stwo-zig/tree/archive/pre-restructure-20261108/autoresearch/guest_efficiency/notes/client-pass2-20261102.md) (archived).
- **JUMPDEST service (patches 0028, 0029).** With `ADMITTED_JUMPDEST` (default
  in `--profile v3`; `--v3-set sha,arith,mem` builds the previous v3 client)
  or `CANDIDATE_JUMPDEST` evmone's analysis is one `jumpdest_bitmap` ECALL
  (0x0003_0040) on the zero-padded copy: code length rounded up to 8,
  the copy 64-byte aligned inside its analysis-cache slot (0029 adds 56 bytes
  per slot), sentinel scan as the fallback for a misaligned copy. Without the
  defines both patches are inert (v3 per-block instruction counts identical).
  Proposal and measurements:
  [`proposals/jumpdest-bitmap.md`](https://github.com/starknet-innovation/stwo-zig/tree/archive/pre-restructure-20261108/autoresearch/guest_efficiency/proposals/jumpdest-bitmap.md) (archived).

## Ethereum guest execution qualification

Across 15 Ethereum blocks, patches 0032–0034 reduced the previous qualified guest from 1,285,147,794 to 1,267,505,327 retired instructions and from 315 to 310 CPU shards. Block 26,235,168 fell from 101,193,870 to 99,260,098 instructions and from 25 to 24 shards. All 15 host executions and all 15 physical RV64IM executions reproduced the expected 43-byte outputs. The receipt includes the input and ELF hashes, per-block steps, shard counts and service RAM traffic. [Execution qualification](../../../vectors/reports/rv64-productionisation/ethereum-guest-execution-20261123.json)

The interpreter fuses Solidity selector tests, width masks, negated comparisons and a bounded forward word-copy loop. Before consuming the loop it validates both jump destinations, stack limits, memory extent, aliasing and the full gas charge. Unmatched cases return to ordinary opcode dispatch. The KZG verifier combines the two prepared Miller loops in one field accumulator and shares their 62 doubling squares. Its regression compares the full Fp12 product before final exponentiation with two separate Miller loops.

Rebuild the RV64IM guest with `build.py --profile v3 --isa im` and a fresh output directory. Read its ELF digest from `build-manifest-v3.json` and its `__input_*`, `__output_*`, `__heap_*`, `__stack_*` and `__halt_flag` symbols from that exact ELF. Run the Ethereum execution driver with `engine=fast`, `execution_profile=ethereum_services_v3`, the input digest and expected output in the receipt. Run `build.py --host --tests` for the independent switch interpreter and cryptographic regression tests. The live GPU service binds its own ELF, ROM and derived proof classes. Promotion of a rebuilt guest requires a freshly verified recursive root for that identity.

## Tests

`build.py --host --tests` runs:

- `tests/mem_fuzz.cpp`: 400k randomized checks of the memory functions.
- `tests/secp_differential.cpp`: stwo recovery vs evmone's reference on random
  and edge-case signatures; hinted verification accepts true keys and rejects
  negated keys, wrong recid and wrong hash.
- `tests/curves_test.cpp`: the candidate curve/field/sponge glue with every hook
  live (portable blst, services emulated): ecMul/ecAdd vs a naive intx
  double-and-add, Fq2 and blst Fp/Fp2 products vs intx, BLS12-381 pairing
  bilinearity, evmone's KZG vectors, and the sponge vs the admitted Keccak path
  for every length 0..600 at all alignments.
- `tests/trie_differential.cpp`: witness trie vs a from-scratch reference MPT
  encoder (lookups, puts/removes over several root rounds, embedded nodes;
  partial witnesses report Missing and abort on mutation).
- `tests/keccak_copy_test.cpp`: the fused copy + Keccak-256 of contract
  codes vs `ethash_keccak256` (0..700 bytes, every source alignment).
- `tests/jumpdest_scan_test.cpp`: the guest-only sentinel JUMPDEST scan vs
  evmone's reference scan (200k codes).
- `tests/xlat_difffuzz.cpp`: fused cgoto vs plain switch dispatch (plus the
  parked xlat dispatch when `wip/0024` is applied).
- `tests/zeg_words_test.cpp`: the patched word paths (seeded hash, non-zero
  byte count, flat row index, fixed-size compare/copy at every alignment).

`tests/tamper.py HOST BLOCK_DIR` mutates inputs: targeted payload, header and
transaction mutations must abort, random bit flips must abort or reproduce the
exact output.

## Known gaps

- **EEST.** The archived [`eest/`](https://github.com/starknet-innovation/stwo-zig/tree/archive/pre-restructure-20261108/autoresearch/guest_efficiency/eest/README.md) harness runs the `tests@v21.0.2`
  blockchain tests (Prague, Osaka, BPO1, BPO2 and transitions) on the host and
  RV64 builds of this client's EEST variant (`build.py --eest`: fork and blob
  schedule from the schema id; never deployed). Receipt:
  [`eest-guests-v1.json`](https://github.com/starknet-innovation/stwo-zig/tree/archive/pre-restructure-20261108/vectors/reports/rv64-ethereum-20261101/eest-guests-v1.json) (archived). The run found
  and fixed two witness-dependence bugs (patch 0005: EIP-7702 delegation code
  registered at authorization; `glue/stwo_client.cpp`: trie inserts before
  deletions). Still open: table pre-sizing from the gas limit fails on synthetic
  near-2^63 gas limits, and a max-depth EIP-7702 call stack exhausts memory.
- **Forks.** Only the `ElectraFulu` request shape (Prague, Osaka, BPO1, BPO2,
  mainnet schedule, chain id 1). Amsterdam (`Gloas`) inputs are rejected.
- **Hash-map seed.** ziskethone's seeded table hash uses the payload's block
  hash; a block author could grind it to slow lookups (cost, not soundness).
- **Lazy rows.** Table vectors are pre-sized from the gas limit; a block with
  far more probes than that reallocates them (correct for the code paths
  exercised, not audited for every reference held across a load).
- **EIP-161 corner.** A pre-existing empty account that is touched but not
  modified is not deleted (none exist on mainnet since 2016).
- **Heap.** 212 MiB window; ziskethone's JUMPDEST analysis cache reserves
  85 MiB of it (1152 slots, sized for full call depth).
- **Candidate costs** are the sibling track's nominal estimates
  (`codex/rv64-guest-candidate-services`, [`proposals/candidate-service-costs.json`](https://github.com/starknet-innovation/stwo-zig/tree/archive/pre-restructure-20261108/autoresearch/guest_efficiency/proposals/candidate-service-costs.json));
  the archived [`calibration.json`](https://github.com/starknet-innovation/stwo-zig/tree/archive/pre-restructure-20261108/autoresearch/guest_efficiency/calibration.json) has none, so candidate runs used a
  scratch copy of the harness with those entries.
