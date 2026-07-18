# nomenox

Pure-Rust **protein-function classifier for hypothetical proteins**. A compact CNN
reads a protein sequence and assigns Pfam family membership over a fixed vocabulary —
intended for CDS with **no homology hit** (otherwise-"hypothetical" proteins). It was
**distilled from an ESM-2 teacher** into a small sequence CNN so it runs pure-Rust on
CPU; per-family **calibrated thresholds** mean it only emits a call when confident and
otherwise stays silent (leaving the protein hypothetical — it never fabricates a
function). Self-contained, database-free, no ML runtime, no Python at inference time.

```rust
let m = nomenox::Model::embedded();
for call in m.predict(b"MKRVLKFGGTSVANAERFLRVADIL") {
    println!("{} {} ({:.2})", call.pfam, call.name, call.score);
}
```

CLI:

```
nomenox proteins.faa          # best confident Pfam call per protein (or "hypothetical")
nomenox --all proteins.faa    # every above-threshold call
#id       pfam     name  score
sp|P0A910 PF00691  -     0.94
```

## Status: proof of concept

The shippable student CNN is distilled from an ESM-2 teacher (teacher top-1 0.98). On a
held-out set the student reaches top-1 ≈ 0.36 overall, but with the calibrated thresholds
its **precision-on-calls is ≈ 0.87** (it only calls when confident, ≈ 13% coverage). It is
strictly additive and non-destructive — proteins it is unsure about are left hypothetical.
Transfer to truly novel orphan proteins is not yet validated at scale, so treat calls as
advisory.

## Model & provenance

`one_hot(21ch)` → 4 × `[Conv1d(k=7, C=128) → ReLU]` → global-average pool →
`Linear(C, 1000)` → per-family sigmoid + calibrated threshold. Architecture is our own,
trained from scratch on public UniProt/SwissProt (CC-BY 4.0) with Pfam labels; distilled
from a frozen-ESM-2 + MLP teacher. No external tool's source code or trained weights are
used. Weights are gzip-compressed and embedded in the crate; inference is pure-Rust and
verified to match the reference PyTorch student to 3 decimal places.

## Changelog

### 0.1.0
- Initial proof-of-concept release: distilled CNN Pfam-family classifier (1000 families),
  per-family calibrated thresholds, gzip-embedded weights, pure-Rust inference.

## License

MIT OR Apache-2.0. Training data derived from UniProt (CC-BY 4.0).
