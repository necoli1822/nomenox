//! nomenox — pure-Rust protein-function classifier for hypothetical proteins.
//!
//! A compact convolutional network that reads a protein sequence and predicts Pfam
//! family membership over a fixed vocabulary, for CDS that have **no homology hit**
//! (otherwise-"hypothetical" proteins). It was **distilled from an ESM-2 teacher**
//! (frozen embeddings + MLP head over the vocabulary) into a small sequence CNN so it
//! can run pure-Rust on CPU; per-family **calibrated thresholds** mean it only emits a
//! call when confident and otherwise stays silent (leaving the protein hypothetical —
//! it never fabricates a function). Architecture is our own, trained from scratch on
//! public UniProt/SwissProt (CC-BY); no external tool's code or weights are used.
//!
//! Inference is self-contained: weights embedded at compile time, every op hand-rolled
//! (one-hot encode, `conv1d` + ReLU, global-average pool, linear head, per-family
//! sigmoid + threshold).
//!
//! ```
//! let m = nomenox::Model::embedded();
//! for call in m.predict(b"MSKIVKIIGREIIDSRGNPTVEAEVHLEGGFVGMAAAPSGASTGSRE") {
//!     println!("{} {} ({:.2})", call.pfam, call.name, call.score);
//! }
//! ```

use serde::Deserialize;
use std::sync::OnceLock;

#[derive(Deserialize)]
struct VocabEntry {
    pfam: String,
    #[serde(default)]
    name: String,
}

#[derive(Deserialize)]
struct Weights {
    #[serde(rename = "L")]
    l: usize,
    #[serde(rename = "C")]
    c: usize,
    k: usize,
    nsym: usize,
    aa_order: String,
    vocab_size: usize,
    conv1_w: Vec<Vec<Vec<f32>>>, // [C][nsym][k]  (nsym = 21 input channels)
    conv1_b: Vec<f32>,
    conv2_w: Vec<Vec<Vec<f32>>>, // [C][C][k]
    conv2_b: Vec<f32>,
    conv3_w: Vec<Vec<Vec<f32>>>,
    conv3_b: Vec<f32>,
    conv4_w: Vec<Vec<Vec<f32>>>,
    conv4_b: Vec<f32>,
    head_w: Vec<Vec<f32>>,  // [vocab_size][C]
    head_b: Vec<f32>,       // [vocab_size]
    thresholds: Vec<f32>,   // [vocab_size] per-family decision boundary
    vocab: Vec<VocabEntry>, // [vocab_size]
}

/// A confident family call for a protein.
#[derive(Debug, Clone)]
pub struct Call {
    /// Pfam accession (e.g. "PF04055").
    pub pfam: String,
    /// Family name (may be empty if not recorded in the vocabulary).
    pub name: String,
    /// Sigmoid probability for this family (exceeds the family's calibrated threshold).
    pub score: f32,
}

/// A loaded nomenox model.
pub struct Model {
    w: Weights,
    /// aa byte -> index 1..=20 (0 = pad, 21 = other/unknown), matching training.
    aa_to_idx: [usize; 256],
}

/// Trained weights, gzip-compressed and embedded at compile time (kept under the
/// crates.io size limit; decompressed once in [`Model::embedded`]).
const EMBEDDED_WEIGHTS_GZ: &[u8] = include_bytes!("../weights.json.gz");

impl Model {
    /// The model backed by the embedded distilled weights (parsed once, cached).
    pub fn embedded() -> &'static Model {
        static M: OnceLock<Model> = OnceLock::new();
        M.get_or_init(|| {
            use std::io::Read;
            let mut s = String::new();
            flate2::read::GzDecoder::new(EMBEDDED_WEIGHTS_GZ)
                .read_to_string(&mut s)
                .expect("decompress embedded weights");
            Model::from_json(&s).expect("embedded weights parse")
        })
    }

    /// Load a model from a `weights.json` string.
    pub fn from_json(s: &str) -> Result<Model, String> {
        let w: Weights = serde_json::from_str(s).map_err(|e| e.to_string())?;
        // training: AA_ORDER chars -> 1..=20, unknown/other -> nsym (=21), pad -> 0.
        let other = w.nsym; // 21
        let mut aa_to_idx = [other; 256];
        for (i, c) in w.aa_order.bytes().enumerate() {
            aa_to_idx[c as usize] = i + 1;
            aa_to_idx[c.to_ascii_lowercase() as usize] = i + 1;
        }
        Ok(Model { aa_to_idx, w })
    }

    /// The vocabulary size (number of Pfam families the model can call).
    pub fn vocab_size(&self) -> usize {
        self.w.vocab_size
    }

    /// Predict confident Pfam family calls for `aa`, sorted by score descending.
    /// Only families whose probability exceeds their calibrated threshold are returned;
    /// an empty result means "leave hypothetical" (the model is not confident).
    pub fn predict(&self, aa: &[u8]) -> Vec<Call> {
        let l = self.w.l;
        let c = self.w.c;
        let nch = self.w.nsym; // 21 one-hot channels (indices 1..=21 -> channels 0..=20)

        // encode indices [L] (0 = pad), first L residues
        let mut idx = vec![0usize; l];
        for t in 0..l.min(aa.len()) {
            idx[t] = self.aa_to_idx[aa[t] as usize];
        }
        // one-hot [nch][L]: index j in 1..=nch -> channel j-1; index 0 (pad) -> none
        let mut oh = vec![0.0f32; nch * l];
        for t in 0..l {
            let j = idx[t];
            if j >= 1 && j <= nch {
                oh[(j - 1) * l + t] = 1.0;
            }
        }
        // 4 conv+ReLU (same-pad)
        let h1 = conv1d_relu(&oh, nch, l, &self.w.conv1_w, &self.w.conv1_b, c, self.w.k);
        let h2 = conv1d_relu(&h1, c, l, &self.w.conv2_w, &self.w.conv2_b, c, self.w.k);
        let h3 = conv1d_relu(&h2, c, l, &self.w.conv3_w, &self.w.conv3_b, c, self.w.k);
        let h = conv1d_relu(&h3, c, l, &self.w.conv4_w, &self.w.conv4_b, c, self.w.k);

        // global average pool over the full length L (matches training)
        let mut pooled = vec![0.0f32; c];
        for ci in 0..c {
            let mut s = 0.0;
            let base = ci * l;
            for t in 0..l {
                s += h[base + t];
            }
            pooled[ci] = s / l as f32;
        }
        // vocab head + per-family sigmoid/threshold
        let mut calls = Vec::new();
        for v in 0..self.w.vocab_size {
            let mut logit = self.w.head_b[v];
            let row = &self.w.head_w[v];
            for ci in 0..c {
                logit += row[ci] * pooled[ci];
            }
            let p = 1.0 / (1.0 + (-logit).exp());
            if p > self.w.thresholds[v] {
                calls.push(Call {
                    pfam: self.w.vocab[v].pfam.clone(),
                    name: self.w.vocab[v].name.clone(),
                    score: p,
                });
            }
        }
        calls.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        calls
    }

    /// Predict for many sequences in parallel (rayon). Same results as mapping
    /// [`Model::predict`], one `Vec<Call>` per input in order.
    pub fn predict_many(&self, seqs: &[&[u8]]) -> Vec<Vec<Call>> {
        use rayon::prelude::*;
        seqs.par_iter().map(|s| self.predict(s)).collect()
    }
}

fn conv1d_relu(
    inp: &[f32],
    cin: usize,
    l: usize,
    w: &[Vec<Vec<f32>>],
    b: &[f32],
    cout: usize,
    k: usize,
) -> Vec<f32> {
    let pad = k / 2;
    let mut out = vec![0.0f32; cout * l];
    for co in 0..cout {
        let wco = &w[co];
        let bias = b[co];
        for t in 0..l {
            let mut acc = bias;
            for ci in 0..cin {
                let wci = &wco[ci];
                let base = ci * l;
                for dk in 0..k {
                    let ii = t as isize + dk as isize - pad as isize;
                    if ii >= 0 && (ii as usize) < l {
                        acc += wci[dk] * inp[base + ii as usize];
                    }
                }
            }
            out[co * l + t] = if acc > 0.0 { acc } else { 0.0 };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_loads_and_predicts() {
        let m = Model::embedded();
        assert_eq!(m.vocab_size(), 1000);
        // structural sanity: prediction runs and calls (if any) are sorted desc.
        let calls = m.predict(b"MSKIVKIIGREIIDSRGNPTVEAEVHLEGGFVGMAAAPSGASTGSREALELRDGDKSRFLGKG");
        for w in calls.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }
}
