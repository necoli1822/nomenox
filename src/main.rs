//! nomenox CLI — assign Pfam families to protein FASTA (hypothetical proteins).
//!
//!   nomenox proteins.faa      # id<TAB>pfam<TAB>name<TAB>score  (best call per protein)
//!   nomenox --all proteins.faa   # all above-threshold calls per protein
//!   nomenox < proteins.faa

use nomenox::Model;
use std::io::{self, Read, Write};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let all = args.iter().any(|a| a == "--all");
    args.retain(|a| a != "--all");
    let input = match args.first() {
        Some(p) if p != "-" => std::fs::read_to_string(p).unwrap_or_else(|e| {
            eprintln!("nomenox: cannot read {p}: {e}");
            std::process::exit(1);
        }),
        _ => {
            let mut s = String::new();
            io::stdin().read_to_string(&mut s).ok();
            s
        }
    };
    let m = Model::embedded();
    let recs = fasta_iter(&input);
    let seqs: Vec<&[u8]> = recs.iter().map(|(_, s)| s.as_bytes()).collect();
    let preds = m.predict_many(&seqs);
    let out = io::stdout();
    let mut w = out.lock();
    let _ = writeln!(w, "#id\tpfam\tname\tscore");
    for ((id, _), calls) in recs.iter().zip(preds.iter()) {
        if calls.is_empty() {
            let _ = writeln!(w, "{id}\t-\thypothetical protein\t-");
        } else if all {
            for c in calls {
                let _ = writeln!(w, "{id}\t{}\t{}\t{:.3}", c.pfam, c.name, c.score);
            }
        } else {
            let c = &calls[0];
            let _ = writeln!(w, "{id}\t{}\t{}\t{:.3}", c.pfam, c.name, c.score);
        }
    }
}

fn fasta_iter(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let (mut id, mut seq) = (None::<String>, String::new());
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix('>') {
            if let Some(p) = id.take() {
                out.push((p, std::mem::take(&mut seq)));
            }
            id = Some(rest.split_whitespace().next().unwrap_or("").to_string());
        } else {
            seq.extend(line.split_whitespace());
        }
    }
    if let Some(p) = id.take() {
        out.push((p, seq));
    }
    out
}
