use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use clap::Subcommand;
use legion::safetensors::{
    Checkpoint, QuantMethod, SafetensorsFile, TensorInfo, detect_quant_method,
};
use serde_json::json;

#[derive(Subcommand)]
pub enum StCommand {
    /// Print a safetensors file's or checkpoint directory's tensors, dtypes and sizes.
    Inspect {
        /// A `.safetensors` file, or a model directory (single file or sharded index).
        path: PathBuf,
        /// List every tensor.
        #[arg(long)]
        tensors: bool,
        /// Machine-readable output.
        #[arg(long)]
        json: bool,
    },
}

pub fn run(cmd: StCommand) -> anyhow::Result<()> {
    match cmd {
        StCommand::Inspect {
            path,
            tensors,
            json,
        } => inspect(&path, tensors, json),
    }
}

struct Summary {
    files: Vec<String>,
    tensors: Vec<TensorInfo>,
    config: Option<serde_json::Value>,
    metadata: BTreeMap<String, String>,
    /// The method, or the reason it could not be determined. An unknown declared method is
    /// still worth listing, so a failure here does not abort the inspection.
    quant: Result<QuantMethod, String>,
}

fn inspect(path: &Path, list: bool, as_json: bool) -> anyhow::Result<()> {
    let s = if path.is_dir() {
        let c = Checkpoint::open(path)?;
        Summary {
            files: c
                .shards()
                .iter()
                .filter_map(|f| f.path().file_name())
                .map(|n| n.to_string_lossy().into_owned())
                .collect(),
            tensors: c.tensors(),
            config: c.config().cloned(),
            metadata: c
                .shards()
                .first()
                .map(|f| f.metadata().clone())
                .unwrap_or_default(),
            quant: c.quant_method().map_err(|e| e.to_string()),
        }
    } else {
        let f = SafetensorsFile::open(path)?;
        let tensors = f.tensors();
        let quant = detect_quant_method(None, &tensors)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        Summary {
            files: vec![path.display().to_string()],
            tensors,
            config: None,
            metadata: f.metadata().clone(),
            quant: Ok(quant),
        }
    };

    // dtype -> (tensors, elements, bytes)
    let mut by_dtype: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    for t in &s.tensors {
        let e = by_dtype.entry(t.dtype.to_string()).or_default();
        e.0 += 1;
        e.1 += t.numel();
        e.2 += t.nbytes();
    }
    let total_bytes: usize = by_dtype.values().map(|v| v.2).sum();
    let cfg_str = |k: &str| s.config.as_ref().and_then(|c| c.get(k)).cloned();

    if as_json {
        let dtypes: BTreeMap<_, _> = by_dtype
            .iter()
            .map(|(k, v)| (k, json!({"tensors": v.0, "elements": v.1, "bytes": v.2})))
            .collect();
        let quant = match &s.quant {
            Ok(q) => serde_json::to_value(q)?,
            Err(e) => json!({"error": e}),
        };
        let mut out = json!({
            "path": path.display().to_string(),
            "files": s.files,
            "tensor_count": s.tensors.len(),
            "total_bytes": total_bytes,
            "dtypes": dtypes,
            "metadata": s.metadata,
            "model_type": cfg_str("model_type"),
            "architectures": cfg_str("architectures"),
            "quantization": quant,
        });
        if list {
            out["tensors"] = serde_json::to_value(&s.tensors)?;
        }
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    let gib = |b: usize| b as f64 / (1u64 << 30) as f64;
    println!("path:          {}", path.display());
    println!("files:         {}", s.files.len());
    if let Some(v) = cfg_str("model_type") {
        println!("model_type:    {v}");
    }
    if let Some(v) = cfg_str("architectures") {
        println!("architectures: {v}");
    }
    match &s.quant {
        Ok(q) => println!("quantization:  {q}"),
        Err(e) => println!("quantization:  unknown ({e})"),
    }
    for (k, v) in &s.metadata {
        println!("metadata:      {k} = {v}");
    }
    println!(
        "tensors:       {} ({:.2} GiB)",
        s.tensors.len(),
        gib(total_bytes)
    );
    println!();
    println!(
        "{:<10} {:>8} {:>16} {:>10}",
        "dtype", "tensors", "elements", "GiB"
    );
    for (dtype, (n, elems, bytes)) in &by_dtype {
        println!("{dtype:<10} {n:>8} {elems:>16} {:>10.3}", gib(*bytes));
    }
    if list {
        println!();
        for t in &s.tensors {
            println!(
                "{:<64} {:<8} {:<24} {}",
                t.name,
                t.dtype,
                format!("{:?}", t.shape),
                t.nbytes()
            );
        }
    }
    Ok(())
}
