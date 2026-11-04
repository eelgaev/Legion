//! Resolve `--model`: a GGUF file, or a directory holding one (a download folder).

use std::path::{Path, PathBuf};

use anyhow::Context;

/// The five-digit sequence number of a split-GGUF shard (`name-NNNNN-of-MMMMM.gguf`,
/// without the extension), if the stem is one.
pub(super) fn shard_sequence(stem: &str) -> Option<&str> {
    let (rest, total) = stem.split_at(stem.len().checked_sub(5)?);
    if !total.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let rest = rest.strip_suffix("-of-")?;
    let (rest, seq) = rest.split_at(rest.len().checked_sub(5)?);
    if !seq.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    rest.strip_suffix('-')?;
    Some(seq)
}

/// `path` unchanged when it is not a directory (a missing file is the GGUF loader's
/// problem to report); when it is, the single `.gguf` model file inside it. Vision
/// projectors (`mmproj*`, case-insensitive) and the trailing shards of a split GGUF are
/// skipped; only the `-00001-of-MMMMM.gguf` shard is a candidate. A Hugging Face checkpoint
/// directory (`config.json` plus `model.safetensors` or its index) is returned as is: it is
/// loaded directly (see `legion::hf`).
pub(super) fn resolve_model_path(path: &Path) -> anyhow::Result<PathBuf> {
    if !path.is_dir() {
        return Ok(path.to_path_buf());
    }
    if path.join("config.json").is_file()
        && (path.join("model.safetensors").is_file()
            || path.join("model.safetensors.index.json").is_file())
    {
        tracing::info!("model: {} (safetensors checkpoint)", path.display());
        return Ok(path.to_path_buf());
    }
    let mut names: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(path).with_context(|| format!("reading {}", path.display()))? {
        let entry = entry?;
        // Hugging Face cache snapshots hold symlinks into blobs/, so follow them.
        if !entry.path().is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.ends_with(".gguf")
            || name
                .get(..6)
                .is_some_and(|p| p.eq_ignore_ascii_case("mmproj"))
        {
            continue;
        }
        let stem = name.strip_suffix(".gguf").unwrap();
        if let Some(seq) = shard_sequence(stem)
            && seq != "00001"
        {
            continue; // a later shard; the first one holds the model
        }
        names.push(name);
    }
    names.sort();
    match names.as_slice() {
        [] => anyhow::bail!(
            "{}: no .gguf model file in this directory (mmproj files are skipped)",
            path.display()
        ),
        [name] => {
            let resolved = path.join(name);
            tracing::info!("model: {}", resolved.display());
            Ok(resolved)
        }
        _ => anyhow::bail!(
            "{}: several GGUF models, pass one with --model: {}",
            path.display(),
            names.join(", ")
        ),
    }
}

#[cfg(test)]
mod tests;
