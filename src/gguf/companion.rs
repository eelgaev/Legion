//! Companion files: tensors a model keeps in a GGUF of their own, which the main file's
//! metadata names and which sits next to it. DeepSeek-V4.1-Flash's GGUFs keep the Engram tables
//! (`blk.{l}.engram_embd.weight`, about 110 GB at IQ4_NL) in one file that every quantization of
//! the backbone shares: `{arch}.engram.file` names it, and `{arch}.engram.source` in both files
//! ties the tables to the checkpoint they were made from.
//!
//! An attached companion is one more file after the shards ([`GgufFile::shard_paths`]), so its
//! tensors are listed, placed, mapped and loaded like any shard's. A companion that is not there
//! is not an error when opening (tools read the main file alone); the architecture reports the
//! tensors it lacks, with the path it expected ([`companion_path`]).

#[cfg(test)]
mod tests;

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};

use super::{GgufError, GgufFile};

/// The metadata keys (after `{arch}.`) that name a companion file.
pub const FILE_KEYS: [&str; 1] = ["engram.file"];

/// The key (after `{arch}.`) both files may carry to show they belong together.
pub const SOURCE_KEY: &str = "engram.source";

fn invalid(path: &Path, reason: impl Into<String>) -> GgufError {
    GgufError::Companion {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

/// The file `{arch}.{key}` names for `model` (e.g. `key` = `engram.file`): a plain file name,
/// in the directory of the model's first shard. `None` when the model has no such key.
pub fn companion_path(model: &GgufFile, key: &str) -> Result<Option<PathBuf>, GgufError> {
    let Some(arch) = model.architecture() else {
        return Ok(None);
    };
    let full = format!("{arch}.{key}");
    let Some(name) = model.get_str(&full) else {
        return Ok(None);
    };
    let mut parts = Path::new(&name).components();
    if !matches!(
        (parts.next(), parts.next()),
        (Some(Component::Normal(_)), None)
    ) {
        return Err(invalid(
            model.path(),
            format!("`{full}` = `{name}` is not a plain file name"),
        ));
    }
    let dir = model.path().parent().unwrap_or_else(|| Path::new(""));
    Ok(Some(dir.join(name)))
}

/// Opens the companions `model` names and appends them to its files. A companion whose tensors
/// are all in the model already (tables kept in the main file too) is left out; one that
/// shares only some of them, or was made from another checkpoint, is an error.
pub(super) fn attach(mut model: GgufFile) -> Result<GgufFile, GgufError> {
    for key in FILE_KEYS {
        let Some(path) = companion_path(&model, key)? else {
            continue;
        };
        if !path.is_file() {
            continue;
        }
        let companion = GgufFile::open_single(&path)?;
        check(&model, &companion)?;
        let ours: HashSet<String> = model.tensors().into_iter().map(|t| t.name).collect();
        let theirs = companion.tensors();
        let shared: Vec<&str> = theirs
            .iter()
            .filter(|t| ours.contains(&t.name))
            .map(|t| t.name.as_str())
            .collect();
        if shared.len() == theirs.len() {
            continue;
        }
        if let Some(name) = shared.first() {
            return Err(invalid(
                &path,
                format!(
                    "`{name}` is in the model too ({} of its {} tensors are)",
                    shared.len(),
                    theirs.len()
                ),
            ));
        }
        model.siblings.push(companion);
    }
    Ok(model)
}

/// A companion is one file, made from the model's checkpoint when both say which.
fn check(model: &GgufFile, companion: &GgufFile) -> Result<(), GgufError> {
    if companion.get_u64("split.count").is_some_and(|n| n > 1) {
        return Err(invalid(
            companion.path(),
            "a split companion is not supported",
        ));
    }
    let Some(arch) = model.architecture() else {
        return Ok(());
    };
    let key = format!("{arch}.{SOURCE_KEY}");
    if let (Some(ours), Some(theirs)) = (model.get_str(&key), companion.get_str(&key))
        && ours != theirs
    {
        return Err(invalid(
            companion.path(),
            format!(
                "`{key}` is `{theirs}`, the model's (`{}`) is `{ours}`",
                model.path().display()
            ),
        ));
    }
    Ok(())
}
