//! Split discovery follows llama.cpp's MIT-licensed `src/llama.cpp`
//! (`llama_split_path` / `llama_split_prefix`) and `src/llama-model-loader.cpp`:
//! filenames are one-based, while `split.no` is zero-based.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use super::{GgufError, GgufFile};

impl GgufFile {
    pub(super) fn shards(&self) -> impl Iterator<Item = &Self> {
        std::iter::once(self).chain(&self.siblings)
    }

    /// Number of files backing this model.
    pub fn shard_count(&self) -> usize {
        match &self.synth {
            Some(s) => s.files.len(),
            None => 1 + self.siblings.len(),
        }
    }

    /// Paths in shard order; [`super::TensorInfo::file_index`] indexes this sequence.
    pub fn shard_paths(&self) -> Box<dyn Iterator<Item = &Path> + '_> {
        match &self.synth {
            Some(s) => Box::new(s.files.iter().map(PathBuf::as_path)),
            None => Box::new(self.shards().map(Self::path)),
        }
    }
}

fn invalid(file: &GgufFile, reason: impl Into<String>) -> GgufError {
    GgufError::Split {
        path: file.path.clone(),
        reason: reason.into(),
    }
}

fn integer(file: &GgufFile, key: &str) -> Result<u64, GgufError> {
    file.get_u64(key).ok_or_else(|| {
        invalid(
            file,
            format!("missing or invalid `{key}` (expected a nonnegative integer)"),
        )
    })
}

fn expect(file: &GgufFile, key: &str, expected: u64) -> Result<(), GgufError> {
    let got = integer(file, key)?;
    if got != expected {
        return Err(invalid(
            file,
            format!("`{key}` is {got}, expected {expected}"),
        ));
    }
    Ok(())
}

pub(super) fn open(path: &Path) -> Result<GgufFile, GgufError> {
    let selected = GgufFile::open_single(path)?;
    let count = match selected.get("split.count") {
        None => return Ok(selected),
        Some(_) => integer(&selected, "split.count")?,
    };
    if count <= 1 {
        return Ok(selected);
    }
    if count > u16::MAX.into() {
        return Err(invalid(
            &selected,
            "`split.count` exceeds the GGUF u16 limit",
        ));
    }
    let index = integer(&selected, "split.no")?;
    if index >= count {
        return Err(invalid(
            &selected,
            format!("`split.no` {index} is outside `split.count` {count}"),
        ));
    }
    let suffix = format!("-{:05}-of-{count:05}.gguf", index + 1);
    let prefix = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_suffix(&suffix))
        .filter(|p| !p.is_empty())
        .ok_or_else(|| invalid(&selected, format!("filename must end with `{suffix}`")))?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let mut selected = Some(selected);
    let mut shards = Vec::new();
    for i in 0..count {
        let shard = if i == index {
            selected.take().expect("selected shard used once")
        } else {
            GgufFile::open_single(
                &parent.join(format!("{prefix}-{:05}-of-{count:05}.gguf", i + 1)),
            )?
        };
        expect(&shard, "split.no", i)?;
        expect(&shard, "split.count", count)?;
        shards.push(shard);
    }

    let total = integer(&shards[0], "split.tensors.count")?;
    let mut names = HashSet::new();
    for shard in &shards {
        expect(shard, "split.tensors.count", total)?;
        for tensor in shard.tensors() {
            if !names.insert(tensor.name.clone()) {
                return Err(invalid(
                    shard,
                    format!("duplicate tensor `{}` across shards", tensor.name),
                ));
            }
        }
    }
    if names.len() as u64 != total {
        return Err(invalid(
            &shards[0],
            format!(
                "`split.tensors.count` is {total}, found {} tensors across {count} shards",
                names.len()
            ),
        ));
    }
    let mut shards = shards.into_iter();
    let mut first = shards.next().expect("split model has at least two shards");
    first.siblings = shards.collect();
    Ok(first)
}
