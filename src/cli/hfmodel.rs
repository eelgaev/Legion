//! `--hfmodel <org/name>[@<revision>]`: name a model the way Hugging Face does, and load it
//! from the local Hugging Face cache, offline. Resolves to the snapshot directory that a
//! previous `huggingface-cli download` (or `hf download`) left in the cache; it then loads
//! exactly as `--model <dir>` would (a safetensors checkpoint, or the GGUF files inside the
//! snapshot). Legion never downloads and never reads tokens.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::Context;

use super::model_path;

/// The model file or checkpoint directory that `--hfmodel <org/name>[@<revision>]` names in
/// the user's Hugging Face cache.
pub(super) fn resolve(spec: &str) -> anyhow::Result<PathBuf> {
    let env = CacheEnv {
        hf_hub_cache: std::env::var_os("HF_HUB_CACHE"),
        huggingface_hub_cache: std::env::var_os("HUGGINGFACE_HUB_CACHE"),
        hf_home: std::env::var_os("HF_HOME"),
    };
    // The home directory is only needed when no variable names the cache.
    let home = std::env::home_dir().unwrap_or_default();
    let root = cache_root_from(&env, &home);
    anyhow::ensure!(
        root.is_absolute() || !home.as_os_str().is_empty(),
        "cannot find the home directory for the default Hugging Face cache; set HF_HUB_CACHE"
    );
    resolve_in(&root, spec)
}

/// The repo id (`org/name`) of a `--hfmodel` value: the default model id `/v1/models`
/// reports, as the snapshot's file name is a commit hash.
pub(super) fn repo_id(spec: &str) -> Option<String> {
    parse(spec).ok().map(|(repo, _)| repo)
}

/// The cache-relevant environment variables, read once (`huggingface_hub`'s names).
struct CacheEnv {
    hf_hub_cache: Option<OsString>,
    huggingface_hub_cache: Option<OsString>,
    hf_home: Option<OsString>,
}

/// The Hugging Face hub cache, in `huggingface_hub`'s precedence: `$HF_HUB_CACHE`, else
/// `$HUGGINGFACE_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`.
fn cache_root_from(env: &CacheEnv, home: &Path) -> PathBuf {
    if let Some(path) = env.hf_hub_cache.as_ref().filter(|path| !path.is_empty()) {
        return path.into();
    }
    if let Some(path) = env
        .huggingface_hub_cache
        .as_ref()
        .filter(|path| !path.is_empty())
    {
        return path.into();
    }
    if let Some(hf_home) = env.hf_home.as_ref().filter(|path| !path.is_empty()) {
        return PathBuf::from(hf_home).join("hub");
    }
    home.join(".cache").join("huggingface").join("hub")
}

/// Resolves against a given cache root (a test builds one in a temp dir).
fn resolve_in(root: &Path, spec: &str) -> anyhow::Result<PathBuf> {
    let (repo, revision) = parse(spec)?;
    let fetch = fetch_command(&repo, revision.as_deref());
    let repo_dir = root.join(format!("models--{}", repo.replace('/', "--")));
    if !repo_dir.is_dir() {
        anyhow::bail!(
            "{repo} is not in the Hugging Face cache: no {};\nfetch it with `{fetch}` \
             (Legion itself never downloads)",
            repo_dir.display()
        );
    }
    let snapshot = resolve_revision(&repo_dir, &repo, revision.as_deref().unwrap_or("main"))?;
    tracing::info!("--hfmodel {repo}: snapshot {}", snapshot.display());
    if !has_model_files(&snapshot) {
        return Err(empty_snapshot_error(&snapshot, &fetch));
    }
    let resolved = model_path::resolve_model_path(&snapshot)?;
    // A download stopped part way leaves the small files (config.json, the index, the first
    // shard) in the snapshot and the rest as `.incomplete` blobs: say so here rather than let
    // the loader fail on the first missing shard.
    let missing = missing_files(&resolved);
    if !missing.is_empty() {
        let shown: Vec<&str> = missing.iter().take(3).map(String::as_str).collect();
        anyhow::bail!(
            "{}: the download did not finish: {} file(s) the model needs are not in the \
             snapshot ({}{}; {} .incomplete blobs); re-run `{fetch}` to complete it",
            snapshot.display(),
            missing.len(),
            shown.join(", "),
            if missing.len() > shown.len() {
                ", ..."
            } else {
                ""
            },
            incomplete_blobs(&snapshot),
        );
    }
    Ok(resolved)
}

/// The files a resolved model needs that its snapshot lacks: the shards a
/// `model.safetensors.index.json` names, or the later shards of a split GGUF. An index that
/// does not parse is left to the loader to report.
fn missing_files(resolved: &Path) -> Vec<String> {
    if resolved.is_dir() {
        let index = resolved.join("model.safetensors.index.json");
        let Some(index) = std::fs::read(&index)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        else {
            return Vec::new();
        };
        let mut files: Vec<String> = index
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .map(|map| {
                map.values()
                    .filter_map(serde_json::Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        files.sort();
        files.dedup();
        files.retain(|file| !resolved.join(file).is_file());
        return files;
    }
    // `<name>-00001-of-MMMMM.gguf`: every shard up to MMMMM.
    let (Some(dir), Some(name)) = (resolved.parent(), resolved.file_name()) else {
        return Vec::new();
    };
    let name = name.to_string_lossy();
    let Some(stem) = name.strip_suffix(".gguf") else {
        return Vec::new();
    };
    if model_path::shard_sequence(stem) != Some("00001") {
        return Vec::new();
    }
    let (prefix, total) = stem.split_at(stem.len() - 5);
    let prefix = &prefix[..prefix.len() - "00001-of-".len()];
    let total: u32 = total.parse().unwrap_or(0);
    (2..=total)
        .map(|seq| format!("{prefix}{seq:05}-of-{total:05}.gguf"))
        .filter(|shard| !dir.join(shard).is_file())
        .collect()
}

/// The number of `.incomplete` blobs (downloads in progress or stopped) of the repo a
/// snapshot belongs to (`<repo>/snapshots/<commit>`).
fn incomplete_blobs(snapshot: &Path) -> usize {
    snapshot
        .ancestors()
        .nth(2)
        .and_then(|repo_dir| std::fs::read_dir(repo_dir.join("blobs")).ok())
        .map(|blobs| {
            blobs
                .filter_map(|blob| blob.ok())
                .filter(|blob| blob.file_name().to_string_lossy().ends_with(".incomplete"))
                .count()
        })
        .unwrap_or(0)
}

/// The `huggingface-cli download` line that would fetch what is missing (the newer `hf`
/// client takes the same arguments).
fn fetch_command(repo: &str, revision: Option<&str>) -> String {
    let revision = revision
        .map(|rev| format!(" --revision {rev}"))
        .unwrap_or_default();
    format!("huggingface-cli download {repo}{revision} (or: hf download {repo}{revision})")
}

/// `<org/name>` and the optional `@<revision>` of a `--hfmodel` value. The revision may be a
/// branch, tag or commit hash; slashes inside it (a branch like `releases/v1`) are fine, but
/// it must stay within `refs/`: no `..`, empty or absolute components.
fn parse(spec: &str) -> anyhow::Result<(String, Option<String>)> {
    let spec = spec.trim();
    let (repo, revision) = match spec.split_once('@') {
        Some((repo, revision)) => (repo, Some(revision)),
        None => (spec, None),
    };
    let expected = || format!("--hfmodel expects <org/name>[@<revision>], got {spec:?}");
    let Some((org, name)) = repo.split_once('/') else {
        anyhow::bail!(
            "{}: no '/' between the organization and the repo name",
            expected()
        );
    };
    anyhow::ensure!(
        !org.is_empty()
            && !name.is_empty()
            && !name.contains('/')
            && !repo.chars().any(char::is_whitespace),
        "{}: the organization and repo name must be non-empty, without whitespace or \
         a second '/'",
        expected()
    );
    if let Some(revision) = revision {
        anyhow::ensure!(
            !revision.is_empty()
                && !revision.chars().any(|c| c.is_whitespace() || c == '\\')
                && !revision
                    .split('/')
                    .any(|part| part.is_empty() || part == "." || part == ".."),
            "{}: the revision must be a plain name or commit hash (a branch may use '/')",
            expected()
        );
    }
    Ok((repo.to_string(), revision.map(str::to_string)))
}

/// The snapshot directory a repo's revision points at: the commit hash written in
/// `refs/<revision>` (what a branch or tag name resolves through), else a full or
/// abbreviated commit hash naming a `snapshots/` entry.
fn resolve_revision(repo_dir: &Path, repo: &str, revision: &str) -> anyhow::Result<PathBuf> {
    let refs = repo_dir.join("refs").join(revision);
    match std::fs::read_to_string(&refs) {
        // An empty refs file (a broken cache) is treated like a missing one below.
        Ok(commit) if !commit.trim().is_empty() => {
            let commit = commit.trim();
            let snapshot = repo_dir.join("snapshots").join(commit);
            if snapshot.is_dir() {
                return Ok(snapshot);
            }
            anyhow::bail!(
                "{repo}: revision {revision} points at commit {commit}, but there is no \
                 snapshot for it in the cache ({}); re-run the download to fetch it",
                repo_dir.join("snapshots").display()
            )
        }
        Ok(_) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err).with_context(|| format!("reading {}", refs.display())),
    }
    // No `refs/<revision>`: the revision must be a commit hash, full or abbreviated.
    let snapshots_dir = repo_dir.join("snapshots");
    let mut snapshots: Vec<String> = std::fs::read_dir(&snapshots_dir)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    snapshots.sort();
    if let Some(exact) = snapshots.iter().find(|name| name.as_str() == revision) {
        return Ok(snapshots_dir.join(exact));
    }
    let matches: Vec<&String> = snapshots
        .iter()
        .filter(|name| name.starts_with(revision))
        .collect();
    match matches.as_slice() {
        [one] => Ok(snapshots_dir.join(one)),
        [] => anyhow::bail!(
            "{repo}: revision {revision} is not in the cache: no refs/{revision} and no \
             snapshot named or starting with it; revisions in the cache: {}",
            recorded_revisions(repo_dir),
        ),
        several => anyhow::bail!(
            "{repo}: revision {revision} matches {} snapshots ({}); pass the full commit \
             hash: --hfmodel {repo}@<hash>",
            several.len(),
            several
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// The revisions a repo's `refs/` records, for the error message.
fn recorded_revisions(repo_dir: &Path) -> String {
    let mut names: Vec<String> = std::fs::read_dir(repo_dir.join("refs"))
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    if names.is_empty() {
        "none".to_string()
    } else {
        names.join(", ")
    }
}

/// Whether a snapshot holds anything loadable: a safetensors checkpoint (`config.json` plus
/// `model.safetensors` or its index) or a model `.gguf` file (not a vision projector, not a
/// later shard of a split GGUF). Snapshot files are symlinks into `blobs/`, so follow them.
fn has_model_files(snapshot: &Path) -> bool {
    if snapshot.join("config.json").is_file()
        && (snapshot.join("model.safetensors").is_file()
            || snapshot.join("model.safetensors.index.json").is_file())
    {
        return true;
    }
    !gguf_models(snapshot).is_empty()
}

/// The model `.gguf` files directly in `dir` (following symlinks): vision projectors
/// (`mmproj*`) and the later shards of a split GGUF are skipped.
fn gguf_models(dir: &Path) -> Vec<PathBuf> {
    let mut models: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| {
                    let name = path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    path.is_file()
                        && name.strip_suffix(".gguf").is_some_and(|stem| {
                            model_path::shard_sequence(stem).is_none_or(|seq| seq == "00001")
                        })
                        && !name
                            .get(..6)
                            .is_some_and(|p| p.eq_ignore_ascii_case("mmproj"))
                })
                .collect()
        })
        .unwrap_or_default();
    models.sort();
    models
}

/// A snapshot without anything loadable at its top level: the download did not finish (its
/// blobs are still `.incomplete`), the GGUF files sit in per-quantization subdirectories
/// (listed, to pass one with `--model`), or the repo holds a layout `--model <dir>` does not
/// take.
fn empty_snapshot_error(snapshot: &Path, fetch: &str) -> anyhow::Error {
    let incomplete = incomplete_blobs(snapshot);
    let mut nested: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| path.is_dir())
                .flat_map(|dir| gguf_models(&dir))
                .collect()
        })
        .unwrap_or_default();
    nested.sort();
    if !nested.is_empty() {
        let list: Vec<String> = nested.iter().map(|p| p.display().to_string()).collect();
        anyhow::anyhow!(
            "{}: the GGUF models are in subdirectories; pick one and pass it with \
             --model <path>: {}",
            snapshot.display(),
            list.join(", ")
        )
    } else if incomplete > 0 {
        anyhow::anyhow!(
            "{}: the snapshot has no model files yet (the download did not finish: {} \
             .incomplete blobs); re-run `{fetch}` to complete it",
            snapshot.display(),
            incomplete
        )
    } else {
        anyhow::anyhow!(
            "{}: the snapshot holds no model files (a safetensors checkpoint, or .gguf \
             files); pass --model <path> if the repo uses an unusual layout",
            snapshot.display()
        )
    }
}

#[cfg(test)]
mod tests;
