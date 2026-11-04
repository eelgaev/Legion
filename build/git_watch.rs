//! Git metadata watch list for `build.rs`, shared with `tests/git_watch.rs`
//! via `#[path]` so the tests exercise the exact code the build script runs.
//!
//! `build.rs` bakes `git rev-parse --short HEAD` into the binary, so it must
//! re-run whenever that hash can change: a commit rewrites `HEAD` (detached)
//! or the branch ref file `HEAD` points at (or `packed-refs` once the ref is
//! packed). Watching `.git` itself would rebuild on every `git add`, so these
//! lines name exactly the files a commit touches.

use std::path::{Component, Path, PathBuf};

/// `cargo:rerun-if-changed` lines for the git metadata that determines the
/// short hash of `dir`. Only existing files are listed, so tarball builds
/// without git metadata print nothing.
pub fn rerun_lines(dir: &Path) -> Vec<String> {
    let Some(gitdir) = gitdir(dir) else {
        return Vec::new();
    };
    let head = gitdir.join("HEAD");
    if !head.is_file() {
        return Vec::new();
    }
    let mut lines = vec![format!("cargo:rerun-if-changed={}", head.display())];
    // A detached HEAD holds the hash itself; a branch HEAD is `ref: <path>`,
    // and committing rewrites the ref file, not HEAD.
    let Ok(head_text) = std::fs::read_to_string(&head) else {
        return lines;
    };
    let Some(reference) = head_text.trim().strip_prefix("ref: ") else {
        return lines;
    };
    let reference = gitdir.join(reference.trim());
    // Once the ref is packed there is no loose file, and `git update-ref`
    // rewrites `packed-refs` instead.
    let watch = if reference.is_file() {
        reference
    } else {
        gitdir.join("packed-refs")
    };
    if watch.is_file() {
        lines.push(format!("cargo:rerun-if-changed={}", watch.display()));
    }
    lines
}

/// The git directory of `dir`: the `.git` directory of a normal checkout, or
/// the `gitdir: <path>` pointer target of a submodule or linked worktree,
/// resolved relative to `dir`. `None` when there is no usable `.git`.
fn gitdir(dir: &Path) -> Option<PathBuf> {
    let dot_git = dir.join(".git");
    if dot_git.is_dir() {
        return Some(normalize(&dot_git));
    }
    let pointer = std::fs::read_to_string(dot_git).ok()?;
    let path = PathBuf::from(pointer.trim().strip_prefix("gitdir:")?.trim());
    let path = if path.is_absolute() {
        path
    } else {
        dir.join(path)
    };
    Some(normalize(&path))
}

/// Lexically resolves `.` and `..` so the printed paths stay readable (the
/// submodule pointer is relative, e.g. `../../.git/modules/third_party/ggml`).
/// No symlink resolution: the result only feeds `cargo:rerun-if-changed`.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push(c);
                }
            }
            other => out.push(other),
        }
    }
    out
}
