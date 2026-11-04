//! Unit tests for `build/git_watch.rs`, the `cargo:rerun-if-changed` lines
//! `build.rs` prints so a new commit re-runs it. The module is shared with the
//! build script via `#[path]`, so these tests exercise the code that runs.

#[path = "../build/git_watch.rs"]
mod git_watch;

use std::fs;
use std::path::Path;

use tempfile::TempDir;

fn write(path: &Path, contents: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, contents).unwrap();
}

/// The watched paths, with the `cargo:rerun-if-changed=` prefix stripped.
fn watched(dir: &Path) -> Vec<String> {
    git_watch::rerun_lines(dir)
        .iter()
        .map(|l| l.strip_prefix("cargo:rerun-if-changed=").unwrap().into())
        .collect()
}

#[test]
fn branch_head_watches_head_and_its_loose_ref() {
    let tmp = TempDir::new().unwrap();
    let repo = tmp.path();
    write(&repo.join(".git/HEAD"), "ref: refs/heads/main\n");
    write(
        &repo.join(".git/refs/heads/main"),
        "0123456789abcdef0123456789abcdef01234567\n",
    );
    assert_eq!(
        watched(repo),
        [
            repo.join(".git/HEAD").display().to_string(),
            repo.join(".git/refs/heads/main").display().to_string(),
        ]
    );
}

#[test]
fn detached_head_only_watches_head() {
    let tmp = TempDir::new().unwrap();
    let repo = tmp.path();
    write(
        &repo.join(".git/HEAD"),
        "0123456789abcdef0123456789abcdef01234567\n",
    );
    assert_eq!(
        watched(repo),
        [repo.join(".git/HEAD").display().to_string()]
    );
}

#[test]
fn packed_ref_watches_packed_refs() {
    let tmp = TempDir::new().unwrap();
    let repo = tmp.path();
    write(&repo.join(".git/HEAD"), "ref: refs/heads/main\n");
    write(
        &repo.join(".git/packed-refs"),
        "0123456789abcdef0123456789abcdef01234567 refs/heads/main\n",
    );
    assert_eq!(
        watched(repo),
        [
            repo.join(".git/HEAD").display().to_string(),
            repo.join(".git/packed-refs").display().to_string(),
        ]
    );
}

#[test]
fn submodule_pointer_resolves_to_the_superproject_gitdir() {
    let tmp = TempDir::new().unwrap();
    let repo = tmp.path();
    // What a submodule checkout looks like: a `gitdir:` pointer file whose
    // target is relative to the submodule, with a detached HEAD there.
    write(
        &repo.join("third_party/ggml/.git"),
        "gitdir: ../../.git/modules/third_party/ggml\n",
    );
    let gitdir = repo.join(".git/modules/third_party/ggml");
    write(
        &gitdir.join("HEAD"),
        "0123456789abcdef0123456789abcdef01234567\n",
    );
    let got = watched(&repo.join("third_party/ggml"));
    assert_eq!(got, [gitdir.join("HEAD").display().to_string()]);
    assert!(!got[0].contains(".."), "pointer not resolved: {got:?}");
}

#[test]
fn submodule_branch_ref_is_watched_in_its_gitdir() {
    let tmp = TempDir::new().unwrap();
    let repo = tmp.path();
    write(&repo.join("sub/.git"), "gitdir: ../.git/modules/sub\n");
    let gitdir = repo.join(".git/modules/sub");
    write(&gitdir.join("HEAD"), "ref: refs/heads/main\n");
    write(
        &gitdir.join("refs/heads/main"),
        "0123456789abcdef0123456789abcdef01234567\n",
    );
    assert_eq!(
        watched(&repo.join("sub")),
        [
            gitdir.join("HEAD").display().to_string(),
            gitdir.join("refs/heads/main").display().to_string(),
        ]
    );
}

#[test]
fn absolute_gitdir_pointer_is_used_as_is() {
    let tmp = TempDir::new().unwrap();
    let repo = tmp.path();
    // Linked worktrees point at an absolute gitdir.
    let gitdir = tmp.path().join("elsewhere");
    write(
        &repo.join(".git"),
        &format!("gitdir: {}\n", gitdir.display()),
    );
    write(
        &gitdir.join("HEAD"),
        "0123456789abcdef0123456789abcdef01234567\n",
    );
    assert_eq!(watched(repo), [gitdir.join("HEAD").display().to_string()]);
}

#[test]
fn tarball_without_git_prints_nothing() {
    let tmp = TempDir::new().unwrap();
    assert!(watched(tmp.path()).is_empty());
    // A `.git` without a HEAD (empty repo) and a `.git` file that is not a
    // gitdir pointer are equally silent.
    fs::create_dir_all(tmp.path().join(".git")).unwrap();
    assert!(watched(tmp.path()).is_empty());
    fs::remove_dir_all(tmp.path().join(".git")).unwrap();
    write(&tmp.path().join(".git"), "not a gitdir pointer\n");
    assert!(watched(tmp.path()).is_empty());
}
