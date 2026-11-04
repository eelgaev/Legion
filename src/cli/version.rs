//! The build identity printed by `--version` and logged once at `serve` startup.
//! The values are baked in by `build.rs`; the hashes are `unknown` in tarball
//! builds without git metadata.

use std::sync::LazyLock;

/// e.g. `0.1.0 (abc1234, ggml 1a2b3c4, cuda off)`. Built once and kept alive,
/// so clap can take it as a `&'static str`.
static VERSION: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{} ({}, ggml {}, cuda {})",
        env!("CARGO_PKG_VERSION"),
        env!("LEGION_GIT_HASH"),
        env!("LEGION_GGML_HASH"),
        env!("LEGION_CUDA"),
    )
});

pub(crate) fn full() -> &'static str {
    &VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A short `git rev-parse --short` hash, or `unknown` without git metadata.
    fn is_hash(s: &str) -> bool {
        s == "unknown" || (s.len() >= 4 && s.chars().all(|c| c.is_ascii_hexdigit()))
    }

    #[test]
    fn version_string_has_the_documented_shape() {
        let v = full();
        assert!(
            v.starts_with(concat!(env!("CARGO_PKG_VERSION"), " (")),
            "{v}"
        );
        assert!(v.ends_with(')'), "{v}");
        let inner = &v[v.find('(').unwrap() + 1..v.len() - 1];
        let parts: Vec<&str> = inner.split(", ").collect();
        assert_eq!(parts.len(), 3, "{v}");
        assert!(is_hash(parts[0]), "bad legion hash in {v}");
        let ggml = parts[1]
            .strip_prefix("ggml ")
            .unwrap_or_else(|| panic!("no \"ggml \" in {v}"));
        assert!(is_hash(ggml), "bad ggml hash in {v}");
        assert_eq!(parts[2], format!("cuda {}", env!("LEGION_CUDA")), "{v}");
    }

    #[test]
    fn cuda_flag_matches_the_feature() {
        let want = if cfg!(feature = "cuda") { "on" } else { "off" };
        assert!(full().ends_with(&format!("cuda {want})")), "{}", full());
    }
}
