//! `legion completions <shell>` prints a completion script on stdout and nothing else.

use std::process::Command;

fn completions(shell: &str) -> String {
    let out = Command::new(env!("CARGO_BIN_EXE_legion"))
        .args(["completions", shell])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "`legion completions {shell}` failed: {out:?}"
    );
    assert!(
        out.stderr.is_empty(),
        "stderr must stay empty, got {:?}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// The script must be non-empty and mention the `serve` subcommand; stderr must
/// stay empty so the output can be redirected straight into a completion file.
#[test]
fn completions_prints_a_script_for_bash_and_fish() {
    for shell in ["bash", "fish"] {
        let script = completions(shell);
        assert!(!script.is_empty(), "{shell} script is empty");
        assert!(script.contains("serve"), "{shell} script lacks `serve`");
    }
}
