//! `legion bench --save <dir>` against `legion serve` (mock engine): the three files, their
//! contents, the refusal to overwrite a saved run, and stdout staying as without `--save`.

mod common;

use std::path::Path;
use std::process::{Child, Command, Output};

use serde_json::Value;

/// A `legion serve` on a port it bound itself; always killed when the guard drops.
struct Server {
    child: Child,
    port: u16,
}

impl Server {
    fn start() -> Server {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_legion"));
        cmd.args(["serve", "--host", "127.0.0.1"]);
        let (child, port) = common::serve::spawn(cmd);
        Server { child, port }
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn bench(url: &str, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_legion"))
        .args(["bench", "--url", url])
        .args(["--concurrency", "1,2", "--requests", "2"])
        .args(["--prompt-tokens", "20", "--gen-tokens", "8"])
        .args(extra)
        .output()
        .unwrap()
}

fn json_file(dir: &Path, name: &str) -> Value {
    let text = std::fs::read_to_string(dir.join(name)).unwrap();
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name}: {e}\n{text}"))
}

fn stdout(o: &Output) -> String {
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout.clone()).unwrap()
}

#[test]
fn save_writes_bench_meta_and_readme() {
    let server = Server::start();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("runs/a/b"); // parents are created
    let o = bench(
        &server.url(),
        &[
            "--json",
            "--save",
            out.to_str().unwrap(),
            "--api-key",
            "hunter2",
        ],
    );
    let printed: Value = serde_json::from_str(&stdout(&o)).unwrap();

    for f in ["bench.json", "meta.json", "README.md"] {
        assert!(out.join(f).is_file(), "{f} missing");
    }
    // bench.json is what --json printed.
    let saved = json_file(&out, "bench.json");
    assert_eq!(saved, printed);
    let reports = saved.as_array().unwrap();
    assert_eq!(reports.len(), 2);
    assert_eq!(reports[0]["concurrency"], 1);
    assert_eq!(reports[1]["ok"], 2);

    let meta = json_file(&out, "meta.json");
    assert_eq!(meta["url"], server.url());
    assert_eq!(meta["model"], "mock");
    assert_eq!(meta["api"], "openai");
    assert_eq!(meta["prompt_tokens"], 20);
    assert_eq!(meta["gen_tokens"], 8);
    assert_eq!(meta["shared_prefix_tokens"], 0);
    assert_eq!(meta["concurrency"], serde_json::json!([1, 2]));
    assert!(meta["server_health"].is_object(), "{meta}");
    assert!(
        meta["legion_version"]
            .as_str()
            .unwrap()
            .starts_with(env!("CARGO_PKG_VERSION"))
    );
    let date = meta["date"].as_str().unwrap();
    assert!(
        date.len() == 20 && date.ends_with('Z') && date.as_bytes()[10] == b'T',
        "{date}"
    );
    let argv: Vec<&str> = meta["argv"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_str().unwrap())
        .collect();
    assert!(
        argv.contains(&"bench") && argv.contains(&"--save"),
        "{argv:?}"
    );
    assert!(
        !argv.contains(&"hunter2"),
        "the API key was saved: {argv:?}"
    );

    let readme = std::fs::read_to_string(out.join("README.md")).unwrap();
    assert!(
        readme.starts_with(&format!("# legion bench mock @ {}\n", server.url())),
        "{readme}"
    );
    assert!(readme.contains(date), "{readme}");
    assert!(readme.contains("```sh\n"), "{readme}");
    assert!(!readme.contains("hunter2"), "{readme}");
    assert!(readme.contains(
        "| C | requests | ok | failed | out tok/s | per-user tok/s | TTFT p50 / p99 (ms) | ITL p50 / p99 (ms) |"
    ));
    let rows: Vec<&str> = readme.lines().filter(|l| l.starts_with("| ")).collect();
    assert_eq!(rows.len(), 1 + 2, "{readme}");
    assert!(rows[1].starts_with("| 1 | 2 | 2 | 0 | "), "{}", rows[1]);
    assert!(rows[2].starts_with("| 2 | 2 | 2 | 0 | "), "{}", rows[2]);
}

#[test]
fn existing_bench_json_is_refused_unless_forced() {
    let server = Server::start();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().to_str().unwrap();
    stdout(&bench(&server.url(), &["--save", out]));
    let first = std::fs::read_to_string(dir.path().join("meta.json")).unwrap();

    let o = bench(&server.url(), &["--save", out]);
    assert!(!o.status.success());
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(stderr.contains("already contains bench.json"), "{stderr}");
    assert!(o.stdout.is_empty(), "the refusal comes before the run");
    assert_eq!(
        std::fs::read_to_string(dir.path().join("meta.json")).unwrap(),
        first
    );

    stdout(&bench(&server.url(), &["--save", out, "--force"]));
    assert_ne!(json_file(dir.path(), "meta.json")["date"], Value::Null);
}

#[test]
fn stdout_is_the_same_with_and_without_save() {
    let server = Server::start();
    let dir = tempfile::tempdir().unwrap();
    let plain = stdout(&bench(&server.url(), &[]));
    let saved = stdout(&bench(
        &server.url(),
        &["--save", dir.path().to_str().unwrap()],
    ));
    // Timings differ run to run; the shape (header, one row per level, footer) does not.
    let shape = |s: &str| {
        s.lines()
            .map(|l| l.split_whitespace().count())
            .collect::<Vec<_>>()
    };
    assert_eq!(shape(&plain), shape(&saved));
    assert!(plain.trim_start().starts_with("conc "), "{plain}");
    let mut files: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    files.sort();
    assert_eq!(files, ["README.md", "bench.json", "meta.json"]);
    // Without --save nothing is written.
    let cwd = tempfile::tempdir().unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_legion"))
        .current_dir(cwd.path())
        .args(["bench", "--url", &server.url(), "--concurrency", "1"])
        .args([
            "--requests",
            "1",
            "--prompt-tokens",
            "20",
            "--gen-tokens",
            "4",
        ])
        .output()
        .unwrap();
    stdout(&o);
    assert_eq!(std::fs::read_dir(cwd.path()).unwrap().count(), 0);
}

#[test]
fn unreachable_server_saves_null_health() {
    let dir = tempfile::tempdir().unwrap();
    let (_held, port) = common::serve::refused_port();
    let url = format!("http://127.0.0.1:{port}");
    let o = bench(&url, &["--save", dir.path().to_str().unwrap()]);
    stdout(&o);
    let meta = json_file(dir.path(), "meta.json");
    assert_eq!(meta["server_health"], Value::Null);
    let reports = json_file(dir.path(), "bench.json");
    assert_eq!(reports[0]["failed"], 2);
}
