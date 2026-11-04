//! `legion bench --workload chats|prefill` against `legion serve` (mock engine).

mod common;

use std::process::{Child, Command, Output};

use serde_json::Value;

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

fn stdout(o: &Output) -> String {
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8(o.stdout.clone()).unwrap()
}

#[test]
fn workload_chats_json_fields() {
    let server = Server::start();
    let o = Command::new(env!("CARGO_BIN_EXE_legion"))
        .args([
            "bench",
            "--url",
            &server.url(),
            "--workload",
            "chats",
            "--json",
        ])
        .output()
        .unwrap();
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let arr = v.as_array().expect("chats json is an array");
    assert_eq!(arr.len(), 3);
    for (i, rec) in arr.iter().enumerate() {
        assert!(rec["prompt"].as_str().unwrap().len() > 20, "prompt {i}");
        assert!(rec.get("prompt_tokens").is_some(), "prompt_tokens {i}");
        assert!(
            rec["completion_tokens"].as_u64().unwrap() >= 1,
            "completion {i}"
        );
        assert!(rec["chunks"].as_u64().unwrap() >= 1, "chunks {i}");
        assert!(rec["ttft_s"].as_f64().is_some(), "ttft_s {i}");
        assert!(rec["total_s"].as_f64().unwrap() > 0.0, "total_s {i}");
        // decode_tok_s may be null when the mock returns a single token.
        assert!(rec.get("decode_tok_s").is_some(), "decode_tok_s key {i}");
        assert!(rec["content"].is_string(), "content {i}");
        assert!(rec["reasoning"].is_string(), "reasoning {i}");
    }
}

#[test]
fn workload_prefill_json_fields() {
    let server = Server::start();
    let o = Command::new(env!("CARGO_BIN_EXE_legion"))
        .args([
            "bench",
            "--url",
            &server.url(),
            "--workload",
            "prefill",
            "--prompt-tokens",
            "512",
            "--json",
        ])
        .output()
        .unwrap();
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let arr = v.as_array().expect("prefill json is an array");
    assert_eq!(arr.len(), 1);
    let rec = &arr[0];
    assert_eq!(rec["n_arg"], 512);
    assert!(rec["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(rec["ttft_s"].as_f64().is_some());
    assert!(rec["prefill_tok_s"].as_f64().unwrap() > 0.0);
    assert!(rec["completion_tokens"].as_u64().unwrap() >= 1);
    assert!(rec["chunks"].as_u64().unwrap() >= 1);
    assert!(rec["total_s"].as_f64().unwrap() > 0.0);
}

#[test]
fn help_documents_workloads() {
    let o = Command::new(env!("CARGO_BIN_EXE_legion"))
        .args(["bench", "--help"])
        .output()
        .unwrap();
    let text = stdout(&o);
    assert!(text.contains("--workload"), "{text}");
    assert!(text.contains("chats"), "{text}");
    assert!(text.contains("prefill"), "{text}");
}

fn legion(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_legion"))
        .args(args)
        .output()
        .unwrap()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// A stub OpenAI server that answers every request with a stream whose only token was
/// swallowed (an empty delta, then usage and `[DONE]`), as a model whose single token is a
/// hidden `<think>` tag or EOS does with `max_tokens` 1.
fn swallowing_stub() -> String {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut conn) = conn else { continue };
            // Read the head, then the body by Content-Length, so the client sees a clean reply.
            let mut buf = Vec::new();
            let mut byte = [0u8; 1];
            while !buf.ends_with(b"\r\n\r\n") && conn.read(&mut byte).unwrap_or(0) == 1 {
                buf.push(byte[0]);
            }
            let head = String::from_utf8_lossy(&buf).to_ascii_lowercase();
            let len = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .map_or(0, |v| v.trim().parse().unwrap_or(0));
            let mut body = vec![0u8; len];
            let _ = conn.read_exact(&mut body);
            let sse = concat!(
                "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
                "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":42,\"completion_tokens\":1}}\n\n",
                "data: [DONE]\n\n",
            );
            let _ = write!(
                conn,
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{sse}",
                sse.len()
            );
        }
    });
    url
}

#[test]
fn prefill_point_without_a_token_does_not_abort_the_sweep() {
    let url = swallowing_stub();
    let dir = tempfile::tempdir().unwrap();
    let save = dir.path().join("run");
    let o = legion(&[
        "bench",
        "--url",
        &url,
        "--workload",
        "prefill",
        "--prompt-tokens",
        "100,200",
        "--concurrency",
        "4",
        "--json",
        "--save",
        save.to_str().unwrap(),
    ]);
    let v: Value = serde_json::from_str(&stdout(&o)).unwrap();
    let arr = v.as_array().unwrap();
    assert_eq!(arr.len(), 2, "{v}");
    for (rec, n) in arr.iter().zip([100, 200]) {
        assert_eq!(rec["n_arg"], n);
        assert_eq!(rec["prompt_tokens"], 42);
        assert!(rec["ttft_s"].is_null(), "{rec}");
        assert!(rec["prefill_tok_s"].is_null(), "{rec}");
    }
    let err = stderr(&o);
    assert!(
        err.contains("warning: --workload prefill ignores the bench-only flags --concurrency"),
        "{err}"
    );
    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(save.join("meta.json")).unwrap()).unwrap();
    assert_eq!(meta["workload"], "prefill");
    assert_eq!(meta["prompt_tokens"], serde_json::json!([100, 200]));
    assert!(meta["concurrency"].is_null(), "{meta}");
    assert!(meta["shared_prefix_tokens"].is_null(), "{meta}");
}

#[test]
fn chats_without_a_token_is_still_an_error() {
    let url = swallowing_stub();
    let o = legion(&["bench", "--url", &url, "--workload", "chats", "--json"]);
    assert!(!o.status.success());
    assert!(stderr(&o).contains("no tokens received"), "{}", stderr(&o));
}

#[test]
fn chats_meta_records_the_workload_not_fake_sizes() {
    let server = Server::start();
    let dir = tempfile::tempdir().unwrap();
    let o = legion(&[
        "bench",
        "--url",
        &server.url(),
        "--workload",
        "chats",
        "--save",
        dir.path().to_str().unwrap(),
    ]);
    stdout(&o);
    let meta: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("meta.json")).unwrap())
            .unwrap();
    assert_eq!(meta["workload"], "chats");
    assert!(meta["prompt_tokens"].is_null(), "{meta}");
    assert!(meta["concurrency"].is_null(), "{meta}");
    assert_eq!(meta["gen_tokens"], 384);
}

#[test]
fn bench_workload_refuses_a_list_of_prompt_tokens() {
    // Fails before any request, so no server is needed.
    let o = legion(&[
        "bench",
        "--url",
        "http://127.0.0.1:1",
        "--prompt-tokens",
        "100,200",
    ]);
    assert!(!o.status.success());
    let err = stderr(&o);
    assert!(
        err.contains("--prompt-tokens takes one value for --workload bench")
            && err.contains("--workload prefill"),
        "{err}"
    );
}

#[test]
fn save_refuses_a_directory_with_another_workloads_run() {
    // Every refusal happens before the first request, so no server is needed.
    let url = "http://127.0.0.1:1";
    let cases: &[(&[&str], &str)] = &[
        (&["--workload", "chats"], "bench.json"),
        (&["--workload", "prefill"], "chats.json"),
        (&["--workload", "chats"], "meta.json"),
        (&["--prompt-tokens", "20"], "chats.json"),
        (&["--prompt-tokens", "20"], "prefill.json"),
        (&["--prompt-tokens", "20"], "meta.json"),
    ];
    for (extra, existing) in cases {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(existing), "{}").unwrap();
        let mut args = vec![
            "bench",
            "--url",
            url,
            "--save",
            dir.path().to_str().unwrap(),
        ];
        args.extend_from_slice(extra);
        let o = legion(&args);
        assert!(!o.status.success(), "{extra:?} over {existing}");
        let err = stderr(&o);
        assert!(
            err.contains(existing) && err.contains("--force"),
            "{extra:?} over {existing}: {err}"
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join(existing)).unwrap(),
            "{}",
            "{existing} untouched"
        );
    }
}
