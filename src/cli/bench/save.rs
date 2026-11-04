//! `legion bench --save <dir>`: `bench.json` (the `--json` output), `meta.json` (how the run
//! was made, and the server's `/health` at the end) and a `README.md` with the table.

use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, bail};
use legion::bench::LevelReport;
use serde_json::{Value, json};

/// What `meta.json` records besides the process args and the reports.
pub struct Meta<'a> {
    pub date: String,
    pub argv: Vec<String>,
    pub url: &'a str,
    pub model: &'a str,
    pub api: &'a str,
    /// `bench`, `chats` or `prefill`.
    pub workload: &'a str,
    /// One length for `bench`, the list of N for `prefill`, `null` for `chats`.
    pub prompt_tokens: Value,
    pub gen_tokens: usize,
    /// Whether requests also sent `min_tokens` (`bench`); `None` (`null`) for the others.
    pub min_tokens: Option<bool>,
    /// `None` (`null`) for the workloads that do not use it.
    pub shared_prefix_tokens: Option<usize>,
    /// `None` (`null`) for chats / prefill: they send one request at a time.
    pub concurrency: Option<&'a [usize]>,
    pub server_health: Value,
    pub legion_version: &'a str,
}

/// The files that mark a directory as holding a saved run, of any workload.
pub const RUN_FILES: &[&str] = &["bench.json", "chats.json", "prefill.json", "meta.json"];

/// Fails before the run, not after it, when `dir` already holds a saved run of any workload:
/// every workload rewrites `meta.json` and `README.md`, so mixing runs in one directory would
/// leave a meta.json that describes only the last one.
pub fn check_free(dir: &Path, force: bool) -> anyhow::Result<()> {
    if force {
        return Ok(());
    }
    if let Some(name) = RUN_FILES.iter().find(|name| dir.join(name).exists()) {
        bail!(
            "{} already contains {name} (a saved run); pass --force to overwrite it",
            dir.display()
        );
    }
    Ok(())
}

/// Writes the three files, creating `dir` and its parents.
pub fn write(
    dir: &Path,
    bench_json: &str,
    meta: &Meta,
    reports: &[LevelReport],
) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let put = |name: &str, body: String| {
        let path = dir.join(name);
        std::fs::write(&path, body).with_context(|| format!("writing {}", path.display()))
    };
    put("bench.json", format!("{bench_json}\n"))?;
    put(
        "meta.json",
        format!("{}\n", serde_json::to_string_pretty(&meta_json(meta))?),
    )?;
    put("README.md", readme(meta, reports))?;
    Ok(())
}

pub fn meta_json(m: &Meta) -> Value {
    json!({
        "date": m.date,
        "argv": m.argv,
        "url": m.url,
        "model": m.model,
        "api": m.api,
        "workload": m.workload,
        "prompt_tokens": m.prompt_tokens,
        "gen_tokens": m.gen_tokens,
        "min_tokens": m.min_tokens,
        "shared_prefix_tokens": m.shared_prefix_tokens,
        "concurrency": m.concurrency,
        "server_health": m.server_health,
        "legion_version": m.legion_version,
    })
}

pub fn readme(m: &Meta, reports: &[LevelReport]) -> String {
    let mut out = format!(
        "# legion bench {} @ {}\n\n{}\n\n```sh\n{}\n```\n\n",
        m.model,
        m.url,
        m.date,
        m.argv
            .iter()
            .map(|a| shell_quote(a))
            .collect::<Vec<_>>()
            .join(" ")
    );
    if let Some(min) = m.min_tokens {
        out.push_str(&format!(
            "Requests: greedy (`temperature: 0`), `ignore_eos`, {} tokens each{}.\n\n",
            m.gen_tokens,
            if min {
                ", `min_tokens` set (the server masks stop tokens per row: on Legion every row's \
                 full logits come back to the host)"
            } else {
                ", no `min_tokens`"
            }
        ));
    }
    out.push_str(
        "| C | requests | ok | failed | out tok/s | per-user tok/s | TTFT p50 / p99 (ms) | ITL p50 / p99 (ms) |\n",
    );
    out.push_str("|---|---|---|---|---|---|---|---|\n");
    for r in reports {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.1} | {:.1} | {:.1} / {:.1} | {:.1} / {:.1} |\n",
            r.concurrency,
            r.requests,
            r.ok,
            r.failed,
            r.output_tok_s,
            r.per_user_tok_s,
            r.ttft_ms.p50,
            r.ttft_ms.p99,
            r.itl_ms.p50,
            r.itl_ms.p99,
        ));
    }
    out
}

/// The process args, with the value of `--api-key` replaced: a saved run is meant to be
/// shared and committed, a key is not.
pub fn redacted_argv(args: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut hide = false;
    args.into_iter()
        .map(|a| {
            if std::mem::take(&mut hide) {
                "***".into()
            } else if a == "--api-key" {
                hide = true;
                a
            } else if a.starts_with("--api-key=") {
                "--api-key=***".into()
            } else {
                a
            }
        })
        .collect()
}

fn shell_quote(arg: &str) -> String {
    let plain = !arg.is_empty()
        && arg
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "_@%+=:,./-".contains(c));
    if plain {
        arg.to_string()
    } else {
        format!("'{}'", arg.replace('\'', r"'\''"))
    }
}

/// `GET {url}/health` as JSON; `null` when the request fails, the status is not a success or
/// the body is not JSON.
pub async fn server_health(client: &reqwest::Client, url: &str) -> Value {
    let get = async {
        let body = client
            .get(format!("{}/health", url.trim_end_matches('/')))
            .timeout(Duration::from_secs(5))
            .send()
            .await?
            .error_for_status()?
            .text()
            .await?;
        anyhow::Ok(serde_json::from_str::<Value>(&body)?)
    };
    get.await.unwrap_or(Value::Null)
}

/// RFC 3339 in UTC at seconds precision, e.g. `2026-10-16T20:28:49Z`.
pub fn rfc3339_utc(t: SystemTime) -> String {
    // Times before the epoch clamp to it: no benchmark ran then.
    let secs = t.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (y, m, d) = civil_from_days(days as i64);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// Proleptic Gregorian date of a day count since 1970-01-01 (Howard Hinnant's public-domain
/// `civil_from_days`).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use legion::bench::Percentiles;

    fn at(secs: u64) -> String {
        rfc3339_utc(UNIX_EPOCH + Duration::from_secs(secs))
    }

    #[test]
    fn rfc3339_known_instants() {
        assert_eq!(at(0), "1970-01-01T00:00:00Z");
        assert_eq!(at(86_399), "1970-01-01T23:59:59Z");
        assert_eq!(at(951_782_400), "2000-02-29T00:00:00Z"); // a leap day in a 400-year leap
        assert_eq!(at(1_000_000_000), "2001-09-09T01:46:40Z");
        assert_eq!(at(1_709_164_800), "2024-02-29T00:00:00Z");
        assert_eq!(at(1_709_251_199), "2024-02-29T23:59:59Z");
        assert_eq!(at(1_709_251_200), "2024-03-01T00:00:00Z");
        assert_eq!(at(4_102_444_799), "2099-12-31T23:59:59Z");
        assert_eq!(at(4_107_542_400), "2100-03-01T00:00:00Z"); // 2100 is not a leap year
        assert_eq!(at(1_792_182_529), "2026-10-16T20:28:49Z");
    }

    #[test]
    fn rfc3339_before_the_epoch_clamps() {
        assert_eq!(
            rfc3339_utc(UNIX_EPOCH - Duration::from_secs(5)),
            "1970-01-01T00:00:00Z"
        );
    }

    #[test]
    fn rfc3339_now_has_the_shape() {
        let s = rfc3339_utc(SystemTime::now());
        assert_eq!(s.len(), 20, "{s}");
        assert!(s.ends_with('Z') && &s[10..11] == "T", "{s}");
    }

    fn pct(p50: f64, p99: f64) -> Percentiles {
        Percentiles { p50, p90: p50, p99 }
    }

    fn report(c: usize) -> LevelReport {
        LevelReport {
            concurrency: c,
            requests: 4 * c,
            ok: 4 * c - 1,
            failed: 1,
            wall_secs: 2.0,
            output_tokens: 100,
            output_tok_s: 123.456,
            per_user_tok_s: 7.25,
            ttft_ms: pct(10.04, 20.06),
            itl_ms: pct(1.0, 2.25),
            e2e_ms: pct(5.0, 6.0),
            first_error: None,
            model_bytes: None,
        }
    }

    fn meta<'a>(argv: Vec<String>, conc: &'a [usize]) -> Meta<'a> {
        Meta {
            date: "2026-10-16T20:28:49Z".into(),
            argv,
            url: "http://h:1",
            model: "m",
            api: "openai",
            workload: "bench",
            prompt_tokens: 512.into(),
            gen_tokens: 256,
            min_tokens: Some(false),
            shared_prefix_tokens: Some(0),
            concurrency: Some(conc),
            server_health: Value::Null,
            legion_version: "0.1.0 (x)",
        }
    }

    #[test]
    fn readme_has_title_date_argv_and_a_row_per_level() {
        let m = meta(
            ["legion", "bench", "--model", "a b"]
                .map(String::from)
                .to_vec(),
            &[1, 8],
        );
        let r = readme(&m, &[report(1), report(8)]);
        let mut lines = r.lines();
        assert_eq!(lines.next(), Some("# legion bench m @ http://h:1"));
        assert!(r.contains("\n2026-10-16T20:28:49Z\n"));
        assert!(r.contains("```sh\nlegion bench --model 'a b'\n```"));
        assert!(r.contains(
            "Requests: greedy (`temperature: 0`), `ignore_eos`, 256 tokens each, no `min_tokens`.\n"
        ));
        assert!(r.contains(
            "| C | requests | ok | failed | out tok/s | per-user tok/s | TTFT p50 / p99 (ms) | ITL p50 / p99 (ms) |\n|---|"
        ));
        assert!(r.contains("| 1 | 4 | 3 | 1 | 123.5 | 7.2 | 10.0 / 20.1 | 1.0 / 2.2 |\n"));
        assert!(r.contains("| 8 | 32 | 31 | 1 | 123.5 | 7.2 | 10.0 / 20.1 | 1.0 / 2.2 |\n"));
        assert_eq!(r.lines().filter(|l| l.starts_with("| ")).count(), 3);
    }

    #[test]
    fn meta_keys_in_order() {
        let m = meta(vec!["legion".into()], &[1, 8]);
        let v = meta_json(&m);
        let keys: Vec<_> = v.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "date",
                "argv",
                "url",
                "model",
                "api",
                "workload",
                "prompt_tokens",
                "gen_tokens",
                "min_tokens",
                "shared_prefix_tokens",
                "concurrency",
                "server_health",
                "legion_version"
            ]
        );
        assert_eq!(v["concurrency"], json!([1, 8]));
        assert_eq!(v["server_health"], Value::Null);
    }

    #[test]
    fn api_keys_are_redacted() {
        let a = |v: &[&str]| redacted_argv(v.iter().map(|s| s.to_string()));
        assert_eq!(
            a(&["legion", "bench", "--api-key", "secret", "--json"]),
            ["legion", "bench", "--api-key", "***", "--json"]
        );
        assert_eq!(a(&["--api-key=secret"]), ["--api-key=***"]);
        assert_eq!(a(&["--model", "x"]), ["--model", "x"]);
    }

    #[test]
    fn shell_quoting() {
        assert_eq!(shell_quote("--url=http://h:1/x,y"), "--url=http://h:1/x,y");
        assert_eq!(shell_quote(""), "''");
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
    }

    #[test]
    fn existing_bench_json_blocks_unless_forced() {
        let dir = tempfile::tempdir().unwrap();
        check_free(dir.path(), false).unwrap();
        std::fs::write(dir.path().join("bench.json"), "[]").unwrap();
        assert!(check_free(dir.path(), false).is_err());
        check_free(dir.path(), true).unwrap();
    }
}
