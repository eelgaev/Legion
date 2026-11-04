mod common;

use axum::body::Body;
use axum::http::Request;
use common::{app, chat, post, send_to};
use serde_json::json;

/// The value of the sample line starting with `series` (name plus labels).
fn sample(body: &str, series: &str) -> Option<f64> {
    body.lines()
        .find_map(|l| l.strip_prefix(series)?.strip_prefix(' ')?.parse().ok())
}

async fn scrape(app: &axum::Router) -> String {
    let (status, body) = send_to(
        app.clone(),
        Request::get("/metrics").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, 200);
    body
}

#[tokio::test]
async fn counts_requests_and_tokens_after_a_chat() {
    let app = app();
    let before = scrape(&app).await;
    assert_eq!(sample(&before, "legion_completion_tokens_total"), Some(0.0));

    let (status, _) = send_to(app.clone(), post("/v1/chat/completions", chat(json!({})))).await;
    assert_eq!(status, 200);

    let body = scrape(&app).await;
    assert_eq!(
        sample(
            &body,
            r#"legion_http_requests_total{route="/v1/chat/completions",status="2xx"}"#
        ),
        Some(1.0),
        "{body}"
    );
    assert!(sample(&body, "legion_prompt_tokens_total").unwrap() > 0.0);
    assert!(sample(&body, "legion_completion_tokens_total").unwrap() > 0.0);
    // The first scrape is itself a request.
    assert_eq!(
        sample(
            &body,
            r#"legion_http_requests_total{route="/metrics",status="2xx"}"#
        ),
        Some(1.0)
    );
}

#[tokio::test]
async fn latency_histograms_have_buckets() {
    let app = app();
    send_to(
        app.clone(),
        post("/v1/chat/completions", chat(json!({"stream": true}))),
    )
    .await;
    let body = scrape(&app).await;
    for name in [
        "legion_time_to_first_token_seconds",
        "legion_inter_token_latency_seconds",
        "legion_request_latency_seconds",
    ] {
        assert!(
            body.contains(&format!("{name}_bucket{{le=\"0.001\"}}")),
            "{name}\n{body}"
        );
        assert!(
            body.contains(&format!("{name}_bucket{{le=\"+Inf\"}}")),
            "{name}"
        );
        assert!(
            sample(&body, &format!("{name}_count")).unwrap() >= 1.0,
            "{name}"
        );
    }
    assert!(sample(&body, "legion_time_to_first_token_seconds_count").unwrap() == 1.0);
}

#[tokio::test]
async fn error_statuses_and_unknown_paths_use_bounded_labels() {
    let app = app();
    send_to(app.clone(), post("/v1/chat/completions", "{")).await;
    send_to(
        app.clone(),
        Request::get("/nope/123").body(Body::empty()).unwrap(),
    )
    .await;
    let body = scrape(&app).await;
    assert_eq!(
        sample(
            &body,
            r#"legion_http_requests_total{route="/v1/chat/completions",status="4xx"}"#
        ),
        Some(1.0),
        "{body}"
    );
    assert_eq!(
        sample(
            &body,
            r#"legion_http_requests_total{route="unmatched",status="4xx"}"#
        ),
        Some(1.0),
        "{body}"
    );
}

#[tokio::test]
async fn engine_gauges_and_in_flight_are_exported() {
    let app = app();
    let body = scrape(&app).await;
    for gauge in [
        "legion_requests_in_flight",
        "legion_engine_running",
        "legion_engine_waiting",
        "legion_engine_kv_blocks_used",
        "legion_engine_kv_blocks_total",
        "legion_engine_prefix_cache_hit_tokens",
        "legion_engine_kv_preemptions",
        "legion_engine_spec_draft_tokens",
        "legion_engine_spec_accepted_tokens",
        "legion_engine_spec_verify_steps",
        "legion_engine_spec_acceptance_rate",
        "legion_engine_spec_mean_accepted_length",
        "legion_engine_spec_mean_k",
    ] {
        assert_eq!(sample(&body, gauge), Some(0.0), "{gauge}\n{body}");
    }
    for phase in ["prefill", "decode", "draft"] {
        for gauge in [
            "legion_engine_model_bytes",
            "legion_engine_model_seconds",
            "legion_engine_model_gb_per_second",
        ] {
            let key = format!("{gauge}{{phase=\"{phase}\"}}");
            assert_eq!(sample(&body, &key), Some(0.0), "{key}\n{body}");
        }
        for gauge in [
            "legion_engine_model_memory_gb_per_second",
            "legion_engine_model_link_gb_per_second",
            "legion_engine_model_roofline_ratio",
        ] {
            let key = format!("{gauge}{{phase=\"{phase}\"}}");
            assert_eq!(sample(&body, &key), Some(0.0), "{key}\n{body}");
        }
        for kind in [
            "dense",
            "experts_device",
            "experts_host",
            "staged",
            "kv",
            "kv_host",
        ] {
            let key =
                format!("legion_engine_model_kind_bytes{{phase=\"{phase}\",kind=\"{kind}\"}}");
            assert_eq!(sample(&body, &key), Some(0.0), "{key}\n{body}");
        }
    }
    // A stage's roofline (none on the mock engine), and no hardware counters without
    // --hw-counters.
    assert_eq!(
        sample(
            &body,
            "legion_engine_roofline_memory_gb_per_second{stage=\"0\"}"
        ),
        Some(0.0),
        "{body}"
    );
    assert!(!body.contains("legion_gpu_"), "{body}");
}
