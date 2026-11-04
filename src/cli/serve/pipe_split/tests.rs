use std::time::Duration;

use legion::engine::lm::DeviceChoice;
use legion::engine::mock::{MockConfig, MockEngine};
use legion::spec::SpecKind;

use super::*;

fn mtp_groups() -> LmConfig {
    let mut cfg = LmConfig::new("m.gguf", DeviceChoice::Cuda);
    cfg.pipeline_groups = 4;
    cfg.pipeline_mtp_split = true;
    cfg.spec.kind = SpecKind::Mtp;
    cfg.gpus = 4;
    cfg
}

#[test]
fn measures_only_groups_with_the_mtp_head_and_no_given_split() {
    assert_eq!(skip_reason(&mtp_groups()), None);
    let mut cfg = mtp_groups();
    cfg.pipeline_groups = 0;
    assert!(skip_reason(&cfg).unwrap().contains("--pipeline-groups"));
    let mut cfg = mtp_groups();
    cfg.spec.kind = SpecKind::None;
    assert!(skip_reason(&cfg).unwrap().contains("MTP"));
    let mut cfg = mtp_groups();
    cfg.layer_split = vec![14, 28, 42];
    assert!(skip_reason(&cfg).unwrap().contains("--layer-split"));
    let mut cfg = mtp_groups();
    cfg.gpus = 1;
    assert!(skip_reason(&cfg).unwrap().contains("--gpus"));
    let mut cfg = mtp_groups();
    cfg.tp = 2;
    assert!(skip_reason(&cfg).is_some());
}

#[test]
fn the_requests_are_greedy_and_ignore_eos() {
    let req = request("m", 9, 64);
    assert_eq!(req.prompt, Prompt::Text(PROMPTS[1].into()));
    assert_eq!(req.sampling.temperature, 0.0);
    assert_eq!(req.max_tokens, 64);
    assert!(req.ignore_eos);
}

#[tokio::test]
async fn a_batch_runs_every_request_to_its_end() {
    let engine = MockEngine::new(MockConfig {
        step: Duration::from_millis(1),
        ..Default::default()
    });
    run_batch(&engine, 4, 8).await.unwrap();
    assert_eq!(engine.stats().running, 0);
    assert!(engine.stats().generated_tokens >= 4 * 8);
}
