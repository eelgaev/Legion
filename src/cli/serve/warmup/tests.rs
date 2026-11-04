use std::time::Duration;

use legion::engine::mock::{MockConfig, MockEngine};

use super::*;

#[tokio::test]
async fn runs_a_prompt_of_the_asked_size() {
    let engine = MockEngine::new(MockConfig {
        step: Duration::from_millis(1),
        ..Default::default()
    });
    assert_eq!(run(&engine, 300).await.unwrap(), 300);
    // It finished: nothing is left running.
    assert_eq!(engine.stats().running, 0);
}

#[test]
fn the_request_is_greedy_and_short() {
    let req = request("m", 5);
    assert_eq!(req.prompt, Prompt::Text("the the the the the ".into()));
    assert_eq!(req.sampling.temperature, 0.0);
    assert_eq!(req.max_tokens, 1);
}

#[test]
fn shape_plan_keeps_order_and_drops_zeros_repeats_and_too_long() {
    assert_eq!(
        shape_plan(&[1024, 256, 128], Some(81920)),
        vec![1024, 256, 128]
    );
    assert_eq!(shape_plan(&[0], Some(81920)), Vec::<usize>::new());
    assert_eq!(shape_plan(&[256, 0, 256, 64], None), vec![256, 64]);
    // Half the context at most.
    assert_eq!(shape_plan(&[1024, 256, 128], Some(1024)), vec![256, 128]);
}

#[tokio::test]
async fn shapes_run_one_prompt_per_size() {
    let engine = MockEngine::new(MockConfig {
        step: Duration::from_millis(1),
        ..Default::default()
    });
    shapes(&engine, &[64, 32]).await;
    assert_eq!(engine.stats().running, 0);
    assert_eq!(engine.stats().prompt_tokens, 96);
}
