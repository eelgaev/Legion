use super::*;

#[test]
fn decode_and_verify_steps_queue_their_fills() {
    assert!(queued(1, false, false, false));
}

#[test]
fn prompt_steps_fill_synchronously() {
    assert!(!queued(1, true, false, false));
}

#[test]
fn pipelined_steps_fill_synchronously() {
    // Micro-chunks over the stages fill through `early_inputs` or the dataflow's `flow_io`.
    assert!(!queued(2, false, false, false));
    assert!(!queued(4, true, false, false));
}

#[test]
fn steps_another_gpu_reads_fill_synchronously() {
    assert!(!queued(1, false, true, false));
    assert!(!queued(1, false, false, true));
}
