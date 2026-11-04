use super::*;

/// Two stages on two cache devices, the MTP layer (40) on the last one.
fn two_devices(il: usize) -> Option<usize> {
    Some(usize::from(il >= 20))
}

#[test]
fn splits_off_the_last_stages_target_layers() {
    assert_eq!(tail_split(&[0..20, 20..41], 40, two_devices), Some(20..40));
    // Three stages: the stages before the last run first.
    let three = |il: usize| Some(il / 14);
    assert_eq!(
        tail_split(&[0..14, 14..28, 28..41], 40, three),
        Some(28..40)
    );
}

#[test]
fn one_stage_does_not_split() {
    let one = vec![Range { start: 0, end: 41 }];
    assert_eq!(tail_split(&one, 40, two_devices), None);
    assert_eq!(tail_split(&[], 40, two_devices), None);
}

#[test]
fn a_device_on_both_sides_does_not_split() {
    // One cache device for every layer: two calls would plan swaps twice.
    assert_eq!(tail_split(&[0..20, 20..41], 40, |_| Some(0)), None);
    // A device whose layers cross the stage cut.
    let crossing = |il: usize| Some(usize::from(il >= 22));
    assert_eq!(tail_split(&[0..20, 20..41], 40, crossing), None);
}

#[test]
fn a_side_without_tiered_layers_does_not_split() {
    let last_only = |il: usize| (il >= 20).then_some(1);
    assert_eq!(tail_split(&[0..20, 20..41], 40, last_only), None);
    let first_only = |il: usize| (il < 20).then_some(0);
    assert_eq!(tail_split(&[0..20, 20..41], 40, first_only), None);
    // Untiered layers between tiered ones are fine.
    let sparse = |il: usize| il.is_multiple_of(2).then_some(usize::from(il >= 20));
    assert_eq!(tail_split(&[0..20, 20..41], 40, sparse), Some(20..40));
}

#[test]
fn a_last_stage_of_mtp_layers_only_does_not_split() {
    assert_eq!(tail_split(&[0..40, 40..41], 40, two_devices), None);
}

#[test]
fn queued_inputs_on_the_cpu_are_set_at_once() {
    use crate::ggml::{Context, types};
    let mut backend = Backend::cpu().unwrap();
    let ctx = Context::new(2, 0);
    let a = ctx.new_tensor(types::I32, &[3]).named("a");
    let b = ctx.new_tensor(types::F32, &[2]).named("b");
    ctx.alloc_tensors(backend.buffer_type()).unwrap();
    let mut io = PinnedIo::default();
    io.begin(&mut backend, PINNED_MAX);
    let mut q = Queued {
        io: &mut io,
        backend: &mut backend,
    };
    q.i32(a, &[4, 5, 6]);
    q.f32(b, &[0.5, -1.0]);
    io.finish(&mut backend);
    assert_eq!(a.to_i32(), [4, 5, 6]);
    assert_eq!(b.to_f32(), [0.5, -1.0]);
}
