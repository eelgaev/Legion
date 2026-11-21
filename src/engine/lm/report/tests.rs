use super::*;

const GIB: u64 = 1 << 30;

fn stages(n: usize) -> Vec<StageMemory> {
    (0..n)
        .map(|i| StageMemory {
            device: format!("CUDA{i}"),
            ..Default::default()
        })
        .collect()
}

/// `--pair-tp` over 4 GPUs (#125): 40 layers, 10 per stage, and each layer's copy on the
/// pair's other GPU as entry `40 + layer`. Each copy's slots count under the GPU holding it,
/// not all under the last stage.
#[test]
fn pair_copies_count_on_their_gpu() {
    let n_layer = 40;
    let stage_of = |il: usize| (il / 10).min(3);
    let mut entries = Vec::new();
    for il in 0..n_layer {
        let own = stage_of(il);
        entries.push((il, Some(own), 100, GIB));
        // The partner: CUDA0 <-> CUDA1, CUDA2 <-> CUDA3.
        entries.push((n_layer + il, Some(own ^ 1), 50, GIB / 2));
    }
    let mut s = stages(4);
    let cuda: Vec<Option<usize>> = (0..4).map(Some).collect();
    add_expert_slots(&mut s, &cuda, &entries, stage_of);
    for st in &s {
        assert_eq!(st.expert_layers, 20, "{}", st.device);
        assert_eq!(st.expert_slots, 10 * 100 + 10 * 50, "{}", st.device);
        assert_eq!(st.expert_slot_bytes, 10 * GIB + 5 * GIB, "{}", st.device);
    }
}

/// Without a matching stage (the CPU backend, a meta device over tensor-parallel GPUs), an
/// entry counts under its layer's stage, as before.
#[test]
fn unmatched_entries_follow_their_layer() {
    let mut s = stages(2);
    let entries = [(0, None, 10, 1), (1, Some(1), 20, 2), (5, Some(7), 30, 3)];
    add_expert_slots(&mut s, &[Some(0), None], &entries, |il| {
        usize::from(il >= 1)
    });
    assert_eq!((s[0].expert_layers, s[0].expert_slots), (1, 10));
    assert_eq!(
        (
            s[1].expert_layers,
            s[1].expert_slots,
            s[1].expert_slot_bytes
        ),
        (2, 50, 5)
    );
}

/// A plain pipeline (every entry on its own stage's GPU) is unchanged.
#[test]
fn pipeline_is_by_layer() {
    let stage_of = |il: usize| usize::from(il >= 4);
    let entries: Vec<_> = (0..8).map(|il| (il, Some(stage_of(il)), 8, 4)).collect();
    let mut s = stages(2);
    add_expert_slots(&mut s, &[Some(0), Some(1)], &entries, stage_of);
    for st in &s {
        assert_eq!(
            (st.expert_layers, st.expert_slots, st.expert_slot_bytes),
            (4, 32, 16)
        );
    }
}
