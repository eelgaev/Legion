use std::mem::{offset_of, size_of};

use super::*;

/// `PtxParams`' field offsets by name.
fn offsets() -> Vec<(&'static str, usize)> {
    vec![
        ("w", offset_of!(PtxParams, w)),
        ("w2", offset_of!(PtxParams, w2)),
        ("w_nb1", offset_of!(PtxParams, w_nb1)),
        ("w_nb2", offset_of!(PtxParams, w_nb2)),
        ("xq", offset_of!(PtxParams, xq)),
        ("xq_col", offset_of!(PtxParams, xq_col)),
        ("ids", offset_of!(PtxParams, ids)),
        ("ids_nb1", offset_of!(PtxParams, ids_nb1)),
        ("dst", offset_of!(PtxParams, dst)),
        ("dst_nb1", offset_of!(PtxParams, dst_nb1)),
        ("dst_nb2", offset_of!(PtxParams, dst_nb2)),
        ("rows", offset_of!(PtxParams, rows)),
        ("n_used", offset_of!(PtxParams, n_used)),
        ("n_x", offset_of!(PtxParams, n_x)),
        ("vram_below", offset_of!(PtxParams, vram_below)),
        ("n_w", offset_of!(PtxParams, n_w)),
        ("part", offset_of!(PtxParams, part)),
        ("units", offset_of!(PtxParams, units)),
        ("lshift", offset_of!(PtxParams, lshift)),
        ("limit", offset_of!(PtxParams, limit)),
        ("n_tok", offset_of!(PtxParams, n_tok)),
    ]
}

/// The .ptx header's parameter table: `<offset> <name> <u64|i64|i32|f32>` in its comments.
fn header_table() -> Vec<(String, usize)> {
    let mut out = Vec::new();
    for line in MODULE
        .text
        .lines()
        .filter_map(|l| l.trim().strip_prefix("//"))
    {
        let t: Vec<&str> = line.split_whitespace().collect();
        for w in t.windows(3) {
            if let Ok(off) = w[0].parse::<usize>()
                && ["u64", "i64", "i32", "f32"].contains(&w[2])
            {
                out.push((w[1].to_string(), off));
            }
        }
    }
    out
}

#[test]
fn params_match_the_ptx_header() {
    assert_eq!(size_of::<PtxParams>(), 128);
    assert!(MODULE.text.contains(".param .align 8 .b8 P[128]"));
    let table = header_table();
    let fields = offsets();
    assert_eq!(table.len(), fields.len(), "{table:?}");
    for (name, off) in &fields {
        let got = table.iter().find(|(n, _)| n == name).map(|(_, o)| *o);
        assert_eq!(got, Some(*off), "field {name}");
    }
}

/// Every `[P+n]` the kernels read is a field's offset.
#[test]
fn ptx_reads_only_field_offsets() {
    let fields = offsets();
    let mut seen = 0;
    for part in MODULE.text.split("[P+").skip(1) {
        let off: usize = part.split(']').next().unwrap().parse().unwrap();
        assert!(fields.iter().any(|(_, o)| *o == off), "[P+{off}]");
        seen += 1;
    }
    assert!(seen > 30);
}

#[test]
fn module_targets_volta_and_has_both_kernels() {
    assert_eq!(MODULE.target(), Some(70));
    assert_eq!(
        MODULE.entries(),
        vec![Q4K_GLU, Q4K_GLU_USERS, Q51_MV, Q51_MV_USERS]
    );
    assert_eq!(
        MODULE
            .text
            .matches(&format!(".maxntid {THREADS}, 1, 1"))
            .count(),
        4
    );
}

/// The several-user forms read the step's token count; each code path per user count is there.
#[test]
fn users_forms_have_a_path_per_user_count() {
    let users = |name: &str| {
        let at = MODULE.text.find(&format!(".entry {name}(")).unwrap();
        let body = &MODULE.text[at..];
        body[..body.find("\n}\n").unwrap()].to_string()
    };
    for name in [Q4K_GLU_USERS, Q51_MV_USERS] {
        let b = users(name);
        assert!(b.contains("[P+124]"), "{name} reads n_tok");
        assert!(b.contains("vote.sync.ballot.b32"), "{name} finds the users");
        for n in 1..=4 {
            assert!(b.contains(&format!("\nU{n}:\n")), "{name} path U{n}");
        }
    }
    for name in [Q4K_GLU, Q51_MV] {
        let b = users(name);
        assert!(
            !b.contains("[P+124]") && !b.contains("\nU2:\n"),
            "{name}: one user"
        );
    }
}

#[test]
fn several_tokens_take_the_users_forms() {
    assert_eq!(entry_for(types::Q4_K, true, 1), Some(Q4K_GLU));
    assert_eq!(entry_for(types::Q4_K, true, 3), Some(Q4K_GLU_USERS));
    assert_eq!(entry_for(types::Q5_1, false, 1), Some(Q51_MV));
    assert_eq!(entry_for(types::Q5_1, false, 2), Some(Q51_MV_USERS));
    assert_eq!(entry_for(types::Q5_1, true, 2), None);
}

#[test]
fn kernels_only_for_their_formats() {
    assert_eq!(kernel_for(types::Q4_K, true), Some(Q4K_GLU));
    assert_eq!(kernel_for(types::Q5_1, false), Some(Q51_MV));
    assert_eq!(kernel_for(types::Q4_K, false), None);
    assert_eq!(kernel_for(types::Q5_1, true), None);
    assert_eq!(kernel_for(types::Q5_K, true), None);
    assert_eq!(kernel_for(types::Q8_0, false), None);
}

/// Qwen3.8-Flash-Next's experts: gate/up 2560 -> 640 Q4_K, down 640 -> 2560 Q5_1.
#[test]
fn lanes_and_units_follow_the_tile_kernel() {
    assert_eq!(units_per_row(types::Q4_K, 2560), 160);
    assert_eq!(lanes_per_row(160), 32);
    assert_eq!(units_per_row(types::Q5_1, 640), 40);
    assert_eq!(lanes_per_row(40), 8);
    assert_eq!(lanes_per_row(4), 1);
    assert_eq!(lanes_per_row(7), 1);
    assert_eq!(lanes_per_row(8), 2);
    assert_eq!(lanes_per_row(1000), 32);
}

fn params(rows: i32, lshift: i32, vram_below: i32, part: i32) -> PtxParams {
    PtxParams {
        w: std::ptr::null(),
        w2: std::ptr::null(),
        w_nb1: 0,
        w_nb2: 0,
        xq: std::ptr::null(),
        xq_col: 0,
        ids: std::ptr::null(),
        ids_nb1: 0,
        dst: std::ptr::null_mut(),
        dst_nb1: 0,
        dst_nb2: 0,
        rows,
        n_used: 10,
        n_x: 1,
        vram_below,
        n_w: 300,
        part,
        units: 0,
        lshift,
        limit: 0.0,
        n_tok: 1,
    }
}

#[test]
fn plan_covers_every_row_and_pair() {
    // Q4_K: 32 lanes a row, a row per warp, 4 rows a CTA.
    let p = plan(types::Q4_K, true, &params(640, 5, 44, 0), 10).unwrap();
    assert_eq!(p.entry, Q4K_GLU);
    assert_eq!(p.grid, [160, 10, 1]);
    // Q5_1: 8 lanes a row, 16 rows a CTA; 2560 rows.
    let p = plan(types::Q5_1, false, &params(2560, 3, 44, 1), 30).unwrap();
    assert_eq!(p.grid, [160, 30, 1]);
    // A partial last block.
    let p = plan(types::Q5_1, false, &params(2561, 3, 44, 3), 1).unwrap();
    assert_eq!(p.grid, [161, 1, 1]);
    // Three tokens: the users form, a CTA row per pair still (duplicates leave at once).
    let three = PtxParams {
        n_tok: 3,
        ..params(640, 5, 44, 0)
    };
    let p = plan(types::Q4_K, true, &three, 30).unwrap();
    assert_eq!(p.entry, Q4K_GLU_USERS);
    assert_eq!(p.grid, [160, 30, 1]);
}

#[test]
fn plan_skips_what_the_kernels_do_not_take() {
    // No cache slots, the host parts of a split, another format, nothing to do.
    assert!(plan(types::Q4_K, true, &params(640, 5, 0, 0), 10).is_none());
    assert!(plan(types::Q4_K, true, &params(640, 5, 44, 2), 10).is_none());
    assert!(plan(types::Q4_K, true, &params(640, 5, 44, 4), 10).is_none());
    assert!(plan(types::Q8_0, false, &params(640, 5, 44, 0), 10).is_none());
    assert!(plan(types::Q4_K, true, &params(640, 5, 44, 0), 0).is_none());
}

#[test]
fn shapes_need_whole_aligned_blocks() {
    // Q4_K 2560: 10 blocks of 144 bytes.
    assert!(shape_ok(types::Q4_K, 2560, 1440, 1440 * 640, 4096));
    assert!(!shape_ok(types::Q4_K, 2560, 1456, 1456 * 640, 4096));
    assert!(!shape_ok(types::Q4_K, 2560, 1440, 1440 * 640, 4104));
    assert!(!shape_ok(types::Q4_K, 2048 + 32, 1440, 1440 * 640, 4096));
    // Q5_1 640: 20 blocks of 24 bytes.
    assert!(shape_ok(types::Q5_1, 640, 480, 480 * 2560, 8));
    assert!(!shape_ok(types::Q5_1, 640, 480, 480 * 2560 + 4, 8));
    assert!(!shape_ok(types::Q6_K, 640, 480, 480 * 2560, 8));
}
