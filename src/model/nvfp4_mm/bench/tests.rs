use super::*;

#[test]
fn variant_names_are_unique_and_the_reference_comes_first() {
    assert_eq!(VARIANTS[0].ffn, Ffn::Chain);
    assert!(VARIANTS[0].glu.is_none() && VARIANTS[0].mv.is_none());
    for (i, a) in VARIANTS.iter().enumerate() {
        for b in &VARIANTS[i + 1..] {
            assert_ne!(a.name, b.name);
        }
        for (name, _, _) in PROBE_CHUNKS {
            assert_ne!(a.name, name);
        }
    }
}

#[test]
fn picked_variants_keep_the_reference() {
    assert_eq!(
        picked_variants(&[]),
        (0..VARIANTS.len()).collect::<Vec<_>>()
    );
    let only = ["glu nopdl".to_string(), "glu tt2".to_string()];
    let picked = picked_variants(&only);
    assert_eq!(picked[0], 0);
    let names: Vec<_> = picked.iter().map(|&i| VARIANTS[i].name).collect();
    assert_eq!(names, ["chain", "glu nopdl", "glu tt2"]);
    assert_eq!(picked_variants(&["nothing".to_string()]), [0]);
}

#[test]
fn only_nopdl_variants_turn_the_programmatic_launch_off() {
    for v in VARIANTS {
        assert_eq!(v.pdl, !v.name.ends_with("nopdl"), "{}", v.name);
    }
}

#[test]
fn only_notile_variants_turn_the_tile_prefetch_off() {
    for v in VARIANTS {
        assert_eq!(v.tile, !v.name.contains("notile"), "{}", v.name);
    }
}

#[test]
fn the_tile_defaults_are_compiled_tile_prefetch_variants() {
    use crate::model::nvfp4_mm::{DOWN_MV_TILE, GLU_MV, GLU_MV_TILE};
    // Whole-tile prefetch is compiled for 16-block stages of plain-width tiles only.
    for g in [GLU_MV_TILE, DOWN_MV_TILE] {
        assert!(g.pft > 0 && g.rtw == 1 && g.sb == 16, "{g:?}");
    }
    assert_eq!(GLU_MV.pft, 0);
}

#[test]
fn tile_and_exact_variants_are_named_for_them() {
    for v in VARIANTS {
        assert_eq!(v.exact, v.name.contains("exact"), "{}", v.name);
        let nt = if v.name.contains("nt2") {
            2
        } else if v.name.contains("nt4") {
            4
        } else {
            0
        };
        assert_eq!(v.nt, nt, "{}", v.name);
    }
}

#[test]
fn recorded_routings_keep_the_model_layers_in_file_order() {
    let mut file = Vec::new();
    let w = crate::tp_net::ep_place::write_record;
    w(&mut file, 0, 512, 2, &[1, 2, 3, 4]).unwrap();
    w(&mut file, 48, 512, 2, &[5, 6]).unwrap(); // the MTP head
    w(&mut file, 1, 512, 3, &[7, 8, 9]).unwrap(); // other ids per token
    w(&mut file, 47, 512, 2, &[9, 9]).unwrap();
    file.extend_from_slice(&[1, 0, 0]); // a truncated record
    let got = recorded_routings(&file, 48, 2);
    assert_eq!(got, vec![vec![1, 2, 3, 4], vec![9, 9]]);
}

#[test]
fn recorded_sets_take_whole_sets_of_one_step_size() {
    let rec: Vec<Vec<i32>> = vec![
        vec![0, 1],
        vec![2, 3, 4, 5],
        vec![6, 7],
        vec![8, 9],
        vec![10, 11],
    ];
    let sets = recorded_sets(&rec, 2, 1, 2, 4);
    assert_eq!(
        sets,
        vec![vec![vec![0, 1], vec![6, 7]], vec![vec![8, 9], vec![10, 11]]]
    );
    assert_eq!(recorded_sets(&rec, 2, 1, 2, 1).len(), 1);
    assert!(recorded_sets(&rec, 2, 2, 2, 4).is_empty());
}

#[test]
fn routing_tiles_count_each_experts_own_tiles() {
    // Expert 3: 9 pairs, expert 5: 1, expert 7: 8.
    let mut ids = vec![3; 9];
    ids.push(5);
    ids.extend([7; 8]);
    assert_eq!(routing_tiles(&ids, 8), 2 + 1 + 1);
    assert_eq!(routing_tiles(&ids, 16), 3);
    assert_eq!(routing_tiles(&[], 8), 0);
}
