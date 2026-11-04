//! The placement table's checks and slot tables, the route record format, and the greedy
//! splitter on synthetic routing.

use super::*;

fn placement(owner: Vec<u8>) -> Placement {
    Placement {
        n_expert: owner.len(),
        ranks: 2,
        layers: BTreeMap::from([(1, owner)]),
    }
}

#[test]
fn slot_tables_are_permutations_of_the_share() {
    let p = placement(vec![1, 0, 0, 1, 1, 0, 1, 0]);
    p.check().unwrap();
    assert_eq!(p.experts_of(1, 0), [1, 2, 5, 7]);
    assert_eq!(p.experts_of(1, 1), [0, 3, 4, 6]);
    assert_eq!(p.slot_table(1, 0), [-1, 0, 1, -1, -1, 2, -1, 3]);
    assert_eq!(p.slot_table(1, 1), [0, -1, -1, 1, 2, -1, 3, -1]);
    // Every expert has exactly one rank's slot, each rank's slots are 0..E/2 once.
    for rank in 0..2 {
        let mut slots: Vec<i32> = p
            .slot_table(1, rank)
            .into_iter()
            .filter(|&s| s >= 0)
            .collect();
        slots.sort();
        assert_eq!(slots, [0, 1, 2, 3]);
    }
    // A layer the table does not name: the contiguous split.
    assert!(!p.places(0));
    assert_eq!(p.experts_of(0, 0), [0, 1, 2, 3]);
    assert_eq!(p.slot_table(0, 1), [-1, -1, -1, -1, 0, 1, 2, 3]);
    assert_eq!(runs(&[0, 1, 2, 5, 7, 8]), [(0, 3), (5, 1), (7, 2)]);
    assert_eq!(runs(&[]), []);
}

#[test]
fn check_refuses_uneven_or_broken_tables() {
    assert!(placement(vec![0, 0, 0, 1, 1, 1, 1, 0]).check().is_ok());
    // Five experts on rank 0.
    assert!(placement(vec![0, 0, 0, 0, 0, 1, 1, 1]).check().is_err());
    // A rank that does not exist.
    assert!(placement(vec![0, 0, 0, 2, 1, 1, 1, 1]).check().is_err());
    // Too few entries.
    let mut p = placement(vec![0, 1, 0, 1, 0, 1, 0, 1]);
    p.layers.insert(2, vec![0, 1]);
    assert!(p.check().is_err());
    // Experts that do not divide.
    let p = Placement {
        n_expert: 7,
        ranks: 2,
        layers: BTreeMap::new(),
    };
    assert!(p.check().is_err());
    assert!(placement(vec![0, 1, 0, 1, 0, 1, 0, 1]).fits(8, 2).is_ok());
    assert!(placement(vec![0, 1, 0, 1, 0, 1, 0, 1]).fits(16, 2).is_err());
}

#[test]
fn json_round_trips() {
    let mut p = placement(vec![1, 0, 0, 1, 1, 0, 1, 0]);
    p.layers.insert(40, vec![0, 0, 1, 1, 0, 0, 1, 1]);
    let back: Placement = serde_json::from_str(&p.to_json()).unwrap();
    assert_eq!(back, p);
    let dir = std::env::temp_dir().join(format!("legion-ep-place-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("place.json");
    std::fs::write(&path, p.to_json()).unwrap();
    assert_eq!(Placement::load(&path).unwrap(), p);
    std::fs::write(
        &path,
        r#"{"n_expert": 4, "ranks": 2, "layers": {"0": [0, 0, 0, 1]}}"#,
    )
    .unwrap();
    assert!(Placement::load(&path).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn records_round_trip_and_skip_padding() {
    let mut b = Vec::new();
    write_record(&mut b, 3, 8, 2, &[0, 1, 2, 3]).unwrap();
    write_record(&mut b, 5, 8, 2, &[4, 5]).unwrap();
    // A padding row (-1 ids) is not a token.
    write_record(&mut b, 3, 8, 2, &[6, 7, -1, -1]).unwrap();
    // A record cut short by a stop is dropped.
    let mut cut = Vec::new();
    write_record(&mut cut, 5, 8, 2, &[1, 2]).unwrap();
    b.extend_from_slice(&cut[..cut.len() - 3]);
    let r = Recorded::parse(&b).unwrap();
    assert_eq!(r.n_expert, 8);
    assert_eq!(r.layers[&3].ids, [0, 1, 2, 3, 6, 7]);
    assert_eq!(r.layers[&3].tokens(), 3);
    assert_eq!(r.layers[&5].ids, [4, 5]);
    let mut more = r.clone();
    more.extend(r.clone()).unwrap();
    assert_eq!(more.layers[&3].tokens(), 6);
    let mut other = Vec::new();
    write_record(&mut other, 3, 16, 2, &[0, 1]).unwrap();
    assert!(more.extend(Recorded::parse(&other).unwrap()).is_err());
}

/// Tokens that pick `group` experts of a block of `group` consecutive ones together: the
/// contiguous split puts a whole block on one rank.
fn blocks(n_expert: usize, group: usize, tokens: usize) -> Rows {
    let mut ids = Vec::new();
    for t in 0..tokens {
        let b = (t * 7) % (n_expert / group);
        ids.extend((0..group).map(|i| (b * group + i) as u16));
    }
    Rows { k: group, ids }
}

#[test]
fn splitter_separates_experts_picked_together() {
    // Pairs {0,1}, {2,3}, ...: contiguous puts both of a pair on one rank (max 2); the split
    // gives each rank one of every pair (max 1, the floor).
    let rows = blocks(16, 2, 200);
    let cont = contiguous(16, 2);
    assert_eq!(mean_max(&rows, &cont, 2), 2.0);
    let owner = split(&rows, 16, 2, 8);
    let p = placement(owner.clone());
    p.check().unwrap();
    assert_eq!(mean_max(&rows, &owner, 2), 1.0);
    // Blocks of four: the floor is two each.
    let rows = blocks(32, 4, 300);
    assert_eq!(mean_max(&rows, &contiguous(32, 2), 2), 4.0);
    let owner = split(&rows, 32, 2, 8);
    assert_eq!(mean_max(&rows, &owner, 2), 2.0);
    // Deterministic.
    assert_eq!(split(&rows, 32, 2, 8), owner);
}

#[test]
fn splitter_never_does_worse_than_its_greedy_start_and_keeps_shares_even() {
    // Pseudo-random routing with some structure: k = 6 of 64.
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    let mut ids = Vec::new();
    for _ in 0..500 {
        let mut tok: Vec<u16> = Vec::new();
        while tok.len() < 6 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            // Skewed: low ids picked more often, in runs.
            let e = ((x % 64) * (x % 64) / 64) as u16;
            if !tok.contains(&e) {
                tok.push(e);
            }
        }
        ids.extend(tok);
    }
    let rows = Rows { k: 6, ids };
    let greedy = split(&rows, 64, 2, 0);
    let owner = split(&rows, 64, 2, 8);
    placement(owner.clone()).check().unwrap();
    let (g, s, c) = (
        mean_max(&rows, &greedy, 2),
        mean_max(&rows, &owner, 2),
        mean_max(&rows, &contiguous(64, 2), 2),
    );
    assert!(s <= g, "{s} > {g}");
    assert!(s < c, "{s} >= contiguous {c}");
    assert!(s >= 3.0);
    // No tokens: contiguous.
    assert_eq!(split(&Rows::default(), 8, 2, 4), contiguous(8, 2));
}
