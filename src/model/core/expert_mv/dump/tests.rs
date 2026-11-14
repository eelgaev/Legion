use super::*;

#[test]
fn config_parses_the_variable() {
    assert_eq!(
        Config::parse("/tmp/d.bin"),
        Some(Config {
            path: "/tmp/d.bin".into(),
            every: 1,
            full: 0,
            max: u64::MAX
        })
    );
    assert_eq!(
        Config::parse("d,every=7,full=96,max=5000"),
        Some(Config {
            path: "d".into(),
            every: 7,
            full: 96,
            max: 5000
        })
    );
    assert_eq!(Config::parse(""), None);
    assert_eq!(Config::parse("d,every=0"), None);
    assert_eq!(Config::parse("d,each=2"), None);
    assert_eq!(Config::parse("d,full"), None);
}

#[test]
fn digest_tells_rows_apart() {
    let a: Vec<u8> = (0..2560u32)
        .flat_map(|i| (i as f32).to_bits().to_le_bytes())
        .collect();
    let mut b = a.clone();
    b[1234] ^= 1;
    assert_eq!(digest(&a), digest(&a));
    assert_ne!(digest(&a), digest(&b));
    assert_ne!(digest(&a[..8]), digest(&a[..9]));
    assert_ne!(digest(&[]), digest(&[0]));
}

fn sample(glu: bool, full: bool, pad: bool) -> Record {
    let (n_used, n_tok, rows, k) = (3, 2, 4, 64);
    let pairs = n_used * n_tok;
    let mats = if glu { 2 } else { 1 };
    let n_x = if glu { 1 } else { n_used };
    Record {
        seq: 0x1_0000_0002,
        flags: if glu { F_GLU } else { 0 }
            | if full { F_FULL } else { 0 }
            | if pad { F_PAD } else { 0 },
        ty: 12,
        k,
        rows,
        n_used,
        n_tok,
        n_x,
        limit: 1.5f32.to_bits(),
        n_w: 600,
        vram_below: 400,
        ids: vec![5, 9, 400, 5, 77, 599],
        pad: if pad { vec![0, 0] } else { Vec::new() },
        prints: (0..mats)
            .map(|m| vec![m as u8 + 1; pairs * PRINT])
            .collect(),
        xq: (0..n_tok * n_x * (k / 32) * 36).map(|i| i as u8).collect(),
        digests: (0..mats)
            .map(|m| (0..pairs).map(|p| [m as u64, p as u64]).collect())
            .collect(),
        act_equal: if glu { vec![1; pairs] } else { Vec::new() },
        full: if full {
            (0..mats)
                .map(|m| (0..pairs * rows).map(|i| (i + m) as f32 * -0.5).collect())
                .collect()
        } else {
            Vec::new()
        },
    }
}

#[test]
fn records_round_trip() {
    let all = [
        sample(true, true, false),
        sample(false, false, true),
        sample(true, false, true),
        sample(false, true, false),
    ];
    let bytes: Vec<u8> = all.iter().flat_map(Record::to_bytes).collect();
    let mut r = &bytes[..];
    for want in &all {
        assert_eq!(Record::read(&mut r).unwrap().as_ref(), Some(want));
    }
    assert_eq!(Record::read(&mut r).unwrap(), None);
}

#[test]
fn a_cut_or_foreign_file_is_an_error() {
    let bytes = sample(true, true, false).to_bytes();
    let cut = &bytes[..bytes.len() - 3];
    assert_eq!(
        Record::read(&mut &cut[..]).unwrap_err().kind(),
        io::ErrorKind::UnexpectedEof
    );
    let mut foreign = bytes.clone();
    foreign[0] ^= 0xff;
    assert_eq!(
        Record::read(&mut &foreign[..]).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    // A header whose sizes do not add up to its body.
    let mut wrong = bytes;
    wrong[7 * 4] += 1;
    assert_eq!(
        Record::read(&mut &wrong[..]).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
}

#[test]
fn dead_pairs_are_padding_rows_no_real_row_shares() {
    // Row 1 is padding (a copy of row 0): its pairs are dead unless row 0 chose the expert too.
    let rec = sample(false, false, true);
    assert!(!rec.dead(0, 0));
    assert!(!rec.dead(1, 0)); // expert 5, which row 0 chose
    assert!(rec.dead(1, 1)); // 77
    assert!(rec.dead(1, 2)); // 599
    assert!(!sample(false, false, false).dead(1, 1));
}
