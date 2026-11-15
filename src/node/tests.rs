use super::*;

fn round_trip_request(r: Request) -> Request {
    let f = r.encode();
    let mut buf = Vec::new();
    write_frame(&mut buf, &f).unwrap();
    let (k, h, b) = read_frame(&mut buf.as_slice()).unwrap().unwrap();
    Request::decode(k, &h, b).unwrap()
}

fn round_trip_reply(r: Reply) -> Reply {
    let mut buf = Vec::new();
    write_frame(&mut buf, &r.encode()).unwrap();
    let (k, h, b) = read_frame(&mut buf.as_slice()).unwrap().unwrap();
    Reply::decode(k, &h, b).unwrap()
}

fn step() -> Request {
    Request::Step {
        groups: vec![
            WireGroup {
                slot: 3,
                verify: false,
                seqs: vec![
                    WireSeq {
                        tokens: vec![1, 2, 3],
                        n_past: 17,
                        blocks: vec![9, 4],
                        ring: 0,
                        logits: (1, 0),
                    },
                    WireSeq {
                        tokens: vec![],
                        n_past: 0,
                        blocks: vec![],
                        ring: 2,
                        logits: (4, 7),
                    },
                ],
            },
            WireGroup {
                slot: 0,
                verify: true,
                seqs: Vec::new(),
            },
        ],
        rows: vec![0.5, -1.25, f32::MIN_POSITIVE, 3e38],
    }
}

#[test]
fn requests_round_trip() {
    for r in [
        hello(48, 22, 1234),
        Request::Resize { n_blocks: 77 },
        step(),
        Request::ClearSlot { slot: 5 },
        Request::CopyStates {
            copies: vec![(1, 2, 0, 3), (0, 9, 1, 0)],
        },
        tp_hello(3, 2, 60, 999),
        Request::TpStep {
            groups: match step() {
                Request::Step { groups, .. } => groups,
                _ => unreachable!(),
            },
            micro: 512,
            taper: 3,
            gather_budget: u64::MAX - 1,
            head: 2,
        },
        Request::SaveRing {
            ring: 4,
            slot: 9,
            n: 1000,
            restore: true,
        },
        Request::EnableRingSaves { slots: 48 },
        Request::TpMtp {
            groups: match step() {
                Request::Step { groups, .. } => groups,
                _ => unreachable!(),
            },
            gather_budget: 4096,
            h: vec![0.5, -1.25, f32::MIN_POSITIVE, 3e7],
        },
        Request::TpCommit {
            commits: vec![(3, 1), (0, 4)],
        },
        Request::TpPrefetch {
            before: vec![7, 8, u32::MAX],
            tokens: vec![1, 2, 3, 4],
        },
        Request::TpPrefetch {
            before: vec![],
            tokens: vec![9],
        },
        Request::TpMtpChain {
            groups: match step() {
                Request::Step { groups, .. } => groups,
                _ => unreachable!(),
            },
            gather_budget: 0,
            parts: vec![
                RowsPart::Sent { len: 1 },
                RowsPart::Step {
                    back: 0,
                    layer: 2,
                    start: 30,
                    len: 10,
                    check: 0,
                },
                RowsPart::Sent { len: 1 },
            ],
            h: vec![0.25, -7.5],
            caps: vec![8, 1, 0],
            p_min: 0.35f32.to_bits(),
            tails: vec![vec![5, 6, 7], vec![], vec![u32::MAX]],
        },
        Request::TpMtpRows {
            groups: match step() {
                Request::Step { groups, .. } => groups,
                _ => unreachable!(),
            },
            gather_budget: 8192,
            parts: vec![
                RowsPart::Sent { len: 2 },
                RowsPart::Step {
                    back: 1,
                    layer: 0,
                    start: 1 << 33,
                    len: 40_960,
                    check: u64::MAX,
                },
                RowsPart::Sent { len: 0 },
            ],
            h: vec![1.5, -0.0],
        },
        Request::Draft {
            reqs: vec![
                WireDraft {
                    seq: u64::MAX - 3,
                    base: 9,
                    tokens: vec![11, 12, 13],
                    start: 17,
                    ran: 1,
                    off: 4,
                    max: 3,
                    blocks: vec![2, 7],
                    ring: 1,
                },
                WireDraft {
                    seq: 0,
                    base: 0,
                    tokens: Vec::new(),
                    start: 0,
                    ran: 8192,
                    off: 0,
                    max: 0,
                    blocks: Vec::new(),
                    ring: 0,
                },
            ],
            p_min: 0.125f32.to_bits(),
        },
        Request::Draft {
            reqs: Vec::new(),
            p_min: 0,
        },
        Request::DraftRelease { seq: 1 << 40 },
    ] {
        assert_eq!(round_trip_request(r.clone()), r);
    }
}

#[test]
fn rows_check_sees_every_bit_and_is_never_zero() {
    let a = [0.5f32, -1.25, 3e7];
    assert_eq!(rows_check(&a), rows_check(&a.clone()));
    assert_ne!(rows_check(&a), 0);
    assert_ne!(rows_check(&[]), 0);
    // -0.0 and 0.0 compare equal as floats, not as bits.
    assert_ne!(rows_check(&[0.0]), rows_check(&[-0.0]));
    let mut b = a;
    b[2] = f32::from_bits(b[2].to_bits() ^ 1);
    assert_ne!(rows_check(&a), rows_check(&b));
    // Order matters.
    assert_ne!(rows_check(&[1.0, 2.0]), rows_check(&[2.0, 1.0]));
}

#[test]
fn tp_hello_checks_the_build_ranks_and_model() {
    let mine = (2, 1, 60);
    assert!(check_tp_hello((VERSION, 2, 1, 60), mine).is_ok());
    assert!(check_tp_hello((VERSION + 1, 2, 1, 60), mine).is_err());
    assert!(check_tp_hello((VERSION, 3, 1, 60), mine).is_err());
    assert!(check_tp_hello((VERSION, 2, 0, 60), mine).is_err());
    assert!(check_tp_hello((VERSION, 2, 1, 48), mine).is_err());
}

#[test]
fn replies_round_trip() {
    for r in [
        Reply::Hello { n_blocks: 10 },
        Reply::Done,
        Reply::Step {
            argmax: vec![4, 5],
            rows: vec![vec![1.0, 2.0], vec![], vec![-3.5]],
            ms: 12.5,
        },
        Reply::Drafts {
            drafts: vec![vec![(5, 0.75), (9, 0.5)], Vec::new(), vec![(0, 1.0)]],
            passes: 3,
            ms: 4.25,
        },
        Reply::Drafts {
            drafts: Vec::new(),
            passes: 0,
            ms: 0.0,
        },
        Reply::Error("out of memory".into()),
    ] {
        assert_eq!(round_trip_reply(r.clone()), r);
    }
}

#[test]
fn residual_bits_cross_unchanged() {
    // Every float pattern, NaN payloads included, arrives as sent.
    let rows: Vec<f32> = [0u32, 1, 0x7fc0_0001, 0xffff_ffff, 0x8000_0000, 0x3f80_0000]
        .into_iter()
        .map(f32::from_bits)
        .collect();
    let Request::Step { rows: back, .. } = round_trip_request(Request::Step {
        groups: Vec::new(),
        rows: rows.clone(),
    }) else {
        panic!("not a step");
    };
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&back), bits(&rows));
}

#[test]
fn short_or_long_frames_are_refused() {
    let f = Request::ClearSlot { slot: 1 }.encode();
    assert!(Request::decode(f.kind, &f.head[..2], Vec::new()).is_err());
    let mut long = f.head.clone();
    long.push(0);
    assert!(Request::decode(f.kind, &long, Vec::new()).is_err());
    assert!(Request::decode(999, &f.head, Vec::new()).is_err());
}

#[test]
fn clean_end_of_stream_is_none() {
    assert!(read_frame(&mut [].as_slice()).unwrap().is_none());
}

#[test]
fn hello_checks_version_and_split() {
    assert!(check_hello(VERSION, 48, 22, (48, 22)).is_ok());
    assert!(check_hello(VERSION + 1, 48, 22, (48, 22)).is_err());
    assert!(check_hello(VERSION, 48, 23, (48, 22)).is_err());
}

#[test]
fn shares_cover_the_layers() {
    let c = |rank| NodeConfig {
        rank,
        peer: String::new(),
        split: None,
    };
    assert_eq!(c(0).share(48, 20), 0..20);
    assert_eq!(c(1).share(48, 20), 20..48);
}

#[test]
fn link_answers_in_order_over_tcp() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let server = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut seen = 0u32;
        serve_one(stream, &mut |req| {
            seen += 1;
            match req {
                Request::Step { rows, .. } => Reply::Step {
                    argmax: vec![seen, rows.len() as u32],
                    rows: vec![rows],
                    ms: 1.0,
                },
                Request::ClearSlot { slot: 99 } => Reply::Error("bad slot".into()),
                _ => Reply::Done,
            }
        })
        .unwrap();
        seen
    });
    let mut link = Link::connect(&addr, Duration::from_secs(5)).unwrap();
    // Two steps without waiting, then the last reply is the second's.
    let big: Vec<f32> = (0..300_000).map(|i| i as f32).collect();
    link.send(Request::Step {
        groups: Vec::new(),
        rows: vec![1.0],
    })
    .unwrap();
    link.send(Request::Step {
        groups: Vec::new(),
        rows: big.clone(),
    })
    .unwrap();
    assert_eq!(link.owed(), 2);
    match link.wait().unwrap() {
        Reply::Step { argmax, rows, .. } => {
            assert_eq!(argmax, vec![2, 300_000]);
            assert_eq!(rows, vec![big]);
        }
        r => panic!("{r:?}"),
    }
    assert_eq!(link.owed(), 0);
    assert_eq!(link.times.steps, 1);
    // An error reply among the owed ones is the wait's error.
    link.send(Request::ClearSlot { slot: 99 }).unwrap();
    link.send(Request::ClearSlot { slot: 1 }).unwrap();
    assert!(link.wait().unwrap_err().contains("bad slot"));
    drop(link);
    assert_eq!(server.join().unwrap(), 4);
}
