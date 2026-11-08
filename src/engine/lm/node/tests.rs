use super::*;

#[test]
fn each_tensor_loads_on_one_rank() {
    let n = 8;
    for (name, r0, r1) in [
        ("token_embd.weight", true, false),
        ("rope_freqs.weight", true, true),
        ("output_norm.weight", false, true),
        ("output.weight", false, true),
        ("blk.0.attn_q.weight", true, false),
        ("blk.2.ffn_down_exps.weight", true, false),
        ("blk.3.ffn_down_exps.weight", false, true),
        ("blk.7.attn_norm.weight", false, true),
        // An MTP head's layer, past the last: nowhere while the head does not draft.
        ("blk.8.nextn.eh_proj.weight", false, false),
    ] {
        assert_eq!(owns(0, 3, n, false, name), r0, "{name} on rank 0");
        assert_eq!(owns(1, 3, n, false, name), r1, "{name} on rank 1");
    }
}

#[test]
fn a_drafting_mtp_head_loads_on_rank_1() {
    let n = 8;
    for (name, r0, r1) in [
        // The head reads the embeddings of the tokens it drafts from: a copy on rank 1.
        ("token_embd.weight", true, true),
        ("rope_freqs.weight", true, true),
        ("output.weight", false, true),
        ("blk.2.attn_q.weight", true, false),
        ("blk.3.attn_q.weight", false, true),
        ("blk.8.nextn.eh_proj.weight", false, true),
        ("blk.8.ffn_down_exps.weight", false, true),
    ] {
        assert_eq!(owns(0, 3, n, true, name), r0, "{name} on rank 0");
        assert_eq!(owns(1, 3, n, true, name), r1, "{name} on rank 1");
    }
}

fn tensor(name: &str, size: u64) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        ggml_type: crate::ggml::types::BF16,
        shape: vec![size],
        file_index: 0,
        offset: 0,
        size,
    }
}

#[test]
fn a_drafting_head_counts_on_rank_1() {
    let mut tensors = vec![
        tensor("token_embd.weight", 7),
        tensor("output.weight", 5),
        tensor("blk.4.nextn.eh_proj.weight", 30),
    ];
    for il in 0..4 {
        tensors.push(tensor(&format!("blk.{il}.attn_q.weight"), 10));
    }
    // Without drafting the head is nowhere.
    let (bytes, first, last) = byte_shares(&tensors, 4, false);
    assert_eq!((bytes.clone(), first, last), (vec![10; 4], 7, 5));
    assert_eq!(balanced_split(&bytes, first, last), 2);
    // With it, rank 1 holds the head's layer and a copy of the embeddings, and the balanced
    // split gives rank 0 a layer more.
    let (bytes, first, last) = byte_shares(&tensors, 4, true);
    assert_eq!((bytes.clone(), first, last), (vec![10; 4], 7, 42));
    assert_eq!(balanced_split(&bytes, first, last), 3);
}

#[test]
fn balanced_split_evens_the_bytes() {
    // Rank 0 has the embeddings, rank 1 the head.
    let bytes = [10u64; 8];
    assert_eq!(balanced_split(&bytes, 0, 0), 4);
    assert_eq!(balanced_split(&bytes, 20, 0), 3);
    assert_eq!(balanced_split(&bytes, 0, 20), 5);
}

#[test]
fn fitting_split_or_none() {
    let bytes = [10u64; 8];
    assert_eq!(fitting_split(&bytes, 0, 0, 40), Some(4));
    // The head on rank 1 leaves it room for 3 layers: only a split at 5 fits.
    assert_eq!(fitting_split(&bytes, 0, 15, 50), Some(5));
    // Several fit: the most even one.
    assert_eq!(fitting_split(&bytes, 0, 0, 60), Some(4));
    assert_eq!(fitting_split(&bytes, 0, 0, 30), None);
}

#[test]
fn shares_take_rank_bytes() {
    let bytes = [1u64, 2, 3, 4];
    assert_eq!(rank_bytes(&bytes, 10, 20, 2), (13, 27));
}

/// What `--spec mtp` over two nodes runs (#560), on the tiny qwen4exp over a loopback link:
/// rank 0's steps over its layers, rank 1's ([`Follower`]) over the rest with the MTP head,
/// drafting from the rows its last step captured, and a verify step's commit mirrored. The
/// drafts (and their confidences) and the verify logits are one machine's, bit for bit.
#[test]
fn rank_1_drafts_as_one_machine_does() {
    use crate::ggml::Backend;
    use crate::gguf::GgufFile;
    use crate::model::core::{Group, Logits, Remote, SeqInput};
    const SPLIT: usize = 2;
    let dir = tempfile::tempdir().unwrap();
    crate::model::qwen4::tiny::write(
        dir.path().join("tiny.gguf"),
        Some(&dir.path().join("mtp.gguf")),
        11,
    )
    .unwrap();
    // An executor over `layers` (to the last: with the MTP head and the last layer's rows).
    let exec = |dir: &std::path::Path, layers: Option<Range<usize>>| {
        let mut b = Backend::cpu().unwrap();
        b.set_threads(2);
        b.set_use_ref(true);
        let mut m = Model::load(&GgufFile::open(dir.join("tiny.gguf")).unwrap(), |_| {
            b.buffer_type()
        })
        .unwrap();
        let n = m.hp.n_layer;
        let split = layers.is_some();
        let layers = layers.map_or(0..n, |r| r.start..r.end.min(n));
        let head = layers.end == n;
        if head {
            m.load_mtp(&GgufFile::open(dir.join("mtp.gguf")).unwrap(), |_| {
                b.buffer_type()
            })
            .unwrap();
        }
        let mut e = Executor::new_staged(m, vec![(b, layers)], 16, 4, 0).unwrap();
        if split {
            // As the engine runs a split model (`Executor::commit` mirrors its commits).
            e.no_deferred_commits();
        }
        e.enable_spec(4).unwrap();
        if head {
            e.enable_mtp().unwrap();
            e.capture_hidden(vec![n - 1]);
        }
        e
    };
    let prompt: Vec<u32> = "drafts over two nodes".bytes().map(u32::from).collect();
    const BLOCKS: &[u32] = &[0, 1, 2, 3];
    fn group(tokens: &[u32], n_past: usize, logits: Logits, verify: bool) -> Group<'_> {
        Group {
            slot: 0,
            seqs: vec![SeqInput {
                tokens,
                n_past,
                blocks: BLOCKS,
                ring: 0,
                logits,
                media: None,
            }],
            verify,
        }
    }
    let half = prompt.len() / 2;
    let bonus = 5u32;
    // One sequence: its prompt in two chunks (the first only taken in), drafts after the
    // second, a verify step of the bonus token and those drafts keeping two, and drafts after
    // it. `draft` drafts after a step from its rows (or rank 1's): per round, the drafts.
    type Drafts = Vec<(u32, u32)>;
    let run =
        |e: &mut Executor,
         draft: &mut dyn FnMut(&mut Executor, &[u32], usize, usize, &[f32], usize) -> Drafts,
         verify_with: Option<&Drafts>|
         -> (Vec<Drafts>, Vec<u32>) {
            let mut rounds = Vec::new();
            let o = e
                .step(&[group(&prompt[..half], 0, Logits::None, false)])
                .unwrap();
            let rows = o.hidden.first().cloned().unwrap_or_default();
            assert!(draft(e, &prompt, 0, half, &rows, 0).is_empty());
            let o = e
                .step(&[group(&prompt[half..], half, Logits::Last, false)])
                .unwrap();
            let rows = o.hidden.first().cloned().unwrap_or_default();
            let mut history = prompt.clone();
            history.push(bonus);
            let first = draft(e, &history, half, prompt.len() - half, &rows, 3);
            assert_eq!(first.len(), 3, "p_min 0 drafts all it may");
            rounds.push(first.clone());
            let ds = verify_with.unwrap_or(&first);
            let mut fed = vec![bonus];
            fed.extend(ds.iter().map(|d| d.0));
            let o = e
                .step(&[group(&fed, prompt.len(), Logits::All, true)])
                .unwrap();
            let logits: Vec<u32> = o.rows.concat().iter().map(|x| x.to_bits()).collect();
            let rows = o.hidden.first().cloned().unwrap_or_default();
            e.commit(&[(0, 2)]).unwrap();
            history.push(fed[1]);
            history.push(9);
            rounds.push(draft(e, &history, prompt.len(), 2, &rows, 3));
            (rounds, logits)
        };
    let bits = |d: &Draft| -> Drafts {
        d.tokens
            .iter()
            .zip(&d.confidence)
            .map(|(&t, p)| (t, p.to_bits()))
            .collect()
    };
    // One machine.
    let mut one = exec(dir.path(), None);
    let mut mtp = MtpDrafter::new(0.0);
    let mut local =
        |e: &mut Executor, tokens: &[u32], start: usize, ran: usize, rows: &[f32], max: usize| {
            let w = e.mtp_width();
            let mut out = Draft::default();
            let mut batch = [DraftRequest {
                seq: 1,
                tokens,
                start,
                ran,
                hidden: vec![&rows[..ran * w]],
                max,
                out: &mut out,
            }];
            mtp.step(e, &mut batch, &[BLOCKS], &[0]).unwrap();
            bits(&out)
        };
    let (want, want_logits) = run(&mut one, &mut local, None);
    drop(one);
    // Two nodes over loopback.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let path = dir.path().to_path_buf();
    let rank1 = std::thread::spawn(move || {
        let e = exec(&path, Some(SPLIT..usize::MAX));
        let n = e.model().hp.n_layer;
        let mut f = Follower::new(e, Some(MtpDrafter::new(0.0)), (n, SPLIT), 16);
        let (stream, _) = listener.accept().unwrap();
        crate::node::serve_one(stream, &mut |req| {
            f.handle(req).unwrap_or_else(Reply::Error)
        })
        .unwrap();
    });
    let mut rank0 = exec(dir.path(), Some(0..SPLIT));
    let link = crate::node::Link::connect(&addr, std::time::Duration::from_secs(30)).unwrap();
    rank0.set_remote(Remote::new(link));
    let mut mtp = MtpDrafter::new(0.0);
    let mut remote =
        |e: &mut Executor, tokens: &[u32], start: usize, ran: usize, rows: &[f32], max: usize| {
            assert!(rows.is_empty(), "rank 0 captures nothing");
            let mut out = Draft::default();
            let mut batch = [DraftRequest {
                seq: 1,
                tokens,
                start,
                ran,
                hidden: Vec::new(),
                max,
                out: &mut out,
            }];
            mtp.step_remote(e, &mut batch, &[BLOCKS], &[0], &[0])
                .unwrap();
            bits(&out)
        };
    let (got, got_logits) = run(&mut rank0, &mut remote, Some(&want[0]));
    drop(rank0);
    rank1.join().unwrap();
    assert_eq!(got, want, "drafts and confidences");
    assert_eq!(got_logits, want_logits, "verify logits");
}
