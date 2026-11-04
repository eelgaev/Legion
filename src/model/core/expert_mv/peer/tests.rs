use super::*;

#[test]
fn pairs_by_host_node() {
    // An AC922: GPUs 0, 1 on node 0, 2, 3 on node 8.
    let ac922 = [Some(0), Some(0), Some(8), Some(8)];
    assert_eq!(pairs(&ac922), [Some(1), Some(0), Some(3), Some(2)]);
    // Two GPUs on different sockets, or nodes unknown: no peer.
    assert_eq!(pairs(&[Some(0), Some(8)]), [None, None]);
    assert_eq!(pairs(&[None, None]), [None, None]);
    // A lone GPU on its node.
    assert_eq!(
        pairs(&[Some(0), Some(0), Some(8)]),
        [Some(1), Some(0), None]
    );
}

#[test]
fn share_and_slots() {
    assert_eq!(share(), (2, 1));
    assert_eq!(slots(16), 8);
    assert_eq!(slots(15), 8);
    assert_eq!(slots(1), 1);
    set_share(2, 3);
    assert_eq!(share(), (3, 2));
    assert_eq!(slots(16), 11);
    // Out of range: clamped.
    set_share(5, 3);
    assert_eq!(share(), (3, 3));
    set_share(0, 0);
    assert_eq!(share(), (2, 1));
}

#[test]
fn automatic_on_an_ac922_pair_only() {
    // Two V100s of one NVLink pair (GPUs 0, 1 on node 0, or 2, 3 on node 8).
    assert_eq!(auto(true, &[Some(0), Some(0)]), Ok(()));
    assert_eq!(auto(true, &[Some(8), Some(8)]), Ok(()));
    // Not an AC922 (a PCIe box, a GB10, a GH200).
    assert_eq!(auto(false, &[Some(0), Some(0)]), Err(AutoOff::Platform));
    // One GPU, three, or all four (measured no gain).
    assert_eq!(auto(true, &[Some(0)]), Err(AutoOff::Stages));
    assert_eq!(auto(true, &[]), Err(AutoOff::Stages));
    assert_eq!(
        auto(true, &[Some(0), Some(0), Some(8)]),
        Err(AutoOff::Stages)
    );
    assert_eq!(
        auto(true, &[Some(0), Some(0), Some(8), Some(8)]),
        Err(AutoOff::Stages)
    );
    // One GPU per socket, or the nodes unknown: no NVLink pair.
    assert_eq!(auto(true, &[Some(0), Some(8)]), Err(AutoOff::NotPaired));
    assert_eq!(auto(true, &[None, None]), Err(AutoOff::NotPaired));
}

#[test]
fn automatic_paired_on_every_ac922_pair() {
    // GLM: the two GPUs of one pair, or all four.
    assert_eq!(auto_paired(true, &[Some(0), Some(0)]), Ok(()));
    assert_eq!(
        auto_paired(true, &[Some(0), Some(0), Some(8), Some(8)]),
        Ok(())
    );
    assert_eq!(
        auto_paired(false, &[Some(0), Some(0), Some(8), Some(8)]),
        Err(AutoOff::Platform)
    );
    // A GPU left without a peer.
    assert_eq!(auto_paired(true, &[Some(0)]), Err(AutoOff::Stages));
    assert_eq!(
        auto_paired(true, &[Some(0), Some(0), Some(8)]),
        Err(AutoOff::Stages)
    );
    assert_eq!(
        auto_paired(true, &[Some(0), Some(8), Some(0), Some(16)]),
        Err(AutoOff::NotPaired)
    );
    assert_eq!(auto_paired(true, &[None, None]), Err(AutoOff::NotPaired));
}

#[test]
fn off_by_default_forced_and_automatic() {
    assert_eq!(forced(), None);
    assert!(!enabled());
    assert_eq!(forced_from(AUTO), None);
    assert_eq!(forced_from(FORCED_ON), Some(true));
    assert_eq!(forced_from(FORCED_OFF), Some(false));
    // Forcing it off changes no graph (it is off until a load turns it on), so this test may
    // do it beside others; it puts the state back.
    set_enabled(false);
    assert_eq!(forced(), Some(false));
    assert!(!enabled());
    FORCED.store(AUTO, Ordering::Relaxed);
    assert_eq!(forced(), None);
    assert_eq!(ctas(), CTAS as i32);
    {
        let _idle = Idle::new(true);
        assert!(IDLE.load(Ordering::Relaxed));
    }
    assert!(!IDLE.load(Ordering::Relaxed));
}

#[test]
fn share_parses_take_of_every() {
    let s: Share = "2/3".parse().unwrap();
    assert_eq!(s, Share { take: 2, every: 3 });
    assert_eq!(" 1 / 1 ".parse::<Share>(), Ok(Share { take: 1, every: 1 }));
    assert_eq!(
        "64/64".parse::<Share>(),
        Ok(Share {
            take: 64,
            every: 64
        })
    );
    for bad in ["", "2", "0/3", "4/3", "1/65", "x/2", "1/2/3", "-1/2"] {
        assert!(bad.parse::<Share>().is_err(), "{bad:?} parsed");
    }
}
