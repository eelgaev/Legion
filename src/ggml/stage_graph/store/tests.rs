use super::*;

fn kept(words: &[u64], val: u32, at: Instant) -> Kept<u32> {
    Kept::new(words.to_vec(), val, at)
}

#[test]
fn hash_sees_every_word_and_the_length() {
    let w: Vec<u64> = (0..37).map(|i| i * 0x1234_5678 + 7).collect();
    let h = hash(&w);
    assert_eq!(h, hash(&w.clone()), "the same words, the same hash");
    for i in 0..w.len() {
        let mut v = w.clone();
        v[i] ^= 1 << (i % 64);
        assert_ne!(hash(&v), h, "word {i} changed");
    }
    assert_ne!(hash(&w[..36]), h, "one word fewer");
    let mut longer = w.clone();
    longer.push(0);
    assert_ne!(hash(&longer), h, "a zero word more");
    assert_ne!(hash(&[]), hash(&[0]));
}

/// A graph built again with the words of one before finds that one's set; other words, or
/// the same words on another device, do not.
#[test]
fn a_set_is_found_by_its_words_on_its_device() {
    let mut s = Store::default();
    let now = Instant::now();
    let w = [3, 1, 4, 1, 5];
    let h = hash(&w);
    s.insert(10, h, kept(&w, 7, now), now, 64);
    assert!(s.take_words(11, h, &w).is_none(), "another device");
    assert!(s.take_words(10, h, &[3, 1, 4, 1, 6]).is_none());
    // The same key with other words (a hash collision) is no match.
    assert!(s.take_words(10, h, &[2, 7, 1, 8]).is_none());
    let k = s.take_words(10, h, &w).expect("the same words");
    assert_eq!(k.val, 7);
    assert!(s.take_words(10, h, &w).is_none(), "taken out while used");
    s.put(10, h, k, now);
    assert!(s.take_words(10, h, &w).is_some(), "put back");
}

/// A graph names its own set by id: once another set replaced it under the same key, the
/// graph plans its own again.
#[test]
fn a_set_is_taken_back_by_its_id_only() {
    let mut s = Store::default();
    let now = Instant::now();
    let w = [9, 9];
    let h = hash(&w);
    let a = kept(&w, 1, now);
    let id_a = a.id;
    s.insert(0, h, a, now, 64);
    let a = s.take_id(0, h, id_a).expect("its own set");
    s.put(0, h, a, now);
    let b = kept(&w, 2, now);
    let id_b = b.id;
    assert_ne!(id_a, id_b);
    s.insert(0, h, b, now, 64);
    assert!(s.take_id(0, h, id_a).is_none(), "replaced");
    assert_eq!(s.take_id(0, h, id_b).map(|k| k.val), Some(2));
}

/// Past the cap, a device's least recently used sets go; other devices' sets stay, and so
/// does every set with no cap.
#[test]
fn the_cap_drops_the_least_recently_used_of_the_device() {
    let mut s = Store::default();
    let t0 = Instant::now();
    let at = |i: u64| t0 + Duration::from_millis(i);
    s.insert(1, 100, kept(&[100], 0, at(0)), at(0), 3);
    for i in 0..3u64 {
        s.insert(0, i, kept(&[i], i as u32, at(i)), at(i), 3);
    }
    assert_eq!(s.len(0), 3);
    // Set 0, the oldest, is used again: set 1 is the least recently used now.
    let id = s.sets[&(0, 0)].id;
    let k = s.take_id(0, 0, id).unwrap();
    s.put(0, 0, k, at(10));
    s.insert(0, 3, kept(&[3], 3, at(11)), at(11), 3);
    assert_eq!(s.len(0), 3);
    assert!(
        s.take_words(0, 1, &[1]).is_none(),
        "the least recently used went"
    );
    for i in [0, 2, 3] {
        assert!(s.take_words(0, i, &[i]).is_some(), "set {i} stays");
    }
    assert_eq!(s.len(1), 1, "the other device's set stays");
    for i in 10..20u64 {
        s.insert(2, i, kept(&[i], 0, at(i)), at(i), 0);
    }
    assert_eq!(s.len(2), 10, "no cap");
}

#[test]
fn sets_unused_for_a_while_are_swept() {
    let mut s = Store::default();
    let t0 = Instant::now();
    s.insert(0, 1, kept(&[1], 0, t0), t0, 0);
    let later = t0 + Duration::from_secs(4);
    s.insert(0, 2, kept(&[2], 0, later), later, 0);
    s.sweep(t0 + Duration::from_secs(10), Duration::from_secs(10));
    assert!(s.take_words(0, 1, &[1]).is_none(), "unused for 10 s");
    assert!(s.take_words(0, 2, &[2]).is_some());
}

/// A set captured ahead is seen by its words on its device only (`has`), without being taken:
/// the graph's first run takes it then.
#[test]
fn has_sees_a_set_by_its_words_and_leaves_it() {
    let mut s = Store::default();
    let now = Instant::now();
    let w = [2, 7, 1, 8];
    let h = hash(&w);
    assert!(!s.has(10, h, &w), "nothing kept yet");
    s.insert(10, h, kept(&w, 5, now), now, 64);
    assert!(s.has(10, h, &w));
    assert!(!s.has(11, h, &w), "another device");
    assert!(
        !s.has(10, h, &[2, 7, 1, 9]),
        "other words under the same key"
    );
    assert_eq!(s.len(10), 1, "still kept");
    assert_eq!(s.take_words(10, h, &w).expect("kept").val, 5);
    assert!(!s.has(10, h, &w), "taken out");
}
