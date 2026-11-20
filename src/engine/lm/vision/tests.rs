use super::*;

/// Qwen's placeholder repeats per image token; Gemma 4's too, between `<|image>` and
/// `<image|>`, its tokens starting after `<|image>`.
#[test]
fn placeholders_expand_to_image_tokens() {
    let qwen = Marks {
        pad: 9,
        open: None,
        close: None,
        layout: ImageLayout::MRope,
    };
    let (t, starts) = expand_tokens(&[1, 9, 2, 9], &qwen, &[(2, 1), (1, 3)]);
    assert_eq!(t, [1, 9, 9, 2, 9, 9, 9]);
    assert_eq!(starts, [1, 4]);
    let gemma = Marks {
        pad: 9,
        open: Some(7),
        close: Some(8),
        layout: ImageLayout::Sequential,
    };
    let (t, starts) = expand_tokens(&[1, 9, 2, 9], &gemma, &[(2, 1), (1, 3)]);
    assert_eq!(t, [1, 7, 9, 9, 8, 2, 7, 9, 9, 9, 8]);
    assert_eq!(starts, [2, 7]);
    assert_eq!(expand_tokens(&[1, 2], &gemma, &[]), (vec![1, 2], vec![]));
}

/// Image keys sit above any vocabulary, differ between the tokens of an image and between
/// images, and are the same for the same image.
#[test]
fn image_keys_tell_images_apart() {
    let a: Vec<u32> = (0..300).map(|i| image_key(1, i)).collect();
    let b: Vec<u32> = (0..300).map(|i| image_key(2, i)).collect();
    assert!(a.iter().chain(&b).all(|&k| k >= 0x8000_0000));
    let mut distinct = a.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), a.len());
    assert!(a.iter().zip(&b).all(|(x, y)| x != y));
    assert_eq!(a, (0..300).map(|i| image_key(1, i)).collect::<Vec<_>>());
}

/// The row cache finds an image by its hash and grid, drops the least recently used images to
/// stay within its bytes, and skips one larger than all of them.
#[test]
fn row_cache_keeps_recent_images_within_its_bytes() {
    let rows = |n: usize, v: f32| -> Arc<[f32]> { vec![v; n].into() };
    let mut c = RowCache::default();
    c.put(1, 2, 2, rows(4, 1.0), 32);
    c.put(2, 2, 2, rows(4, 2.0), 32);
    assert_eq!(c.get(1, 2, 2).unwrap()[0], 1.0);
    assert!(c.get(1, 4, 1).is_none(), "another grid is another image");
    // 1 was used last, so 2 goes.
    c.put(3, 2, 2, rows(4, 3.0), 32);
    assert!(c.get(2, 2, 2).is_none());
    assert_eq!(c.get(3, 2, 2).unwrap()[0], 3.0);
    assert_eq!(c.get(1, 2, 2).unwrap()[0], 1.0);
    assert_eq!(c.bytes, 32);
    c.put(4, 4, 4, rows(16, 4.0), 32);
    assert!(c.get(4, 4, 4).is_none());
    assert_eq!(c.bytes, 32);
}
