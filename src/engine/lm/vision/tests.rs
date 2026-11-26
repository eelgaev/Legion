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

/// GLM-5.3-Flash (#1219): its template's `<|begin_of_image|><|image|><|end_of_image|>` keeps
/// its own marks and `<|image|>` repeats once per image token, positions one per token; with one
/// image and with two (text between them), each image's tokens start right after its
/// `<|begin_of_image|>`. A tokenizer without `<|image|>` is refused, and the architecture takes
/// the `glm5v` tower.
#[test]
fn glm_images_expand_between_their_marks() {
    use crate::gguf::{GgufFile, GgufWriter};
    let dir = tempfile::tempdir().unwrap();
    let vocab = |name: &str, user: &[&str]| {
        let path = dir.path().join(name);
        let mut w = GgufWriter::new();
        crate::model::qwen35::tiny::byte_tokenizer_with(&mut w, user, None).unwrap();
        w.write(&path).unwrap();
        Tokenizer::from_gguf(&GgufFile::open(path).unwrap()).unwrap()
    };
    let tok = vocab(
        "glm.gguf",
        &["<|begin_of_image|>", "<|image|>", "<|end_of_image|>"],
    );
    let marks = Marks::of(GLM_ARCH, &tok).unwrap();
    let id = |t: &str| tok.token_id(t).unwrap();
    let (open, pad, close) = (
        id("<|begin_of_image|>"),
        id("<|image|>"),
        id("<|end_of_image|>"),
    );
    assert_eq!(
        marks,
        Marks {
            pad,
            open: None,
            close: None,
            layout: ImageLayout::Sequential,
        }
    );
    let image = "<|begin_of_image|><|image|><|end_of_image|>";
    let one = tok.encode_with(&format!("a{image}b"), false, true);
    assert_eq!(one, [u32::from(b'a'), open, pad, close, u32::from(b'b')]);
    let (t, starts) = expand_tokens(&one, &marks, &[(3, 2)]);
    let mut want = vec![u32::from(b'a'), open];
    want.extend([pad; 6]);
    want.extend([close, u32::from(b'b')]);
    assert_eq!(t, want);
    assert_eq!(starts, [2]);
    let two = tok.encode_with(&format!("{image}c{image}"), false, true);
    let (t, starts) = expand_tokens(&two, &marks, &[(2, 1), (1, 3)]);
    let c = u32::from(b'c');
    assert_eq!(t, [open, pad, pad, close, c, open, pad, pad, pad, close]);
    assert_eq!(starts, [1, 6]);
    let err = Marks::of(GLM_ARCH, &vocab("plain.gguf", &[])).unwrap_err();
    assert!(err.contains("<|image|>"), "{err}");
    assert_eq!(projector_of(GLM_ARCH), Some(Projector::Glm5v));
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
