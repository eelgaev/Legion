use super::*;

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
