use super::*;

fn span(start: usize, nx: usize, ny: usize, n_embd: usize) -> ImageSpan {
    let rows: Vec<f32> = (0..nx * ny * n_embd).map(|i| i as f32).collect();
    ImageSpan {
        start,
        nx,
        ny,
        rows: rows.into(),
    }
}

/// llama.cpp's layout for `test-1.jpeg` (20 × 15 merged patches) after the 4 tokens
/// `<|im_start|>user\n<|vision_start|>` and before 13 text tokens: the image's tokens all at
/// `t = 4`, rows and columns from 4, the text after it from `4 + max(20, 15)`, and the next
/// position after the 317 tokens 37 (mtmd's `n_past`).
#[test]
fn positions_follow_mtmd_mrope() {
    let m = Media::new(vec![span(4, 20, 15, 2)], 2);
    assert_eq!(m.rope(0), [0, 0, 0, 0]);
    assert_eq!(m.rope(3), [3, 3, 3, 0]);
    assert_eq!(m.rope(4), [4, 4, 4, 0]);
    assert_eq!(m.rope(4 + 19), [4, 4, 23, 0]);
    assert_eq!(m.rope(4 + 20), [4, 5, 4, 0]);
    assert_eq!(m.rope(4 + 299), [4, 18, 23, 0]);
    assert_eq!(m.rope(304), [24, 24, 24, 0]);
    assert_eq!(m.rope(316), [36, 36, 36, 0]);
    assert_eq!(m.rope(317), [37, 37, 37, 0]);
    // A tall image: max(nx, ny) is the rows'.
    let m = Media::new(vec![span(0, 2, 5, 1)], 1);
    assert_eq!(m.rope(9), [0, 4, 1, 0]);
    assert_eq!(m.rope(10), [5, 5, 5, 0]);
}

/// Two images: each saves `nx × ny − max(nx, ny)` positions for everything after it.
#[test]
fn later_images_start_where_the_text_before_them_ends() {
    let m = Media::new(vec![span(1, 3, 2, 1), span(9, 2, 2, 1)], 1);
    // Text at 7, 8: 7 - (6 - 3) = 4.
    assert_eq!(m.rope(7), [4, 4, 4, 0]);
    assert_eq!(m.rope(8), [5, 5, 5, 0]);
    // The second image starts at position 6.
    assert_eq!(m.rope(9), [6, 6, 6, 0]);
    assert_eq!(m.rope(12), [6, 7, 7, 0]);
    // After both: 13 - 3 - 2 = 8.
    assert_eq!(m.rope(13), [8, 8, 8, 0]);
}

#[test]
fn rows_and_counts() {
    let m = Media::new(vec![span(2, 2, 2, 3), span(10, 1, 1, 3)], 3);
    assert_eq!(m.row(1), None);
    assert_eq!(m.row(2), Some(&[0.0, 1.0, 2.0][..]));
    assert_eq!(m.row(5), Some(&[9.0, 10.0, 11.0][..]));
    assert_eq!(m.row(6), None);
    assert_eq!(m.row(10), Some(&[0.0, 1.0, 2.0][..]));
    assert_eq!(m.image_tokens(0..20), 5);
    assert_eq!(m.image_tokens(3..11), 4);
    assert_eq!(m.image_tokens(6..10), 0);
    assert_eq!(m.first_image(), Some(2));
    assert!(Media::default().is_empty());
}
