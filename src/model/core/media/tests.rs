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

/// Gemma 4 (#1217): every token at its own index, images or not; an image token's last key on
/// the sliding-window layers is its image's last token.
#[test]
fn sequential_positions_and_image_ends() {
    let m = Media::sequential(vec![span(2, 3, 2, 1), span(10, 1, 1, 1)], 1);
    assert_eq!(m.layout(), ImageLayout::Sequential);
    for i in 0..14 {
        assert_eq!(m.rope(i), [i as i32, i as i32, i as i32, 0]);
    }
    assert_eq!(m.span_end(1), None);
    assert_eq!(m.span_end(2), Some(7));
    assert_eq!(m.span_end(7), Some(7));
    assert_eq!(m.span_end(8), None);
    assert_eq!(m.span_end(10), Some(10));
    assert_eq!(m.row(3), Some(&[1.0f32][..]));
    // The M-RoPE layout's spans end at the same tokens.
    let q = Media::new(vec![span(2, 3, 2, 1)], 1);
    assert_eq!(q.layout(), ImageLayout::MRope);
    assert_eq!(q.span_end(5), Some(7));
}

/// A prompt chunk never ends inside an image: it stops before an image it reaches, or takes
/// the whole of one it starts with when that fits `cap`.
#[test]
fn chunks_do_not_cut_images() {
    let m = Media::sequential(vec![span(10, 4, 5, 1), span(40, 2, 2, 1)], 1);
    // Ends before, at the start of, or right after an image: as asked.
    assert_eq!(m.uncut(0, 8, 64), 8);
    assert_eq!(m.uncut(0, 10, 64), 10);
    assert_eq!(m.uncut(0, 30, 64), 30);
    // Inside an image the chunk reaches: back to the image's start.
    assert_eq!(m.uncut(0, 15, 64), 10);
    assert_eq!(m.uncut(30, 42, 64), 40);
    // Inside an image the chunk starts with (or resumes in): on to its end, within `cap`.
    assert_eq!(m.uncut(10, 15, 64), 30);
    assert_eq!(m.uncut(12, 15, 64), 30);
    assert_eq!(m.uncut(10, 15, 20), 30);
    assert_eq!(m.uncut(10, 15, 19), 15, "longer than cap: cut");
    assert_eq!(Media::default().uncut(0, 5, 1), 5);
}

/// A step's micro-chunks never cut an image either; with no image cut, the pieces are the
/// ones asked for (text-only steps keep theirs).
#[test]
fn micro_chunks_do_not_cut_images() {
    // An image of 300 tokens at 3,000: the cut at 3,072 moves back to its start, the image
    // is a piece of its own and the next one goes on to the next cut.
    let m = Media::sequential(vec![span(3000, 20, 15, 1)], 1);
    assert_eq!(
        m.uncut_sizes(0, &[3072, 3072, 784]),
        vec![3000, 300, 2844, 784]
    );
    // The same prompt from n_past 100: the image is at 2,900..3,200 of the chunk.
    assert_eq!(m.uncut_sizes(100, &[3072, 3072]), vec![2900, 300, 2944]);
    // No cut inside the image: the same pieces.
    assert_eq!(
        m.uncut_sizes(0, &[2048, 2048, 2048]),
        vec![2048, 2048, 2048]
    );
    assert_eq!(m.uncut_sizes(3300, &[3072, 1000]), vec![3072, 1000]);
    // A chunk that starts inside the image: its cuts move past its end.
    assert_eq!(m.uncut_sizes(3100, &[50, 50, 900]), vec![200, 800]);
    // An image longer than the largest piece (4,000 tokens > 3,072): cut where the pieces
    // were, none longer.
    let m = Media::sequential(vec![span(1000, 100, 40, 1)], 1);
    for (from, sizes) in [
        (0, &[3072, 3072, 784][..]),
        (0, &[3072, 1928]),
        (1000, &[3072, 928]),
    ] {
        assert_eq!(m.uncut_sizes(from, sizes), sizes.to_vec());
    }
    // A chunk that starts inside it with less than a piece of it left: one piece to its end.
    assert_eq!(m.uncut_sizes(4000, &[500, 1500]), vec![1000, 1000]);
    assert_eq!(m.uncut_sizes(3000, &[500, 1500]), vec![500, 1500]);
    // Two images, each cut: each whole in one piece.
    let m = Media::sequential(vec![span(10, 4, 5, 1), span(40, 2, 2, 1)], 1);
    assert_eq!(m.uncut_sizes(0, &[16, 26, 8]), vec![10, 20, 10, 4, 6]);
    // No images: the same pieces.
    let sizes = [3072, 2048, 1024, 17];
    assert_eq!(Media::default().uncut_sizes(5, &sizes), sizes.to_vec());
    assert_eq!(m.uncut_sizes(0, &[7]), vec![7]);
}

/// A lone prompt over several stages with rings: the step is longer than the ring step and
/// its micro-chunks are at most the ring step (512 here). An image after a full window of
/// text stays whole when it fits a piece, and is cut where the pieces were when it does not
/// (700 tokens): no piece is longer than 512, so none overwrites a key its first row still
/// reads (a 700-token piece would).
#[test]
fn micro_chunks_fit_the_rings() {
    let (window, step) = (1024, 512);
    let positions = crate::model::core::KvRing::for_window(window, step, 1).positions();
    // A piece's last key lands on the cell of the key `positions` before it; its first row
    // reads `window - 1` keys back, so the two must be under `positions` apart.
    let fits = |len: usize| len + window - 2 < positions;
    assert!(fits(step) && !fits(700));
    let micro = vec![512; 8];
    // 1,300 text tokens (more than the window), then a 700-token image (28 × 25), then text.
    let m = Media::sequential(vec![span(1300, 28, 25, 1)], 1);
    assert_eq!(m.uncut_sizes(0, &micro), micro);
    // A 300-token image (20 × 15) there: a piece of its own.
    let m = Media::sequential(vec![span(1300, 20, 15, 1)], 1);
    let v = m.uncut_sizes(0, &micro);
    assert_eq!(v, vec![512, 512, 276, 300, 448, 512, 512, 512, 512]);
    assert!(v.iter().all(|&len| fits(len)));
    // The same from n_past 2,000 (the window already full of text before the step).
    let m = Media::sequential(vec![span(2300, 20, 15, 1)], 1);
    let v = m.uncut_sizes(2000, &micro);
    assert_eq!(v, vec![300, 300, 424, 512, 512, 512, 512, 512, 512]);
    assert!(v.iter().all(|&len| fits(len)));
    // Its 512-token piece from the image's start; a 513-token image is cut where the pieces
    // were.
    let m = Media::sequential(vec![span(2300, 32, 16, 1)], 1);
    assert_eq!(m.uncut_sizes(2000, &micro)[..3], [300, 512, 212]);
    let m = Media::sequential(vec![span(2300, 513, 1, 1)], 1);
    assert_eq!(m.uncut_sizes(2000, &micro), micro);
}
