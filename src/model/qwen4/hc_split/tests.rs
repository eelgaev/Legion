//! The split mixer's switch and where it takes a mixer.

use super::*;

#[test]
fn the_switch_reads_off_up_and_both() {
    assert_eq!(mode_of(None), Mode::Up);
    assert_eq!(mode_of(Some("1")), Mode::Up);
    assert_eq!(mode_of(Some("0")), Mode::Off);
    assert_eq!(mode_of(Some(" 0 ")), Mode::Off);
    assert_eq!(mode_of(Some("2")), Mode::Both);
}

#[test]
fn rows_view_the_matrix_rows() {
    let ctx = Context::new(16, 16);
    let w = ctx.new_tensor(types::BF16, &[320, 10240]);
    let v = rows(&ctx, w, 2560 + 1280, 1280);
    assert_eq!(v.ne(), [320, 1280, 1, 1]);
    assert_eq!(v.nb()[1], w.nb()[1]);
    assert!(v.is_contiguous());
}

#[test]
fn the_cpu_graph_keeps_the_whole_mixer() {
    // On the CPU (and without CUDA) the dense op does not take the parts, so neither split does.
    let ctx = Context::new(16, 16);
    let down = ctx.new_tensor(types::BF16, &[10240, 320]);
    let up = ctx.new_tensor(types::BF16, &[320, 10240]);
    let xn = ctx.new_tensor(types::F32, &[10240, 3]);
    let lo = ctx.new_tensor(types::F32, &[320, 3]);
    if cfg!(feature = "cuda") {
        return;
    }
    assert!(up_mix(&ctx, up, lo, xn, 4, 0.25, (0, 0, 2)).is_none());
    assert!(down_lo(&ctx, down, up, xn, 0.25, (0, 0, 2)).is_none());
}
