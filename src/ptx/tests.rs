use super::*;

const SAMPLE: &str = "// a comment .entry not_this\n.version 7.0\n.target sm_70\n.address_size 64\n\n\
.visible .entry first(\n\t.param .u64 p\n)\n{\n\tret;\n}\n\
.visible .entry second_2 (\n)\n{\n\tret;\n}\n";

#[test]
fn entries_skip_comments_and_keep_order() {
    assert_eq!(entries(SAMPLE), vec!["first", "second_2"]);
}

#[test]
fn target_reads_the_sm_version() {
    let m = Module::new("sample", SAMPLE);
    assert_eq!(m.target(), Some(70));
    let m = Module::new("none", ".version 7.0\n");
    assert_eq!(m.target(), None);
    let m = Module::new("suffix", ".target sm_90a, debug\n");
    assert_eq!(m.target(), Some(90));
}
