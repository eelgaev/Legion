use super::*;

#[test]
fn faults_in_and_locks_a_file_mapping() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.bin");
    let data: Vec<u8> = (0..3 * PAGE + 123).map(|i| (i * 7) as u8).collect();
    std::fs::write(&path, &data).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    // SAFETY: a private file of this test, not modified while mapped.
    let map = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let topo = Topology::detect().ok();
    let node = topo.as_ref().map(|t| (t, t.nodes[0].id));
    let r = make_resident(&map[100..], node);
    assert_eq!(r.bytes, 3 * PAGE + 123);
    // Locking needs RLIMIT_MEMLOCK room; either way the bytes read back unchanged.
    assert_eq!(r.locked, r.lock_error.is_none());
    assert_eq!(&map[..], &data[..]);
    assert!(make_resident(&[], None).locked);
}
