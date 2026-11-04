use super::*;

const MEMINFO: &str = "MemTotal:       127535808 kB\n\
MemFree:         5242880 kB\n\
MemAvailable:   100000000 kB\n\
Buffers:            1024 kB\n\
SwapTotal:        16384 kB\n\
SwapFree:         16000 kB\n\
HugePages_Total:       0\n\
HugePages_Free:        0\n\
Hugepagesize:       2048 kB\n";

#[test]
fn mem_available_without_huge_pages() {
    assert_eq!(free_bytes(MEMINFO.as_bytes()), Some(100_000_000 * 1024));
}

#[test]
fn huge_page_pool_gives_its_free_pages() {
    let text = MEMINFO
        .replace("HugePages_Total:       0", "HugePages_Total:     512")
        .replace("HugePages_Free:        0", "HugePages_Free:      100");
    assert_eq!(free_bytes(text.as_bytes()), Some(100 * 2048 * 1024));
}

#[test]
fn no_huge_page_lines_keep_mem_available() {
    let text: String = MEMINFO
        .lines()
        .filter(|l| !l.starts_with("Huge"))
        .map(|l| format!("{l}\n"))
        .collect();
    assert_eq!(free_bytes(text.as_bytes()), Some(100_000_000 * 1024));
}

#[test]
fn nothing_usable_keeps_cuda_free_bytes() {
    assert_eq!(free_bytes(b""), None);
    assert_eq!(free_bytes(b"MemTotal: 1 kB\n"), None);
    assert_eq!(free_bytes(b"MemAvailable: 0 kB\n"), None);
    assert_eq!(free_bytes(b"MemAvailable: kB\n"), None);
}

#[test]
fn reads_only_what_ggml_reads() {
    // ggml reads 2047 bytes: a MemAvailable line past them is not seen.
    let text = format!("{}\nMemAvailable: 7 kB\n", "x".repeat(READ_MAX));
    assert_eq!(free_bytes(text.as_bytes()), None);
    // A line cut at the end parses its digits so far, as sscanf on the NUL-terminated buffer.
    let pad = "y".repeat(READ_MAX - "\nMemAvailable: 12".len());
    let text = format!("{pad}\nMemAvailable: 12345 kB\n");
    assert_eq!(free_bytes(text.as_bytes()), Some(12 * 1024));
}

#[test]
fn field_parses_like_sscanf() {
    assert_eq!(field(b"MemAvailable:   42 kB", b"MemAvailable:"), Some(42));
    assert_eq!(field(b"MemAvailable:42", b"MemAvailable:"), Some(42));
    assert_eq!(field(b"MemAvailable: -3 kB", b"MemAvailable:"), Some(-3));
    assert_eq!(field(b"MemAvailabl: 42", b"MemAvailable:"), None);
    assert_eq!(field(b" MemAvailable: 42", b"MemAvailable:"), None);
}
