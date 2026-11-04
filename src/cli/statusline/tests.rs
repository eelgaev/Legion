use super::*;

#[derive(Clone, Default)]
struct Shared(Arc<Mutex<Vec<u8>>>);

impl Write for Shared {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn log_writer_clears_then_redraws_the_status_line() {
    let terminal = Arc::new(Terminal::default());
    terminal.state.lock().unwrap().line = Some("decode 1".into());
    let output = Shared::default();
    {
        let mut writer = StatusWriter::new(output.clone(), terminal);
        writer.write_all(b"INFO request done\n").unwrap();
    }
    assert_eq!(
        String::from_utf8(output.0.lock().unwrap().clone()).unwrap(),
        "\r\x1b[2KINFO request done\n\r\x1b[2Kdecode 1"
    );
}

#[test]
fn long_lines_are_elided_to_the_current_width() {
    assert_eq!(truncate("prefill 1 | decode 2", 12), "prefill 1 |…");
    assert_eq!(truncate("abc", 3), "abc");
}

/// Two threads logging while a status line is up: every log line comes out whole, between a
/// clear and a redraw of the status line, never split by the other thread's.
#[test]
fn concurrent_log_lines_never_interleave_with_the_status_line() {
    let terminal = Arc::new(Terminal::default());
    terminal.state.lock().unwrap().line = Some("decode 2".into());
    let output = Shared::default();
    let threads: Vec<_> = (0..2)
        .map(|t| {
            let (output, terminal) = (output.clone(), terminal.clone());
            std::thread::spawn(move || {
                for i in 0..200 {
                    let mut writer = StatusWriter::new(output.clone(), terminal.clone());
                    // As a formatter may: the event in pieces, its newline last.
                    write!(writer, "thread {t} ").unwrap();
                    write!(writer, "line {i}").unwrap();
                    writer.write_all(b"\n").unwrap();
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    let text = String::from_utf8(output.0.lock().unwrap().clone()).unwrap();
    let parts: Vec<&str> = text.split(CLEAR_LINE).skip(1).collect();
    assert_eq!(parts.len(), 800);
    let mut seen = [0usize; 2];
    for pair in parts.chunks(2) {
        assert_eq!(pair[1], "decode 2");
        let (t, i) = pair[0]
            .strip_prefix("thread ")
            .and_then(|r| r.strip_suffix('\n'))
            .and_then(|r| r.split_once(" line "))
            .unwrap_or_else(|| panic!("a broken log line: {:?}", pair[0]));
        let t: usize = t.parse().unwrap();
        // Each thread's lines in order.
        assert_eq!(i.parse::<usize>().unwrap(), seen[t]);
        seen[t] += 1;
    }
    assert_eq!(seen, [200, 200]);
}

#[test]
fn without_a_status_line_logs_pass_through() {
    let output = Shared::default();
    {
        let mut writer = StatusWriter::new(output.clone(), Arc::new(Terminal::default()));
        writer.write_all(b"INFO a\nINFO b").unwrap();
    }
    assert_eq!(output.0.lock().unwrap().as_slice(), b"INFO a\nINFO b");
}

#[test]
fn mem_text_shows_device_bandwidth_and_roofline_share() {
    assert_eq!(
        mem_text((0, 1.0), (187_000_000_000, 2.0), 233.0).as_deref(),
        Some("  mem 187 GB/s (80%)")
    );
    assert_eq!(
        mem_text((0, 0.0), (5_000_000_000, 1.0), 0.0).as_deref(),
        Some("  mem 5 GB/s")
    );
    assert_eq!(mem_text((10, 1.0), (10, 1.0), 233.0), None);
}

#[test]
fn queue_text_only_with_priorities_or_several_keys() {
    use legion::engine::queue::{Scheduling, SchedulingPolicy, ShareKey, WaitQueue};
    let stats = |reqs: &[(i32, Option<&str>)]| {
        let mut q = WaitQueue::new(SchedulingPolicy::Priority);
        for &(priority, key) in reqs {
            let share_key = key.map(ShareKey::of);
            q.push(
                (),
                Scheduling {
                    priority,
                    share_key,
                },
            );
        }
        EngineStats {
            waiting: q.len() as u32,
            waiting_classes: q.classes(),
            waiting_keys: q.waiting_keys(),
            ..Default::default()
        }
    };
    // Today's line: priority 0, no keys.
    assert_eq!(queue_text(&stats(&[])), None);
    assert_eq!(queue_text(&stats(&[(0, None), (0, None)])), None);
    assert_eq!(queue_text(&stats(&[(0, Some("a")), (0, Some("a"))])), None);
    assert_eq!(
        queue_text(&stats(&[(10, None), (-1, None), (10, None)])).as_deref(),
        Some(" (p-1:1 p10:2)")
    );
    assert_eq!(
        queue_text(&stats(&[(0, Some("a")), (0, Some("b"))])).as_deref(),
        Some(" (2 keys)")
    );
    assert_eq!(
        queue_text(&stats(&[(1, Some("a")), (0, Some("b"))])).as_deref(),
        Some(" (p0:1 p1:1, 2 keys)")
    );
}
