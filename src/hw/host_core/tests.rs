use super::*;

#[test]
fn an_unmarked_thread_keeps_the_given_set() {
    std::thread::spawn(|| {
        assert_eq!(off_host_core(&[4, 5, 6]), [4, 5, 6]);
        assert!(off_host_core(&[]).is_empty());
    })
    .join()
    .unwrap();
}

#[test]
fn a_marked_thread_leaves_its_own_cpu_out() {
    std::thread::spawn(|| {
        mark_host_thread();
        let all = current_affinity().unwrap_or_default();
        let Some(&host) = all.first() else {
            return;
        };
        // Held on one CPU, the host cannot move while the set is worked out.
        pin_current_thread(&[host]).expect("pin");
        let off = off_host_core(&all);
        assert!(!off.is_empty());
        assert!(off.iter().all(|c| all.contains(c)));
        if all
            .iter()
            .any(|c| !siblings(host).contains(c) && *c != host)
        {
            assert!(!off.contains(&host));
            assert!(off.iter().all(|c| !siblings(host).contains(c)));
        } else {
            // Nothing left without the host's core: the given set stays.
            assert_eq!(off, all);
        }
    })
    .join()
    .unwrap();
}

#[test]
fn an_empty_pin_is_a_no_op() {
    std::thread::spawn(|| {
        let before = current_affinity().expect("affinity");
        pin_helper(&[]);
        assert_eq!(current_affinity().expect("affinity"), before);
    })
    .join()
    .unwrap();
}
