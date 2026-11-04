use super::*;

const VMSTAT: &str = "nr_free_pages 123\n\
allocstall_dma 1\n\
allocstall_normal 4\n\
allocstall_movable 2\n\
pgscan_direct 900\n\
pgscan_direct_throttle 7\n\
compact_stall 3\n\
pgmajfault 11\n\
thp_fault_alloc 5\n\
thp_fault_fallback 9\n\
bogus x\n\
\n";

#[test]
fn parses_the_counters_and_sums_allocstall_zones() {
    let v = VmStat::parse(VMSTAT);
    assert_eq!(
        v,
        VmStat {
            allocstall: 7,
            pgscan_direct: 900,
            compact_stall: 3,
            pgmajfault: 11,
            thp_fault_alloc: 5,
        }
    );
}

#[test]
fn missing_counters_are_zero() {
    assert_eq!(VmStat::parse(""), VmStat::default());
}

#[test]
fn since_is_the_growth_and_never_wraps() {
    let a = VmStat::parse(VMSTAT);
    let mut b = a;
    b.allocstall += 2;
    b.pgmajfault += 1;
    let d = b.since(&a);
    assert_eq!(d.allocstall, 2);
    assert_eq!(d.pgmajfault, 1);
    assert_eq!(d.compact_stall, 0);
    assert_eq!(a.since(&b), VmStat::default());
}
