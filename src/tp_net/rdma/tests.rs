use super::*;

use std::net::{Ipv4Addr, TcpListener};

#[test]
fn parse_gid_reads_sysfs_format() {
    let g = parse_gid("0000:0000:0000:0000:0000:ffff:c000:0201\n").unwrap();
    assert_eq!(
        g.0,
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0xc0, 0x00, 0x02, 0x01
        ]
    );
    assert!(parse_gid("").is_none());
    assert!(parse_gid("0000:0000:0000:0000:0000:ffff:c000").is_none());
    assert!(parse_gid("0000:0000:0000:0000:0000:ffff:c000:0201:0000").is_none());
    assert!(parse_gid("0000:0000:0000:0000:0000:ffff:c000:02zz").is_none());
    assert!(parse_gid("0:0:0:0:0:ffff:c000:0201").is_none());
}

#[test]
fn gid_of_maps_ipv4() {
    let ip = IpAddr::V4(Ipv4Addr::new(192, 0, 2, 2));
    assert_eq!(
        gid_of(ip),
        parse_gid("0000:0000:0000:0000:0000:ffff:c000:0202").unwrap()
    );
}

/// A fake `/sys/class/infiniband` with one GID per (device, port, index, gid, type).
fn fake_sysfs(entries: &[(&str, u8, u8, &str, &str)]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for &(dev, port, index, gid, ty) in entries {
        let p = dir.path().join(dev).join("ports").join(port.to_string());
        std::fs::create_dir_all(p.join("gids")).unwrap();
        std::fs::create_dir_all(p.join("gid_attrs/types")).unwrap();
        std::fs::write(p.join("gids").join(index.to_string()), format!("{gid}\n")).unwrap();
        std::fs::write(
            p.join("gid_attrs/types").join(index.to_string()),
            format!("{ty}\n"),
        )
        .unwrap();
    }
    dir
}

#[test]
fn find_gid_takes_the_roce_v2_gid_of_the_address() {
    let mapped = "0000:0000:0000:0000:0000:ffff:c000:0201";
    let sys = fake_sysfs(&[
        (
            "mlx5_a",
            1,
            0,
            "fe80:0000:0000:0000:0000:0000:0000:0001",
            "IB/RoCE v1",
        ),
        ("mlx5_a", 1, 2, mapped, "IB/RoCE v1"),
        (
            "mlx5_b",
            1,
            0,
            "fe80:0000:0000:0000:0000:0000:0000:0002",
            "RoCE v2",
        ),
        ("mlx5_b", 1, 3, mapped, "RoCE v2"),
    ]);
    let ip = IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1));
    let place = find_gid(sys.path(), ip).unwrap();
    assert_eq!(
        place,
        GidPlace {
            dev: "mlx5_b".into(),
            port: 1,
            index: 3,
            gid: gid_of(ip),
        }
    );
    // Another address, or a box without the class directory: no RDMA.
    assert!(find_gid(sys.path(), IpAddr::V4(Ipv4Addr::new(198, 51, 100, 1))).is_none());
    assert!(find_gid(&sys.path().join("missing"), ip).is_none());
}

#[test]
fn find_gid_skips_roce_v1_only() {
    let sys = fake_sysfs(&[(
        "mlx5_a",
        1,
        1,
        "0000:0000:0000:0000:0000:ffff:c000:0201",
        "IB/RoCE v1",
    )]);
    let ip = IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1));
    assert!(find_gid(sys.path(), ip).is_none());
}

#[test]
fn offer_round_trips() {
    let o = Offer {
        rdma: true,
        qpn: 0x1234,
        psn: 0xabcdef,
        mtu: 5,
        gid: gid_of(IpAddr::V4(Ipv4Addr::new(192, 0, 2, 2))),
    };
    assert_eq!(Offer::decode(&o.encode()).unwrap(), o);
    let none = Offer::default();
    assert_eq!(Offer::decode(&none.encode()).unwrap(), none);
    let mut bad = o.encode();
    bad[0] ^= 1;
    assert!(Offer::decode(&bad).is_err());
}

#[test]
fn swap_offers_over_tcp() {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    let theirs = Offer {
        rdma: true,
        qpn: 7,
        psn: 9,
        mtu: 3,
        gid: Gid([1; 16]),
    };
    let t = std::thread::spawn(move || {
        let (mut s, _) = l.accept().unwrap();
        swap_offers(&mut s, &theirs).unwrap()
    });
    let mut s = TcpStream::connect(addr).unwrap();
    let mine = Offer::default();
    assert_eq!(swap_offers(&mut s, &mine).unwrap(), theirs);
    assert_eq!(t.join().unwrap(), mine);
}

#[test]
fn remote_region_round_trips_and_checks_seq() {
    let r = RemoteRegion {
        addr: 0xffff_0000_1234_5000,
        rkey: 0x42,
        len: 6 << 20,
    };
    let b = r.encode(77);
    assert_eq!(RemoteRegion::decode(&b, 77).unwrap(), r);
    let e = RemoteRegion::decode(&b, 78).unwrap_err();
    assert!(e.contains("out of step"), "{e}");
    let mut bad = b;
    bad[1] ^= 1;
    assert!(RemoteRegion::decode(&bad, 77).is_err());
}
