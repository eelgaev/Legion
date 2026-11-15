use super::*;

/// An AC922: V100s (7.0) over NVLink 2.0 with ATS, on POWER9.
const AC922: (&str, LinkAttrs) = (
    "powerpc64",
    LinkAttrs {
        integrated: false,
        pageable_access: true,
        pageable_uses_host_page_tables: true,
        cc: (7, 0),
    },
);

/// A DGX Spark: the GB10 (12.1), integrated, on Arm.
const GB10: (&str, LinkAttrs) = (
    "aarch64",
    LinkAttrs {
        integrated: true,
        pageable_access: true,
        pageable_uses_host_page_tables: true,
        cc: (12, 1),
    },
);

/// A GH200: an H100/H200 (9.0) over NVLink-C2C with ATS, on Grace.
const GH200: (&str, LinkAttrs) = (
    "aarch64",
    LinkAttrs {
        integrated: false,
        pageable_access: true,
        pageable_uses_host_page_tables: true,
        cc: (9, 0),
    },
);

/// An RTX card behind PCIe on x86 (HMM may give it pageable access, never the host's page
/// tables).
const RTX: (&str, LinkAttrs) = (
    "x86_64",
    LinkAttrs {
        integrated: false,
        pageable_access: true,
        pageable_uses_host_page_tables: false,
        cc: (8, 9),
    },
);

/// An H100 behind PCIe on x86.
const H100_PCIE: (&str, LinkAttrs) = (
    "x86_64",
    LinkAttrs {
        integrated: false,
        pageable_access: false,
        pageable_uses_host_page_tables: false,
        cc: (9, 0),
    },
);

/// A V100 behind PCIe on x86: 7.0 keeps the V100 default it has always had.
const V100_PCIE: (&str, LinkAttrs) = (
    "x86_64",
    LinkAttrs {
        integrated: false,
        pageable_access: false,
        pageable_uses_host_page_tables: false,
        cc: (7, 0),
    },
);

#[test]
fn c2c_only_on_a_grace_hopper() {
    assert!(c2c(GH200.0, GH200.1));
    for (arch, a) in [AC922, GB10, RTX, H100_PCIE, V100_PCIE] {
        assert!(!c2c(arch, a), "{arch} {a:?}");
    }
    // A Hopper on Arm without ATS (pageable access off in the driver): not C2C.
    let no_ats = LinkAttrs {
        pageable_uses_host_page_tables: false,
        ..GH200.1
    };
    assert!(!c2c("aarch64", no_ats));
}

#[test]
fn spill_default_as_volta_was_plus_gh200() {
    // AC922 and a V100 anywhere: on, as `cc == (7, 0)` was.
    assert!(spill_default(AC922.0, AC922.1));
    assert!(spill_default(V100_PCIE.0, V100_PCIE.1));
    // GB10, RTX, a PCIe H100: off, as before.
    assert!(!spill_default(GB10.0, GB10.1));
    assert!(!spill_default(RTX.0, RTX.1));
    assert!(!spill_default(H100_PCIE.0, H100_PCIE.1));
    // GH200: on.
    assert!(spill_default(GH200.0, GH200.1));
    // A C2C GPU other than a Hopper (a Grace Blackwell, 10.0): off until measured.
    let gb200 = LinkAttrs {
        cc: (10, 0),
        ..GH200.1
    };
    assert!(!spill_default("aarch64", gb200));
    // Unknown attributes: off, as `cc == None` was.
    assert!(!spill_default_here(None));
}

#[test]
fn platform_by_attributes() {
    use MemoryKind::*;
    // What the report says stays for an integrated GPU and an AC922, attributes or not.
    assert_eq!(
        platform(IntegratedShared, GB10.0, Some(&[GB10.1]), false),
        Platform::Integrated
    );
    assert_eq!(
        platform(CoherentNvlink, AC922.0, Some(&[AC922.1; 4]), false),
        Platform::CoherentNvlink
    );
    assert_eq!(
        platform(CoherentNvlink, AC922.0, None, false),
        Platform::CoherentNvlink
    );
    // A GH200 by its attributes, whatever NVML calls it.
    assert_eq!(
        platform(Discrete, GH200.0, Some(&[GH200.1]), false),
        Platform::NvlinkC2c
    );
    assert_eq!(
        platform(Discrete, GH200.0, Some(&[GH200.1; 4]), true),
        Platform::NvlinkC2c
    );
    // Every GPU must be: one without ATS makes it PCIe.
    assert_eq!(
        platform(Discrete, GH200.0, Some(&[GH200.1, RTX.1]), true),
        Platform::DiscretePcie
    );
    // PCIe cards: as before.
    assert_eq!(
        platform(Discrete, RTX.0, Some(&[RTX.1]), false),
        Platform::DiscretePcie
    );
    assert_eq!(
        platform(Discrete, H100_PCIE.0, Some(&[H100_PCIE.1]), false),
        Platform::DiscretePcie
    );
    // No attributes (no driver): the NVML name, as before; an empty list counts as none.
    assert_eq!(
        platform(Discrete, "aarch64", None, true),
        Platform::NvlinkC2c
    );
    assert_eq!(
        platform(Discrete, "aarch64", Some(&[]), true),
        Platform::NvlinkC2c
    );
    assert_eq!(
        platform(Discrete, "aarch64", None, false),
        Platform::DiscretePcie
    );
    assert_eq!(
        platform(Discrete, "x86_64", None, false),
        Platform::DiscretePcie
    );
}
