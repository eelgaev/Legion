use super::*;

/// A pool with both stores defers the commits of the slots whose last verify step stored in
/// the packed rows and runs the others at once, each list in the commit's order (#1176).
#[test]
fn commits_split_by_where_the_verify_step_stored() {
    let in_rows = [true, false, false, true, true];
    let (deferred, at_once) = split_by_store(&in_rows, &[(4, 2), (1, 1), (0, 3), (2, 5)]);
    assert_eq!(deferred, [(4, 2), (0, 3)]);
    assert_eq!(at_once, [(1, 1), (2, 5)]);
    let (deferred, at_once) = split_by_store(&in_rows, &[(1, 4)]);
    assert!(deferred.is_empty());
    assert_eq!(at_once, [(1, 4)]);
    let (deferred, at_once) = split_by_store(&in_rows, &[(3, 1), (4, 1)]);
    assert_eq!(deferred, [(3, 1), (4, 1)]);
    assert!(at_once.is_empty());
}

/// KDA pools (GLM) keep more commit graphs than the others, which keep eight (#725).
#[test]
fn kda_pools_keep_more_commit_graphs() {
    assert_eq!(commit_kept(false), 8);
    assert!(commit_kept(true) >= 30, "C=4's sets of runs fit");
}
