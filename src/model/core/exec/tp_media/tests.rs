use super::*;

fn span(start: usize, nx: usize, ny: usize, n_embd: usize, v: f32) -> ImageSpan {
    ImageSpan {
        start,
        nx,
        ny,
        rows: (0..nx * ny * n_embd).map(|i| v + i as f32).collect(),
    }
}

/// Rank 0's request, as rank 1 takes it.
fn take(table: &mut MediaTable, req: Request) -> u32 {
    let Request::TpMedia {
        id,
        n_embd,
        spans,
        rows,
    } = req
    else {
        panic!("not images: {req:?}")
    };
    table.add(id, n_embd, &spans, &rows).unwrap();
    id
}

#[test]
fn images_go_once_and_arrive_as_sent() {
    let a = Media::new(vec![span(3, 2, 3, 4, 0.5), span(20, 1, 2, 4, -9.0)], 4);
    let b = Media::new(vec![span(3, 2, 3, 4, 0.5)], 4);
    let mut sent = SentMedia::default();
    let mut table = MediaTable::default();
    let (ia, req) = sent.id(&a);
    let ia2 = take(&mut table, req.expect("sent the first time"));
    assert_eq!(ia, ia2);
    assert_ne!(ia, NONE);
    // The same images again, and a clone of them (the same rows): not sent again.
    assert_eq!(sent.id(&a), (ia, None));
    assert_eq!(sent.id(&a.clone()), (ia, None));
    // Other images (the same values, other rows): sent under another id.
    let (ib, req) = sent.id(&b);
    assert_ne!(ib, ia);
    take(&mut table, req.expect("other images are sent"));
    let copy = Media::new(vec![span(3, 2, 3, 4, 0.5)], 4);
    let (ic, req) = sent.id(&copy);
    assert!(req.is_some() && ic != ib, "a copy's rows are other rows");
    // Rank 1 holds the same images: the same rows and positions.
    assert_eq!(table.get(ia).unwrap(), Some(&a));
    assert_eq!(table.get(ib).unwrap(), Some(&b));
    for i in 0..30 {
        assert_eq!(table.get(ia).unwrap().unwrap().rope(i), a.rope(i));
        assert_eq!(table.get(ia).unwrap().unwrap().row(i), a.row(i));
    }
    assert_eq!(table.get(NONE).unwrap(), None);
    assert!(table.get(ic.wrapping_add(1)).is_err());
}

#[test]
fn images_no_sequence_holds_are_forgotten() {
    let mut sent = SentMedia::default();
    let mut table = MediaTable::default();
    assert!(sent.gone().is_none());
    let a = Media::new(vec![span(0, 2, 2, 3, 1.0)], 3);
    let b = Media::new(vec![span(5, 1, 1, 3, 2.0), span(9, 3, 1, 3, 3.0)], 3);
    let ia = take(&mut table, sent.id(&a).1.unwrap());
    let ib = take(&mut table, sent.id(&b).1.unwrap());
    // A clone keeps the rows: nothing gone with the first copy.
    let b2 = b.clone();
    drop(b);
    assert!(sent.gone().is_none());
    assert_eq!(sent.id(&b2), (ib, None));
    drop(a);
    let Some(Request::TpMediaDrop { ids }) = sent.gone() else {
        panic!("a's rows are gone")
    };
    assert_eq!(ids, vec![ia]);
    table.forget(&ids);
    assert!(table.get(ia).is_err());
    assert!(table.get(ib).unwrap().is_some());
    assert!(sent.gone().is_none(), "said once");
    drop(b2);
    let Some(Request::TpMediaDrop { ids }) = sent.gone() else {
        panic!("b's rows are gone")
    };
    assert_eq!(ids, vec![ib]);
    table.forget(&ids);
    assert!(table.media.is_empty() && sent.sent.is_empty());
    // New images after: a new id, sent again.
    let c = Media::new(vec![span(0, 2, 2, 3, 1.0)], 3);
    let (ic, req) = sent.id(&c);
    assert!(req.is_some());
    assert!(ic != ia && ic != ib);
}

#[test]
fn ids_skip_none_and_ids_in_use() {
    let mut sent = SentMedia {
        last: u32::MAX - 1,
        ..Default::default()
    };
    let a = Media::new(vec![span(0, 1, 1, 2, 0.0)], 2);
    let b = Media::new(vec![span(0, 1, 1, 2, 0.0)], 2);
    let c = Media::new(vec![span(0, 1, 1, 2, 0.0)], 2);
    assert_eq!(sent.id(&a).0, u32::MAX);
    assert_eq!(sent.id(&b).0, 1);
    sent.last = u32::MAX - 1;
    assert_eq!(sent.id(&c).0, 2, "u32::MAX and 1 are in use");
    // No images at all are a set too (a sequence's media, empty).
    let empty = Media::default();
    let (ie, req) = sent.id(&empty);
    assert!(req.is_some() && ie != NONE);
    assert_eq!(sent.id(&empty.clone()), (ie, None));
}

/// The image-row cache hands one image's rows to every request that sends it, and keeps them
/// past the sequences: each request's images are sent under their own id, and forgotten once
/// that request's sequences are gone, while the rows live on.
#[test]
fn cached_rows_are_forgotten_with_their_sequences() {
    let mut sent = SentMedia::default();
    let mut table = MediaTable::default();
    let cached = span(0, 2, 2, 3, 1.0);
    let a = Media::new(vec![cached.clone()], 3);
    let ia = take(&mut table, sent.id(&a).1.unwrap());
    // The same image again (the same rows), at the same place and at another.
    let b = Media::new(vec![cached.clone()], 3);
    let (ib, req) = sent.id(&b);
    assert!(req.is_some() && ib != ia, "another request's images");
    take(&mut table, req.unwrap());
    let mut moved = cached.clone();
    moved.start = 7;
    let c = Media::new(vec![moved], 3);
    let ic = take(&mut table, sent.id(&c).1.unwrap());
    assert!(ic != ia && ic != ib);
    assert_eq!(table.get(ic).unwrap().unwrap().rope(7), c.rope(7));
    drop((a, b));
    let Some(Request::TpMediaDrop { mut ids }) = sent.gone() else {
        panic!("the first two requests are gone, the cache still holds their rows")
    };
    ids.sort_unstable();
    assert_eq!(ids, vec![ia.min(ib), ia.max(ib)]);
    table.forget(&ids);
    assert!(table.get(ia).is_err() && table.get(ib).is_err());
    assert_eq!(table.get(ic).unwrap(), Some(&c));
    drop(c);
    assert!(matches!(sent.gone(), Some(Request::TpMediaDrop { ids }) if ids == vec![ic]));
    assert!(sent.sent.is_empty());
    assert_eq!(
        std::sync::Arc::strong_count(&cached.rows),
        1,
        "the cache's own"
    );
}

#[test]
fn bad_images_fail_where_they_arrive() {
    let mut t = MediaTable::default();
    let rows = vec![0.0f32; 12];
    assert!(t.add(NONE, 2, &[(0, 2, 3)], &rows).is_err());
    assert!(t.add(1, 2, &[(0, 2, 2)], &rows).is_err(), "rows left over");
    assert!(t.add(1, 2, &[(0, 2, 4)], &rows).is_err(), "rows missing");
    assert!(t.add(1, 2, &[(0, 0, 6)], &[]).is_err(), "an empty image");
    assert!(
        t.add(1, 2, &[(0, 1, 3), (2, 1, 3)], &rows).is_err(),
        "overlap"
    );
    assert!(t.add(1, 0, &[(0, 2, 3)], &[]).is_err(), "rows 0 wide");
    assert!(
        t.add(1, 2, &[(u32::MAX, u32::MAX, u32::MAX)], &rows)
            .is_err(),
        "past the end"
    );
    assert!(t.media.is_empty());
    t.add(1, 2, &[(0, 1, 3), (3, 1, 3)], &rows).unwrap();
    assert_eq!(t.get(1).unwrap().unwrap().spans().len(), 2);
}
