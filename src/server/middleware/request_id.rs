//! Per-request id and access line. Every response carries an `x-request-id` header: a valid
//! incoming id (1-128 printable ASCII chars) is echoed back unchanged, anything else is
//! replaced by a generated `req_...` id. One short `info` line is logged when the response
//! head is sent: `GET /v1/models 200 0.015 ms`.

use std::time::Instant;

use axum::extract::Request;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use axum::middleware::Next;
use axum::response::Response;

const REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");
const MAX_ID_LEN: usize = 128;

pub async fn request_id(req: Request, next: Next) -> Response {
    let id = incoming_id(req.headers())
        .map(str::to_owned)
        .unwrap_or_else(|| crate::api::new_id("req_"));
    let method = req.method().clone();
    let path = req.uri().path().to_owned();
    let start = Instant::now();
    let mut response = next.run(req).await;
    // Rounded to microseconds: a bare f64 millis prints full float noise.
    let ms = (start.elapsed().as_secs_f64() * 1e6).round() / 1e3;
    tracing::info!("{method} {path} {} {ms} ms", response.status().as_u16());
    let value = HeaderValue::from_str(&id).expect("request ids are printable ASCII");
    response.headers_mut().insert(REQUEST_ID, value);
    response
}

/// The client's id if it is 1-128 printable ASCII chars; `None` means "generate one". A
/// header carrying non-printable or non-ASCII bytes fails `to_str` or the byte check and is
/// replaced just like a too-long one.
fn incoming_id(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(REQUEST_ID)
        .and_then(|v| v.to_str().ok())
        .filter(|id| {
            !id.is_empty()
                && id.len() <= MAX_ID_LEN
                && id.bytes().all(|b| (0x20..=0x7e).contains(&b))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(value: &[u8]) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(
            REQUEST_ID,
            HeaderValue::from_bytes(value).expect("valid header bytes"),
        );
        h
    }

    #[test]
    fn valid_ids_are_accepted() {
        assert_eq!(
            incoming_id(&header(b"trace-abc-123")),
            Some("trace-abc-123")
        );
        // Spaces are printable ASCII and allowed mid-value.
        assert_eq!(incoming_id(&header(b"a b c")), Some("a b c"));
        // The length bounds: 1 and 128 chars are valid.
        assert_eq!(incoming_id(&header(b"x")), Some("x"));
        let long = vec![b'y'; MAX_ID_LEN];
        assert_eq!(
            incoming_id(&header(&long)),
            Some(std::str::from_utf8(&long).unwrap())
        );
    }

    #[test]
    fn invalid_ids_are_rejected() {
        // Empty and too long (129 chars).
        assert_eq!(incoming_id(&header(b"")), None);
        let long = vec![b'y'; MAX_ID_LEN + 1];
        assert_eq!(incoming_id(&header(&long)), None);
        // Non-ASCII (obs-text) bytes.
        assert_eq!(incoming_id(&header(b"h\xc3\xa9llo")), None);
        assert_eq!(incoming_id(&header(b"caf\xe9")), None);
        // A tab is not printable.
        assert_eq!(incoming_id(&header(b"a\tb")), None);
    }

    #[test]
    fn missing_header_generates() {
        assert_eq!(incoming_id(&HeaderMap::new()), None);
    }
}
