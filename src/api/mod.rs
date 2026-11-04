//! Wire formats. Each front end has its own types; nothing here knows about engines.

pub mod anthropic;
pub mod openai;

/// A field that may be a single string or a list of strings (`stop`, `prompt`).
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(untagged)]
pub enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

impl OneOrMany {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            OneOrMany::One(s) => vec![s],
            OneOrMany::Many(v) => v,
        }
    }
}

/// Unix time in seconds.
pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A random id with the given prefix, e.g. `chatcmpl-1f0c...`.
pub fn new_id(prefix: &str) -> String {
    format!("{prefix}{}", uuid::Uuid::new_v4().simple())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_or_many_into_vec() {
        let one = OneOrMany::One("x".to_string());
        assert_eq!(one.clone().into_vec(), vec!["x".to_string()]);

        let many = OneOrMany::Many(vec!["x".to_string(), "y".to_string()]);
        assert_eq!(
            many.clone().into_vec(),
            vec!["x".to_string(), "y".to_string()]
        );

        let one_json = r#""x""#;
        let one_deserialized: OneOrMany = serde_json::from_str(one_json).unwrap();
        assert_eq!(one_deserialized.into_vec(), vec!["x".to_string()]);

        let many_json = "[\"x\", \"y\"]";
        let many_deserialized: OneOrMany = serde_json::from_str(many_json).unwrap();
        assert_eq!(
            many_deserialized.into_vec(),
            vec!["x".to_string(), "y".to_string()]
        );
    }

    #[test]
    fn test_new_id() {
        let prefix = "chatcmpl-";
        let id1 = new_id(prefix);
        let id2 = new_id(prefix);

        assert!(id1.starts_with(prefix));
        assert_ne!(id1, id2);

        let suffix = &id1[prefix.len()..];
        assert_eq!(suffix.len(), 32);
        assert!(suffix.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
