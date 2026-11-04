//! Schema validation for engines without a tokenizer (such as the mock).

use super::{GrammarError, Options, matcher::Nfa, schema};

/// Apply the same schema and automaton limits as the default compiler, without a vocabulary.
pub fn check_json_schema(value: &serde_json::Value) -> Result<(), GrammarError> {
    let options = Options::default();
    let node = schema::lower(value, &options)?;
    Nfa::build(&node, &options).map(|_| ())
}
