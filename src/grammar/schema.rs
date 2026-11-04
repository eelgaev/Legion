//! JSON Schema -> byte-level AST. See the module docs of `grammar` for the supported subset.

use serde_json::{Map, Value};

use super::{GrammarError, Options};

#[derive(Clone)]
pub(crate) struct Prop {
    pub key: String,
    pub value: Node,
    pub required: bool,
}

#[derive(Clone)]
pub(crate) enum Node {
    /// Matches nothing (pruned before the NFA is built, except as the root check).
    Never,
    Lit(Vec<u8>),
    Alt(Vec<Node>),
    Str {
        min: usize,
        max: Option<usize>,
    },
    Number {
        integer: bool,
    },
    Object(Vec<Prop>),
    /// `{"any key": value, ...}`
    Map(Box<Node>),
    Array {
        item: Box<Node>,
        min: usize,
        max: Option<usize>,
    },
    /// Any JSON value with this many more nesting levels allowed.
    Any(usize),
    /// These one after the other (tool-call markup, not JSON).
    Seq(Vec<Node>),
    /// The node or nothing.
    Opt(Box<Node>),
    /// The node any number of times.
    Star(Box<Node>),
    /// UTF-8 text that contains neither the (ASCII) terminator `end` nor any of the (ASCII)
    /// `forbid` strings, then `end`: a raw tool-call parameter value up to its closing tag.
    /// A forbidden string that ends where `end` does is part of it (allowed).
    RawUntil {
        end: Vec<u8>,
        forbid: Vec<Vec<u8>>,
    },
}

pub(crate) const ANNOTATIONS_OK: &[&str] = &[
    "$schema",
    "$id",
    "$comment",
    "$defs",
    "definitions",
    "title",
    "description",
    "default",
    "examples",
    "format",
    "deprecated",
    "readOnly",
    "writeOnly",
    "contentMediaType",
    "contentEncoding",
];

const UNSUPPORTED: &[&str] = &[
    "pattern",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "allOf",
    "not",
    "if",
    "then",
    "else",
    "patternProperties",
    "propertyNames",
    "prefixItems",
    "additionalItems",
    "contains",
    "minContains",
    "maxContains",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "dependentRequired",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "unevaluatedItems",
    "$dynamicRef",
    "$anchor",
    "$recursiveRef",
];

pub(crate) fn lit(v: &Value) -> Node {
    Node::Lit(serde_json::to_vec(v).expect("a JSON value serializes"))
}

pub(crate) fn any_object(opts: &Options) -> Node {
    Node::Map(Box::new(Node::Any(opts.any_depth.saturating_sub(1))))
}

pub(crate) fn lower(root: &Value, opts: &Options) -> Result<Node, GrammarError> {
    let mut c = Lowering {
        root,
        opts,
        refs: Vec::new(),
    };
    let node = c.node(root)?;
    if matches!(node, Node::Never) {
        return Err(GrammarError::Unsatisfiable);
    }
    Ok(node)
}

/// `schema`, a part of the document `root` (which its `$ref`s resolve against), lowered on its
/// own. Unlike [`lower`], an unsatisfiable schema comes back as [`Node::Never`].
pub(crate) fn lower_at(root: &Value, schema: &Value, opts: &Options) -> Result<Node, GrammarError> {
    Lowering {
        root,
        opts,
        refs: Vec::new(),
    }
    .node(schema)
}

struct Lowering<'a> {
    root: &'a Value,
    opts: &'a Options,
    refs: Vec<String>,
}

fn bad<T>(msg: impl Into<String>) -> Result<T, GrammarError> {
    Err(GrammarError::Invalid(msg.into()))
}

fn unsupported<T>(msg: impl Into<String>) -> Result<T, GrammarError> {
    Err(GrammarError::Unsupported(msg.into()))
}

pub(crate) fn alt(mut v: Vec<Node>) -> Node {
    v.retain(|n| !matches!(n, Node::Never));
    match v.len() {
        0 => Node::Never,
        1 => v.pop().unwrap(),
        _ => Node::Alt(v),
    }
}

fn usize_kw(m: &Map<String, Value>, key: &str) -> Result<Option<usize>, GrammarError> {
    match m.get(key) {
        None => Ok(None),
        Some(v) => match v.as_u64() {
            Some(n) => Ok(Some(n as usize)),
            None => bad(format!("`{key}` must be a non-negative integer")),
        },
    }
}

impl Lowering<'_> {
    fn any(&self) -> Node {
        Node::Any(self.opts.any_depth)
    }

    fn node(&mut self, schema: &Value) -> Result<Node, GrammarError> {
        let m = match schema {
            Value::Bool(true) => return Ok(self.any()),
            Value::Bool(false) => return Ok(Node::Never),
            Value::Object(m) => m,
            _ => return bad("a schema must be an object or a boolean"),
        };
        for k in m.keys() {
            if UNSUPPORTED.contains(&k.as_str()) {
                return unsupported(format!("`{k}`"));
            }
        }
        if let Some(r) = m.get("$ref") {
            return self.reference(r);
        }
        if let Some(v) = m.get("const") {
            return Ok(lit(v));
        }
        if let Some(v) = m.get("enum") {
            let Some(items) = v.as_array() else {
                return bad("`enum` must be an array");
            };
            return Ok(alt(items.iter().map(lit).collect()));
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(v) = m.get(key) {
                let Some(items) = v.as_array() else {
                    return bad(format!("`{key}` must be an array"));
                };
                let mut alts = Vec::new();
                for s in items {
                    alts.push(self.node(s)?);
                }
                return Ok(alt(alts));
            }
        }
        let types: Vec<&str> = match m.get("type") {
            Some(Value::String(s)) => vec![s.as_str()],
            Some(Value::Array(a)) => {
                let mut t = Vec::new();
                for x in a {
                    match x.as_str() {
                        Some(s) => t.push(s),
                        None => return bad("`type` entries must be strings"),
                    }
                }
                t
            }
            Some(_) => return bad("`type` must be a string or an array"),
            None => {
                if m.contains_key("properties")
                    || m.contains_key("required")
                    || m.contains_key("additionalProperties")
                {
                    vec!["object"]
                } else if m.contains_key("items")
                    || m.contains_key("minItems")
                    || m.contains_key("maxItems")
                {
                    vec!["array"]
                } else if m.contains_key("minLength") || m.contains_key("maxLength") {
                    vec!["string"]
                } else {
                    for k in m.keys() {
                        if !ANNOTATIONS_OK.contains(&k.as_str()) {
                            return unsupported(format!("keyword `{k}`"));
                        }
                    }
                    return Ok(self.any());
                }
            }
        };
        let mut alts = Vec::new();
        for t in types {
            alts.push(self.typed(t, m)?);
        }
        Ok(alt(alts))
    }

    fn reference(&mut self, r: &Value) -> Result<Node, GrammarError> {
        let Some(path) = r.as_str() else {
            return bad("`$ref` must be a string");
        };
        let target = if path == "#" {
            Some(self.root)
        } else if let Some(p) = path.strip_prefix('#') {
            self.root.pointer(p)
        } else {
            return unsupported(format!("non-local `$ref` {path:?}"));
        };
        let Some(target) = target else {
            return bad(format!("`$ref` {path:?} does not resolve"));
        };
        if self.refs.iter().filter(|p| *p == path).count() >= self.opts.recursion_depth {
            return Ok(Node::Never);
        }
        self.refs.push(path.to_string());
        let n = self.node(target);
        self.refs.pop();
        n
    }

    fn typed(&mut self, t: &str, m: &Map<String, Value>) -> Result<Node, GrammarError> {
        match t {
            "string" => {
                let min = usize_kw(m, "minLength")?.unwrap_or(0);
                let max = usize_kw(m, "maxLength")?;
                if max.is_some_and(|x| x < min) {
                    return Ok(Node::Never);
                }
                Ok(Node::Str { min, max })
            }
            "integer" => Ok(Node::Number { integer: true }),
            "number" => Ok(Node::Number { integer: false }),
            "boolean" => Ok(alt(vec![
                Node::Lit(b"true".to_vec()),
                Node::Lit(b"false".to_vec()),
            ])),
            "null" => Ok(Node::Lit(b"null".to_vec())),
            "array" => self.array(m),
            "object" => self.object(m),
            other => bad(format!("unknown type {other:?}")),
        }
    }

    fn array(&mut self, m: &Map<String, Value>) -> Result<Node, GrammarError> {
        let min = usize_kw(m, "minItems")?.unwrap_or(0);
        let max = usize_kw(m, "maxItems")?;
        if max.is_some_and(|x| x < min) {
            return Ok(Node::Never);
        }
        let item = match m.get("items") {
            None => self.any(),
            Some(s) => self.node(s)?,
        };
        if matches!(item, Node::Never) {
            return Ok(if min == 0 {
                Node::Lit(b"[]".to_vec())
            } else {
                Node::Never
            });
        }
        Ok(Node::Array {
            item: Box::new(item),
            min,
            max,
        })
    }

    fn object(&mut self, m: &Map<String, Value>) -> Result<Node, GrammarError> {
        let empty = Map::new();
        let props = match m.get("properties") {
            None => &empty,
            Some(Value::Object(p)) => p,
            Some(_) => return bad("`properties` must be an object"),
        };
        let mut required: Vec<&str> = Vec::new();
        match m.get("required") {
            None => {}
            Some(Value::Array(a)) => {
                for r in a {
                    match r.as_str() {
                        Some(s) => required.push(s),
                        None => return bad("`required` entries must be strings"),
                    }
                }
            }
            Some(_) => return bad("`required` must be an array"),
        }
        for r in &required {
            if !props.contains_key(*r) {
                return unsupported(format!("required property {r:?} is not in `properties`"));
            }
        }
        let additional = m.get("additionalProperties");
        if props.is_empty() {
            return Ok(match additional {
                None | Some(Value::Bool(true)) => Node::Map(Box::new(self.any())),
                Some(Value::Bool(false)) => Node::Lit(b"{}".to_vec()),
                Some(s) => {
                    let v = self.node(s)?;
                    if matches!(v, Node::Never) {
                        Node::Lit(b"{}".to_vec())
                    } else {
                        Node::Map(Box::new(v))
                    }
                }
            });
        }
        match additional {
            None | Some(Value::Bool(false)) => {}
            Some(_) => {
                return unsupported(
                    "`additionalProperties` other than false together with `properties`",
                );
            }
        }
        let mut out = Vec::new();
        for (key, schema) in props {
            let value = self.node(schema)?;
            let req = required.contains(&key.as_str());
            if matches!(value, Node::Never) {
                if req {
                    return Ok(Node::Never);
                }
                continue;
            }
            out.push(Prop {
                key: key.clone(),
                value,
                required: req,
            });
        }
        Ok(Node::Object(out))
    }
}
