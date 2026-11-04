//! What a request gets for the sampling parameters it leaves out: each model's own defaults.
//!
//! Precedence, per parameter: the request's field (always, `temperature: 0` included) >
//! the operator's `--default-*` flag > the model file's declared default (GGUF
//! `general.sampling.*`, as llama.cpp's converter writes them from `generation_config.json`) >
//! the architecture's published recommendation (only when the file declares none at all) >
//! the OpenAI-style default ([`SamplingParams::default`]). `--sampling-defaults openai` skips the
//! model's two layers.
//!
//! The request layer is applied by the front ends (`src/server/convert.rs`); this module
//! resolves the rest into one [`SamplingDefaults`] per model, with where each value came from
//! for the startup log.

use std::fmt;

use super::SamplingParams;

/// Where a default came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The operator's `--default-*` flag.
    Operator,
    /// The model file (GGUF `general.sampling.*`, or a checkpoint's `generation_config.json`).
    Model,
    /// Legion's table of the architecture's published defaults.
    Architecture,
    /// The OpenAI-style default.
    OpenAi,
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Source::Operator => "operator",
            Source::Model => "model file",
            Source::Architecture => "architecture",
            Source::OpenAi => "openai",
        })
    }
}

/// `--sampling-defaults`: whose defaults fill the parameters a request omits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum DefaultsMode {
    /// The model's own (its file, else its architecture's), then OpenAI's.
    #[default]
    Model,
    /// OpenAI's (temperature 1, top_p 1, no top-k / min-p, no penalties).
    Openai,
}

/// One layer of defaults: a value for some parameters.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SamplingLayer {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

impl SamplingLayer {
    /// Whether the layer sets nothing.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// The layer read from a model file's metadata, `get` giving a key's number: llama.cpp's
    /// `general.sampling.temp`, `.top_k`, `.top_p`, `.min_p` and `.penalty_repeat`. A value out of
    /// the API's range (a negative top_k, say) is left out.
    pub fn from_metadata(get: impl Fn(&str) -> Option<f64>) -> Self {
        let f = |k: &str| get(&format!("general.sampling.{k}")).map(|v| v as f32);
        Self {
            temperature: f("temp").filter(|t| (0.0..=2.0).contains(t)),
            top_p: f("top_p").filter(|p| *p > 0.0 && *p <= 1.0),
            top_k: get("general.sampling.top_k")
                .filter(|k| *k >= 0.0 && k.fract() == 0.0 && *k <= u32::MAX as f64)
                .map(|k| k as u32),
            min_p: f("min_p").filter(|p| (0.0..=1.0).contains(p)),
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: f("penalty_repeat").filter(|p| *p > 0.0),
        }
    }

    /// The layer of a Hugging Face checkpoint's `generation_config.json`: `temperature`,
    /// `top_k`, `top_p`, `min_p`, `repetition_penalty`. `"do_sample": false` (transformers'
    /// greedy decoding) is temperature 0.
    pub fn from_generation_config(j: &serde_json::Value) -> Self {
        let layer = Self::from_metadata(|k| {
            let name = match k.strip_prefix("general.sampling.")? {
                "temp" => "temperature",
                "penalty_repeat" => "repetition_penalty",
                other => other,
            };
            j.get(name)?.as_f64()
        });
        let greedy = j.get("do_sample").and_then(serde_json::Value::as_bool) == Some(false);
        Self {
            temperature: if greedy { Some(0.0) } else { layer.temperature },
            ..layer
        }
    }

    /// The layer as GGUF `general.sampling.*` metadata (what [`Self::from_metadata`] reads).
    pub fn metadata(&self) -> Vec<(&'static str, crate::gguf::Value)> {
        use crate::gguf::Value;
        let mut out = Vec::new();
        let mut f = |k, v: Option<f32>| {
            if let Some(v) = v {
                out.push((k, Value::F32(v)));
            }
        };
        f("general.sampling.temp", self.temperature);
        f("general.sampling.top_p", self.top_p);
        f("general.sampling.min_p", self.min_p);
        f("general.sampling.penalty_repeat", self.repetition_penalty);
        if let Some(k) = self.top_k.and_then(|k| i32::try_from(k).ok()) {
            out.push(("general.sampling.top_k", Value::I32(k)));
        }
        out
    }

    /// The layer of a GGUF's `general.sampling.*` keys.
    pub fn from_gguf(g: &crate::gguf::GgufFile) -> Self {
        Self::from_metadata(|k| g.get(k)?.as_f64())
    }

    /// The published defaults of a model architecture (`general.architecture`), for files that
    /// declare none; empty for architectures without a known recommendation.
    pub fn for_architecture(arch: &str) -> Self {
        let tpk = |temperature: f32, top_p: f32, top_k: u32| Self {
            temperature: Some(temperature),
            top_p: Some(top_p),
            top_k: Some(top_k),
            min_p: Some(0.0),
            ..Default::default()
        };
        match arch {
            // Gemma 4's model card.
            "gemma4" => tpk(1.0, 0.95, 64),
            // Qwen3.8-Flash-Next: what its model card gives and its GGUFs declare.
            "qwen4exp" => tpk(1.0, 0.95, 20),
            // Qwen3's model card (thinking mode).
            "qwen3" | "qwen3moe" => tpk(0.6, 0.95, 20),
            _ => Self::default(),
        }
    }
}

/// A default and where it came from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Param<T> {
    pub value: T,
    pub source: Source,
}

/// The defaults of one model, resolved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingDefaults {
    pub temperature: Param<f32>,
    pub top_p: Param<f32>,
    pub top_k: Param<u32>,
    pub min_p: Param<f32>,
    pub presence_penalty: Param<f32>,
    pub frequency_penalty: Param<f32>,
    pub repetition_penalty: Param<f32>,
}

impl Default for SamplingDefaults {
    /// OpenAI's.
    fn default() -> Self {
        Self::resolve(
            DefaultsMode::Openai,
            &SamplingLayer::default(),
            &ModelSampling::default(),
        )
    }
}

/// What a model brings: its file's layer and its architecture's.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelSampling {
    pub file: SamplingLayer,
    pub architecture: SamplingLayer,
}

impl ModelSampling {
    /// A GGUF's keys and its architecture's table.
    pub fn from_gguf(g: &crate::gguf::GgufFile) -> Self {
        Self {
            file: SamplingLayer::from_gguf(g),
            architecture: g
                .architecture()
                .map(|a| SamplingLayer::for_architecture(&a))
                .unwrap_or_default(),
        }
    }
}

impl SamplingDefaults {
    /// The defaults for a model: `operator` over the model's layers (unless `mode` is
    /// `Openai`) over OpenAI's. The architecture's layer counts only when the file's is empty.
    pub fn resolve(mode: DefaultsMode, operator: &SamplingLayer, model: &ModelSampling) -> Self {
        let none = SamplingLayer::default();
        let (file, arch) = match mode {
            DefaultsMode::Openai => (&none, &none),
            DefaultsMode::Model if model.file.is_empty() => (&none, &model.architecture),
            DefaultsMode::Model => (&model.file, &none),
        };
        let o = SamplingParams::default();
        fn pick<T: Copy>(layers: [(Option<T>, Source); 3], fallback: T) -> Param<T> {
            layers
                .into_iter()
                .find_map(|(v, source)| v.map(|value| Param { value, source }))
                .unwrap_or(Param {
                    value: fallback,
                    source: Source::OpenAi,
                })
        }
        macro_rules! field {
            ($f:ident) => {
                pick(
                    [
                        (operator.$f, Source::Operator),
                        (file.$f, Source::Model),
                        (arch.$f, Source::Architecture),
                    ],
                    o.$f,
                )
            };
        }
        Self {
            temperature: field!(temperature),
            top_p: field!(top_p),
            top_k: field!(top_k),
            min_p: field!(min_p),
            presence_penalty: field!(presence_penalty),
            frequency_penalty: field!(frequency_penalty),
            repetition_penalty: field!(repetition_penalty),
        }
    }

    /// One line for the startup log: `temperature=1 (model file), top_k=20 (model file), ...`.
    pub fn describe(&self) -> String {
        let p = |name: &str, v: String, s: Source| format!("{name}={v} ({s})");
        [
            p(
                "temperature",
                self.temperature.value.to_string(),
                self.temperature.source,
            ),
            p("top_p", self.top_p.value.to_string(), self.top_p.source),
            p("top_k", self.top_k.value.to_string(), self.top_k.source),
            p("min_p", self.min_p.value.to_string(), self.min_p.source),
            p(
                "presence_penalty",
                self.presence_penalty.value.to_string(),
                self.presence_penalty.source,
            ),
            p(
                "frequency_penalty",
                self.frequency_penalty.value.to_string(),
                self.frequency_penalty.source,
            ),
            p(
                "repetition_penalty",
                self.repetition_penalty.value.to_string(),
                self.repetition_penalty.source,
            ),
        ]
        .join(", ")
    }
}

#[cfg(test)]
mod tests;
