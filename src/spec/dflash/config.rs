//! A DFlash drafter's hyperparameters, from a GGUF (llama.cpp's `dflash` architecture) or from
//! a Hugging Face checkpoint's `config.json` (`DFlashDraftModel`).

use serde_json::Value as Json;

use crate::gguf::GgufFile;

pub use super::v2::Dflash2;

/// The block size DFlash checkpoints are trained with unless they say otherwise.
const DEFAULT_BLOCK_SIZE: usize = 16;

/// Shape and behaviour of the drafter network (Qwen3-style decoder layers with non-causal
/// attention over the target's fused hidden states plus the noise block).
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub n_layer: usize,
    /// Hidden size of the drafter. It reads the target's token embeddings and writes through the
    /// target's output head unless the checkpoint ships its own, so it matches the target's.
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub rms_eps: f32,
    pub rope_base: f32,
    /// Rotary dimensions per head (`head_dim` unless the rotation is partial).
    pub n_rot: usize,
    /// The trained block size: the noise block is the last token plus up to `block_size - 1`
    /// mask tokens.
    pub block_size: usize,
    /// Target decoder layers whose outputs are fused, in order ([`super::ModelDrafter`]'s
    /// `hidden_layers`; 0-based, the output of layer `i` is the input of layer `i + 1`).
    pub target_layers: Vec<usize>,
    /// The id of the token that stands for "not yet drafted" in the noise block.
    pub mask_token: u32,
    /// Multiplier for the token embeddings (Gemma-style targets), 1 for none.
    pub embedding_scale: f32,
    /// Multiplier for the attention output (`attention_value_scale`), 1 for none.
    pub value_scale: f32,
    /// Attention scale; `1 / sqrt(head_dim)` when unset.
    pub attn_scale: Option<f32>,
    /// Sliding-window attention: the window, and per layer whether it applies. A layer sees
    /// the context positions `p` with `query_position - p < window`.
    pub sliding_window: Option<(usize, Vec<bool>)>,
    /// Whether the noise block attends causally inside itself (default: not).
    pub causal: bool,
    /// Multiplier and soft cap for the draft logits when they are turned into probabilities
    /// (0: none).
    pub logit_scale: f32,
    pub logit_softcap: f32,
    /// DSpark: whether the first block position (the last token itself) already predicts the
    /// next token, so a block drafts `n` tokens from `n` positions. Otherwise (a "bonus
    /// anchor" export) position 0 is not read and the masks give the drafts. Always false
    /// for DFlash, whose first position is never read.
    pub sample_from_anchor: bool,
    /// DFlash 2: the convolutions and the candidate selector ([`super::v2`]).
    pub v2: Option<Dflash2>,
}

/// Which family of checkpoints a [`Config`] is read for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavor {
    Dflash,
    Dspark,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Unsupported(String),
}

fn invalid<T>(msg: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError::Invalid(msg.into()))
}

fn unsupported<T>(msg: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError::Unsupported(msg.into()))
}

impl Config {
    /// Checks the combination of values, once all of them are read.
    fn check(self) -> Result<Self, ConfigError> {
        let c = &self;
        if c.n_layer == 0 || c.n_embd == 0 || c.n_head == 0 || c.head_dim == 0 || c.n_ff == 0 {
            return invalid("a dimension of the drafter is zero");
        }
        if c.n_head_kv == 0 || !c.n_head.is_multiple_of(c.n_head_kv) {
            return invalid(format!(
                "{} attention heads do not divide into {} key/value heads",
                c.n_head, c.n_head_kv
            ));
        }
        if c.n_rot == 0 || c.n_rot > c.head_dim || !c.n_rot.is_multiple_of(2) {
            return invalid(format!(
                "rotary dimensions {} of head size {}",
                c.n_rot, c.head_dim
            ));
        }
        if c.block_size < 2 {
            return invalid(format!(
                "block size {} leaves nothing to draft",
                c.block_size
            ));
        }
        if c.target_layers.is_empty() {
            return invalid("the drafter names no target layers to read hidden states from");
        }
        if let Some((w, pattern)) = &c.sliding_window
            && (*w == 0 || pattern.len() != c.n_layer)
        {
            return invalid("sliding window size or per-layer pattern does not fit the layers");
        }
        if let Some(v2) = &c.v2 {
            if v2.conv_kernel == 0 || v2.conv_group == 0 || !c.n_embd.is_multiple_of(v2.conv_group)
            {
                return invalid(format!(
                    "DFlash 2 convolution: kernel {}, group size {} for a hidden size of {}",
                    v2.conv_kernel, v2.conv_group, c.n_embd
                ));
            }
            if v2.selector_rank == 0 || v2.selector_top_k == 0 {
                return invalid("DFlash 2 selector: rank and top-k must be at least 1");
            }
        }
        Ok(self)
    }

    /// Reads the `dflash.*` metadata of a GGUF (llama.cpp's `convert_hf_to_gguf.py` output).
    /// Variants whose network differs (DFlash2's convolutions and selector, LiLiCorr, the
    /// DeepSeek-V4 backbone) are refused.
    pub fn from_gguf(g: &GgufFile, flavor: Flavor) -> Result<Self, ConfigError> {
        let arch = g.architecture().unwrap_or_default();
        if arch != "dflash" {
            return invalid(format!(
                "architecture `{arch}`, not a DFlash drafter (`dflash`)"
            ));
        }
        let u = |k: &str| g.get(&format!("dflash.{k}")).and_then(|v| v.as_u64());
        let f = |k: &str| g.get(&format!("dflash.{k}")).and_then(|v| v.as_f64());
        let need = |k: &str| -> Result<usize, ConfigError> {
            u(k).map(|v| v as usize)
                .ok_or_else(|| ConfigError::Invalid(format!("metadata `dflash.{k}` is missing")))
        };
        let variant = ["lilicorr.candidate_topk", "dflash.hyper_connection.count"]
            .into_iter()
            .find(|k| g.get(k).and_then(|v| v.as_u64()).is_some_and(|v| v > 0));
        if let Some(key) = variant {
            return unsupported(format!(
                "this drafter uses `{key}` (a LiLiCorr or DeepSeek-V4 variant); \
                 only the DFlash and DFlash 2 networks are implemented"
            ));
        }
        let v2 = dflash2(flavor, |k| u(k).map(|v| v as usize))?;
        if let Some(act) = g.get("dflash.hidden_activation").and_then(|v| match v {
            crate::gguf::Value::String(s) => Some(s),
            _ => None,
        }) && act != "silu"
        {
            return unsupported(format!("hidden activation `{act}` (only silu)"));
        }
        let n_embd = need("embedding_length")?;
        let n_head = need("attention.head_count")?;
        let head_dim = u("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(n_embd / n_head.max(1));
        let target_layers = g
            .get("dflash.target_layers")
            .and_then(|v| {
                v.as_array()
                    .map(|a| a.iter().map(|x| x.as_u64()).collect::<Option<Vec<_>>>())
            })
            .flatten()
            .ok_or_else(|| {
                ConfigError::Invalid("metadata `dflash.target_layers` is missing".into())
            })?;
        // The file stores the layers whose *inputs* are read: layer `i`'s output is the input
        // of layer `i + 1`.
        let target_layers = target_layers
            .into_iter()
            .map(|l| {
                l.checked_sub(1).map(|l| l as usize).ok_or_else(|| {
                    ConfigError::Invalid("target layer 0 has no input to read".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mask_token = g
            .get("tokenizer.ggml.mask_token_id")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                ConfigError::Invalid("metadata `tokenizer.ggml.mask_token_id` is missing".into())
            })? as u32;
        let n_layer = need("block_count")?;
        let sliding_window = match u("attention.sliding_window") {
            Some(w) if w > 0 => {
                let pattern = g
                    .get("dflash.attention.sliding_window_pattern")
                    .and_then(|v| {
                        v.as_array()
                            .map(|a| a.iter().map(|x| x.as_bool()).collect::<Option<Vec<_>>>())
                    })
                    .flatten()
                    .unwrap_or_else(|| vec![true; n_layer]);
                Some((w as usize, pattern))
            }
            _ => None,
        };
        Self {
            n_layer,
            n_embd,
            n_head,
            n_head_kv: need("attention.head_count_kv")?,
            head_dim,
            n_ff: need("feed_forward_length")?,
            rms_eps: f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f32,
            rope_base: f("rope.freq_base").unwrap_or(10_000.0) as f32,
            n_rot: u("rope.dimension_count")
                .map(|v| v as usize)
                .unwrap_or(head_dim),
            block_size: u("block_size")
                .map(|v| v as usize)
                .unwrap_or(DEFAULT_BLOCK_SIZE),
            target_layers,
            mask_token,
            embedding_scale: f("embedding_scale").map_or(1.0, |v| v as f32),
            value_scale: f("attention.value_scale").map_or(1.0, |v| v as f32),
            attn_scale: f("attention.scale").map(|v| v as f32),
            sliding_window,
            causal: g
                .get("dflash.attention.causal")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            logit_scale: f("logit_scale").map_or(0.0, |v| v as f32),
            logit_softcap: f("final_logit_softcapping").map_or(0.0, |v| v as f32),
            sample_from_anchor: flavor == Flavor::Dspark
                && g.get("dflash.sample_from_anchor")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
            v2,
        }
        .check()
    }

    /// Reads a Hugging Face `config.json`: of a `DFlashDraftModel`, or for [`Flavor::Dspark`]
    /// of one of the DSpark classes (`DSparkDraftModel`, `Qwen3DSparkModel`, ...), whose
    /// settings may also be at the top level (DFlash lineage) or in `transformer_layer_config`
    /// next to `aux_hidden_state_layer_ids` (EAGLE-3 lineage, a "bonus anchor" block).
    pub fn from_hf(c: &Json, flavor: Flavor) -> Result<Self, ConfigError> {
        let known: &[&str] = match flavor {
            Flavor::Dflash => &["DFlashDraftModel", "DFlash2DraftModel"],
            Flavor::Dspark => &[
                "DSparkDraftModel",
                "DSparkSpeculator",
                "Qwen3DSparkModel",
                "Lfm2DSparkDraftModel",
                "LingDSparkModel",
            ],
        };
        let archs = c.get("architectures").and_then(Json::as_array);
        if let Some(archs) = archs
            && !archs
                .iter()
                .any(|a| a.as_str().is_some_and(|a| known.contains(&a)))
        {
            return unsupported(format!(
                "architectures {archs:?}: only {known:?} are implemented"
            ));
        }
        let bonus_anchor = c.get("transformer_layer_config").is_some()
            || c.get("aux_hidden_state_layer_ids").is_some();
        let merged;
        let c = match c.get("transformer_layer_config").and_then(Json::as_object) {
            Some(inner) => {
                let mut m = c.as_object().cloned().unwrap_or_default();
                m.extend(inner.iter().map(|(k, v)| (k.clone(), v.clone())));
                merged = Json::Object(m);
                &merged
            }
            None => c,
        };
        if flavor == Flavor::Dspark
            && let Some(t) = c.get("markov_head_type").and_then(Json::as_str)
            && t != "vanilla"
        {
            return unsupported(format!("Markov head type `{t}` (only `vanilla`)"));
        }
        let aux_layers: Option<Vec<u64>> = c
            .get("aux_hidden_state_layer_ids")
            .and_then(Json::as_array)
            .and_then(|a| a.iter().map(Json::as_u64).collect());
        let dflash = c.get("dflash_config").unwrap_or(&Json::Null);
        if dflash
            .get("lilicorr_hidden_size")
            .and_then(Json::as_u64)
            .is_some_and(|v| v > 0)
        {
            return unsupported(
                "`dflash_config.lilicorr_hidden_size` marks a LiLiCorr variant; \
                 only the DFlash and DFlash 2 networks are implemented",
            );
        }
        let v2 = dflash2(flavor, |k| {
            dflash
                .get(k)
                .or_else(|| c.get(k))
                .and_then(Json::as_u64)
                .map(|v| v as usize)
        })?;
        if c.get("rope_is_neox_style").and_then(Json::as_bool) == Some(false) {
            return unsupported("interleaved (non-NeoX) rotary embeddings");
        }
        let num = |k: &str| -> Option<u64> { c.get(k).and_then(Json::as_u64) };
        let need = |k: &str| -> Result<usize, ConfigError> {
            num(k)
                .map(|v| v as usize)
                .ok_or_else(|| ConfigError::Invalid(format!("config.json has no `{k}`")))
        };
        let float = |v: Option<&Json>| v.and_then(Json::as_f64);
        let n_embd = need("hidden_size")?;
        let n_head = need("num_attention_heads")?;
        let head_dim = num("head_dim")
            .map(|v| v as usize)
            .unwrap_or(n_embd / n_head.max(1));
        let rope = c
            .get("rope_parameters")
            .or_else(|| c.get("rope_scaling"))
            .unwrap_or(&Json::Null);
        let rope_base = float(c.get("rope_theta"))
            .or_else(|| float(rope.get("rope_theta")))
            .unwrap_or(10_000.0) as f32;
        let n_rot =
            (head_dim as f64 * float(rope.get("partial_rotary_factor")).unwrap_or(1.0)) as usize;
        let target_layers = dflash
            .get("target_layer_ids")
            .or_else(|| c.get("target_layer_ids"))
            .and_then(Json::as_array)
            .and_then(|a| {
                a.iter()
                    .map(|x| x.as_u64().map(|x| x as usize))
                    .collect::<Option<Vec<_>>>()
            })
            .or_else(|| {
                // The EAGLE-3 spelling counts layers from 1.
                aux_layers.as_ref().and_then(|a| {
                    a.iter()
                        .map(|&l| l.checked_sub(1).map(|l| l as usize))
                        .collect()
                })
            })
            .ok_or_else(|| {
                ConfigError::Invalid("config.json has no `dflash_config.target_layer_ids`".into())
            })?;
        let mask_token = dflash
            .get("mask_token_id")
            .or_else(|| c.get("mask_token_id"))
            .and_then(Json::as_u64)
            .ok_or_else(|| {
                ConfigError::Invalid("config.json has no `dflash_config.mask_token_id`".into())
            })? as u32;
        let n_layer = need("num_hidden_layers")?;
        let use_swa = dflash.get("use_swa").and_then(Json::as_bool) == Some(true);
        let sliding = c
            .get("use_sliding_window")
            .and_then(Json::as_bool)
            .unwrap_or(false)
            || use_swa;
        let window = dflash
            .get("swa_window_size")
            .and_then(Json::as_u64)
            .or_else(|| num("sliding_window"));
        let sliding_window = match (sliding, window) {
            (true, Some(w)) if w > 0 => {
                let pattern = if use_swa {
                    vec![true; n_layer]
                } else {
                    c.get("layer_types")
                        .and_then(Json::as_array)
                        .map(|a| {
                            a.iter()
                                .map(|t| t.as_str() == Some("sliding_attention"))
                                .collect()
                        })
                        .unwrap_or_else(|| vec![true; n_layer])
                };
                Some((w as usize, pattern))
            }
            _ => None,
        };
        let causal = c
            .get("is_causal")
            .and_then(Json::as_bool)
            .or_else(|| dflash.get("causal").and_then(Json::as_bool))
            .unwrap_or(false);
        let output_scale = float(dflash.get("output_multiplier"))
            .or_else(|| float(c.get("output_multiplier")))
            .unwrap_or(0.0);
        let softcap = float(dflash.get("final_logit_softcapping"))
            .or_else(|| float(c.get("final_logit_softcapping")))
            .unwrap_or(0.0);
        Self {
            n_layer,
            n_embd,
            n_head,
            n_head_kv: need("num_key_value_heads")?,
            head_dim,
            n_ff: need("intermediate_size")?,
            rms_eps: float(c.get("rms_norm_eps")).unwrap_or(1e-6) as f32,
            rope_base,
            n_rot,
            block_size: dflash
                .get("block_size")
                .and_then(Json::as_u64)
                .or_else(|| num("block_size"))
                .map_or(DEFAULT_BLOCK_SIZE, |v| v as usize),
            target_layers,
            mask_token,
            embedding_scale: float(dflash.get("input_embedding_scale"))
                .or_else(|| float(c.get("input_embedding_scale")))
                .unwrap_or(1.0) as f32,
            value_scale: float(dflash.get("attention_value_scale")).unwrap_or(1.0) as f32,
            attn_scale: None,
            sliding_window,
            causal,
            logit_scale: output_scale as f32,
            logit_softcap: softcap.max(0.0) as f32,
            sample_from_anchor: flavor == Flavor::Dspark
                && c.get("sample_from_anchor")
                    .and_then(Json::as_bool)
                    .unwrap_or(!bonus_anchor),
            v2,
        }
        .check()
    }
}

/// DFlash 2's settings from `get` (`conv_kernel_size`, `conv_group_size`, `selector_rank`,
/// `selector_top_k`, under the GGUF's `dflash.` prefix or in `dflash_config`): `None` for a
/// DFlash without them. A drafter with only some of them, or a DSpark with them, is refused.
fn dflash2(
    flavor: Flavor,
    get: impl Fn(&str) -> Option<usize>,
) -> Result<Option<Dflash2>, ConfigError> {
    let keys = [
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
    ];
    let v: Vec<Option<usize>> = keys.iter().map(|k| get(k).filter(|&v| v > 0)).collect();
    if v.iter().all(Option::is_none) {
        return Ok(None);
    }
    if flavor == Flavor::Dspark {
        return unsupported("a DSpark drafter with DFlash 2 convolutions or selector");
    }
    if let Some(i) = v.iter().position(Option::is_none) {
        return unsupported(format!(
            "a DFlash 2 drafter without `{}` (convolutions and selector are implemented \
             together only)",
            keys[i]
        ));
    }
    Ok(Some(Dflash2 {
        conv_kernel: v[0].unwrap(),
        conv_group: v[1].unwrap(),
        selector_rank: v[2].unwrap(),
        selector_top_k: v[3].unwrap(),
    }))
}
