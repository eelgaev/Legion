//! An EAGLE-3 checkpoint's `config.json`, and the draft-to-target vocabulary map.

use serde_json::Value;

/// How the draft layer's RoPE frequencies are scaled (`rope_scaling` / `rope_parameters`).
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    None,
    /// Positions divided by `factor`.
    Linear {
        factor: f32,
    },
    /// Llama 3.1: per-frequency factors that stretch only the low frequencies.
    Llama3 {
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_position: f32,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Eagle3Config {
    /// The draft layer's width. It reads the target's token embeddings, so it is the target's
    /// width too.
    pub hidden: usize,
    /// The target's hidden width (`target_hidden_size`, else `hidden`); `fc` reads
    /// `n_aux * target_hidden` values.
    pub target_hidden: usize,
    /// Target hidden states fused by `fc` (3: low, middle and high layer).
    pub n_aux: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub rope_scaling: RopeScaling,
    pub max_pos: usize,
    /// The target's vocabulary (`vocab_size`): rows of `embed_tokens`.
    pub vocab: usize,
    /// Rows of `lm_head`; below `vocab` the checkpoint carries a `d2t` map.
    pub draft_vocab: usize,
    /// `eagle_config.eagle_aux_hidden_state_layer_ids`: target layers whose *outputs* feed
    /// `fc` (SGLang's reading of the field; see [`Self::target_layers`]).
    pub layer_ids: Option<Vec<usize>>,
    /// Bias on the q/k/v projections (`attention_bias`).
    pub attention_bias: bool,
    /// The residual of the layer is the normed hidden state (`norm_before_residual`).
    pub norm_before_residual: bool,
    /// One RMSNorm over the concatenated hidden states, before `fc` (`norm_before_fc`).
    pub norm_before_fc: bool,
    /// One RMSNorm per hidden state, before `fc` (`fc_norm`).
    pub fc_norm: bool,
    /// The hidden state fed back to the next draft step is the normed one (`norm_output`).
    pub norm_output: bool,
    /// Multiplies the logits (`logit_scale`).
    pub logit_scale: f32,
}

impl Eagle3Config {
    pub fn parse(config: &Value) -> Result<Self, String> {
        let get = |k: &str| config.get(k).filter(|v| !v.is_null());
        let uint = |k: &str| -> Result<Option<usize>, String> {
            match get(k) {
                None => Ok(None),
                Some(v) => v
                    .as_u64()
                    .map(|n| Some(n as usize))
                    .ok_or_else(|| format!("config.json: `{k}` is not an integer")),
            }
        };
        let need = |k: &str| uint(k)?.ok_or_else(|| format!("config.json: no `{k}`"));
        let float = |k: &str| get(k).and_then(Value::as_f64);
        let flag = |k: &str, eagle: Option<&Value>| {
            get(k)
                .or_else(|| eagle.and_then(|e| e.get(k)))
                .and_then(Value::as_bool)
                .unwrap_or(false)
        };

        if let Some(n) = uint("num_hidden_layers")?
            && n != 1
        {
            return Err(format!(
                "config.json: an EAGLE-3 head has one decoder layer, not {n}"
            ));
        }
        if let Some(act) = get("hidden_act").and_then(Value::as_str)
            && act != "silu"
        {
            return Err(format!("config.json: hidden_act `{act}` is not supported"));
        }
        let eagle = get("eagle_config");
        if eagle
            .and_then(|e| e.get("use_aux_hidden_state"))
            .and_then(Value::as_bool)
            == Some(false)
        {
            return Err(
                "config.json: eagle_config.use_aux_hidden_state = false is not supported".into(),
            );
        }
        let layer_ids = match eagle
            .and_then(|e| e.get("eagle_aux_hidden_state_layer_ids"))
            .or_else(|| get("eagle_aux_hidden_state_layer_ids"))
        {
            None | Some(Value::Null) => None,
            Some(v) => Some(
                v.as_array()
                    .and_then(|a| a.iter().map(|x| x.as_u64().map(|n| n as usize)).collect())
                    .ok_or(
                        "config.json: eagle_aux_hidden_state_layer_ids is not a list of integers",
                    )?,
            ),
        };
        let n_aux = uint("num_aux_hidden_states")?
            .or(layer_ids.as_ref().map(Vec::len))
            .unwrap_or(3);

        let hidden = need("hidden_size")?;
        let n_head = need("num_attention_heads")?;
        let n_kv_head = uint("num_key_value_heads")?.unwrap_or(n_head);
        let head_dim = match uint("head_dim")? {
            Some(d) => d,
            None if n_head > 0 => hidden / n_head,
            None => 0,
        };
        let vocab = need("vocab_size")?;
        let draft_vocab = uint("draft_vocab_size")?.unwrap_or(vocab);
        let (rope_theta, rope_scaling) = parse_rope(config)?;
        let c = Self {
            hidden,
            target_hidden: uint("target_hidden_size")?.unwrap_or(hidden),
            n_aux,
            n_ff: need("intermediate_size")?,
            n_head,
            n_kv_head,
            head_dim,
            eps: float("rms_norm_eps").unwrap_or(1e-6) as f32,
            rope_theta,
            rope_scaling,
            max_pos: uint("max_position_embeddings")?.unwrap_or(2048),
            vocab,
            draft_vocab,
            layer_ids,
            attention_bias: flag("attention_bias", None),
            norm_before_residual: flag("norm_before_residual", None),
            norm_before_fc: flag("norm_before_fc", eagle),
            fc_norm: flag("fc_norm", None),
            norm_output: flag("norm_output", None),
            logit_scale: float("logit_scale").unwrap_or(1.0) as f32,
        };
        if [
            hidden,
            n_head,
            n_kv_head,
            head_dim,
            c.n_ff,
            c.n_aux,
            vocab,
            draft_vocab,
        ]
        .contains(&0)
            || !n_head.is_multiple_of(n_kv_head)
            || !head_dim.is_multiple_of(2)
        {
            return Err(format!(
                "config.json: inconsistent shapes (hidden {hidden}, {n_head} heads of {head_dim}, \
                 {n_kv_head} KV heads)"
            ));
        }
        if draft_vocab > vocab {
            return Err(format!(
                "config.json: draft_vocab_size {draft_vocab} exceeds vocab_size {vocab}"
            ));
        }
        Ok(c)
    }

    /// The target layers whose outputs the head reads, low to high.
    ///
    /// With `eagle_aux_hidden_state_layer_ids`, those ids: each is the layer whose output is
    /// taken (SGLang's reading of the field, `layers_to_capture = id + 1` on layer inputs; SpecForge
    /// writes it so). Without, the EAGLE-3 recipe: HF `hidden_states[2]`, `[n / 2]` and `[n - 3]`
    /// (inputs of layers 2, n / 2 and n - 3), which are the outputs of layers 1, n / 2 - 1 and
    /// n - 4. vLLM reads explicit ids as hidden-state indices (one layer later); a checkpoint
    /// made for that reading gets a shifted, still valid, input here.
    pub fn target_layers(&self, n_layer: usize) -> Result<Vec<usize>, String> {
        let ids = match &self.layer_ids {
            Some(ids) => ids.clone(),
            None if self.n_aux == 3 => {
                vec![
                    1,
                    (n_layer / 2).saturating_sub(1),
                    n_layer.saturating_sub(4),
                ]
            }
            None => {
                return Err(format!(
                    "config.json: {} hidden states fused and no eagle_aux_hidden_state_layer_ids",
                    self.n_aux
                ));
            }
        };
        if ids.len() != self.n_aux {
            return Err(format!(
                "config.json: {} layer ids for {} fused hidden states",
                ids.len(),
                self.n_aux
            ));
        }
        if let Some(&bad) = ids.iter().find(|&&il| il >= n_layer) {
            return Err(format!(
                "config.json: target layer {bad} does not exist (the target has {n_layer})"
            ));
        }
        Ok(ids)
    }

    /// `rope_freqs`: per rotary pair divisors for the base frequencies, for Llama 3 scaling.
    ///
    /// Adapted from `LlamaModel.generate_extra_tensors` in llama.cpp's `convert_hf_to_gguf.py`
    /// (MIT).
    pub fn rope_factors(&self) -> Option<Vec<f32>> {
        let RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position,
        } = self.rope_scaling
        else {
            return None;
        };
        let low_wavelen = original_max_position / low_freq_factor;
        let high_wavelen = original_max_position / high_freq_factor;
        Some(
            (0..self.head_dim / 2)
                .map(|i| {
                    let freq = self
                        .rope_theta
                        .powf(-(2.0 * i as f32) / self.head_dim as f32);
                    let wavelen = 2.0 * std::f32::consts::PI / freq;
                    if wavelen < high_wavelen {
                        1.0
                    } else if wavelen > low_wavelen {
                        factor
                    } else {
                        let smooth = (original_max_position / wavelen - low_freq_factor)
                            / (high_freq_factor - low_freq_factor);
                        1.0 / ((1.0 - smooth) / factor + smooth)
                    }
                })
                .collect(),
        )
    }

    /// RoPE's `freq_scale` argument: `1 / factor` for linear scaling.
    pub fn rope_freq_scale(&self) -> f32 {
        match self.rope_scaling {
            RopeScaling::Linear { factor } => 1.0 / factor,
            _ => 1.0,
        }
    }
}

fn parse_rope(config: &Value) -> Result<(f32, RopeScaling), String> {
    let params = config
        .get("rope_scaling")
        .filter(|v| !v.is_null())
        .or_else(|| config.get("rope_parameters").filter(|v| !v.is_null()));
    let theta = config
        .get("rope_theta")
        .or_else(|| params.and_then(|p| p.get("rope_theta")))
        .and_then(Value::as_f64)
        .unwrap_or(10_000.0) as f32;
    let Some(p) = params else {
        return Ok((theta, RopeScaling::None));
    };
    let f = |k: &str| p.get(k).and_then(Value::as_f64).map(|x| x as f32);
    let kind = p
        .get("rope_type")
        .or_else(|| p.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("default");
    let scaling = match kind {
        "default" => RopeScaling::None,
        "linear" => RopeScaling::Linear {
            factor: f("factor").ok_or("config.json: rope_scaling has no factor")?,
        },
        "llama3" => {
            let need = |k: &str| f(k).ok_or(format!("config.json: rope_scaling has no {k}"));
            RopeScaling::Llama3 {
                factor: need("factor")?,
                low_freq_factor: need("low_freq_factor")?,
                high_freq_factor: need("high_freq_factor")?,
                original_max_position: need("original_max_position_embeddings")?,
            }
        }
        other => {
            return Err(format!(
                "config.json: rope_scaling `{other}` is not supported"
            ));
        }
    };
    Ok((theta, scaling))
}

/// Draft vocabulary ids to target token ids (`d2t`) and back (`t2d`).
///
/// EAGLE-3 heads predict over a smaller vocabulary (the most frequent tokens of the training
/// data). The checkpoint's `d2t` holds, per draft id `i`, the offset to add: the target token is
/// `i + d2t[i]`. Its `t2d` is the mask of target tokens that have a draft id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabMap {
    /// Target token of each draft id.
    to_target: Vec<u32>,
    /// Draft id of each target token, `u32::MAX` for tokens outside the draft vocabulary.
    to_draft: Vec<u32>,
}

impl VocabMap {
    /// The draft vocabulary is the target's.
    pub fn identity(n: usize) -> Self {
        let ids: Vec<u32> = (0..n as u32).collect();
        Self {
            to_target: ids.clone(),
            to_draft: ids,
        }
    }

    /// From a checkpoint's `d2t` offsets (and `t2d` mask, if it has one) for a target of
    /// `n_vocab` tokens.
    pub fn from_offsets(
        offsets: &[i64],
        n_vocab: usize,
        t2d: Option<&[bool]>,
    ) -> Result<Self, String> {
        let mut to_target = Vec::with_capacity(offsets.len());
        let mut to_draft = vec![u32::MAX; n_vocab];
        for (i, &off) in offsets.iter().enumerate() {
            let t = i as i64 + off;
            if t < 0 || t as usize >= n_vocab {
                return Err(format!(
                    "d2t[{i}] = {off} maps to token {t}, outside the {n_vocab}-token vocabulary"
                ));
            }
            if to_draft[t as usize] != u32::MAX {
                return Err(format!("d2t maps two draft ids to token {t}"));
            }
            to_draft[t as usize] = i as u32;
            to_target.push(t as u32);
        }
        if let Some(mask) = t2d {
            if mask.len() != n_vocab {
                return Err(format!(
                    "t2d has {} entries for a {n_vocab}-token vocabulary",
                    mask.len()
                ));
            }
            for (t, &m) in mask.iter().enumerate() {
                if m != (to_draft[t] != u32::MAX) {
                    return Err(format!("t2d and d2t disagree about token {t}"));
                }
            }
        }
        Ok(Self {
            to_target,
            to_draft,
        })
    }

    /// Draft ids.
    pub fn len(&self) -> usize {
        self.to_target.len()
    }

    pub fn is_empty(&self) -> bool {
        self.to_target.is_empty()
    }

    /// Target tokens (of the target's vocabulary).
    pub fn n_vocab(&self) -> usize {
        self.to_draft.len()
    }

    pub fn to_target(&self, draft: usize) -> u32 {
        self.to_target[draft]
    }

    /// `None` for a token outside the draft vocabulary.
    pub fn to_draft(&self, token: u32) -> Option<usize> {
        match self.to_draft.get(token as usize) {
            Some(&d) if d != u32::MAX => Some(d as usize),
            _ => None,
        }
    }

    /// A distribution over draft ids as one over `n` target tokens (0 outside the draft
    /// vocabulary). Every target token of the map must be below `n`.
    pub fn to_target_probs(&self, probs: &[f32], n: usize) -> Vec<f32> {
        assert_eq!(probs.len(), self.len());
        let mut out = vec![0.0; n];
        for (d, &p) in probs.iter().enumerate() {
            out[self.to_target[d] as usize] = p;
        }
        out
    }

    /// Whether every draft id maps to a token below `n`.
    pub fn fits(&self, n: usize) -> bool {
        self.to_target.iter().all(|&t| (t as usize) < n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn base() -> Value {
        json!({
            "num_hidden_layers": 1, "hidden_size": 64, "intermediate_size": 96,
            "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 259,
            "draft_vocab_size": 128, "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
            "hidden_act": "silu",
        })
    }

    #[test]
    fn parses_and_defaults() {
        let c = Eagle3Config::parse(&base()).unwrap();
        assert_eq!((c.head_dim, c.n_aux, c.target_hidden), (16, 3, 64));
        assert_eq!((c.vocab, c.draft_vocab), (259, 128));
        assert_eq!(c.rope_scaling, RopeScaling::None);
        // EAGLE-3's recipe on a 32-layer target: inputs of layers 2, 16 and 29.
        assert_eq!(c.target_layers(32).unwrap(), vec![1, 15, 28]);
        assert!(c.target_layers(32).unwrap().iter().all(|&l| l < 32));
    }

    #[test]
    fn layer_ids_from_the_config() {
        let mut j = base();
        j["eagle_config"] = json!({"eagle_aux_hidden_state_layer_ids": [0, 2, 3]});
        let c = Eagle3Config::parse(&j).unwrap();
        assert_eq!(c.target_layers(4).unwrap(), vec![0, 2, 3]);
        let e = c.target_layers(3).unwrap_err();
        assert!(e.contains("layer 3 does not exist"), "{e}");
        j["eagle_config"] = json!({"eagle_aux_hidden_state_layer_ids": [0, 2]});
        assert_eq!(Eagle3Config::parse(&j).unwrap().n_aux, 2);
    }

    #[test]
    fn rejects_what_it_cannot_run() {
        let mut j = base();
        j["num_hidden_layers"] = json!(2);
        assert!(
            Eagle3Config::parse(&j)
                .unwrap_err()
                .contains("one decoder layer")
        );
        let mut j = base();
        j["hidden_act"] = json!("gelu");
        assert!(Eagle3Config::parse(&j).unwrap_err().contains("gelu"));
        let mut j = base();
        j["rope_scaling"] = json!({"rope_type": "yarn", "factor": 4.0});
        assert!(Eagle3Config::parse(&j).unwrap_err().contains("yarn"));
        let mut j = base();
        j["draft_vocab_size"] = json!(1000);
        assert!(Eagle3Config::parse(&j).unwrap_err().contains("exceeds"));
        let mut j = base();
        j.as_object_mut().unwrap().remove("hidden_size");
        assert!(Eagle3Config::parse(&j).unwrap_err().contains("hidden_size"));
    }

    #[test]
    fn llama3_rope_factors() {
        let mut j = base();
        j["rope_scaling"] = json!({
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192,
        });
        let c = Eagle3Config::parse(&j).unwrap();
        let f = c.rope_factors().unwrap();
        assert_eq!(f.len(), 8);
        // The highest frequency is untouched, the lowest is stretched by the factor.
        assert_eq!(f[0], 1.0);
        assert_eq!(*f.last().unwrap(), 8.0);
        assert!(f.windows(2).all(|w| w[0] <= w[1]), "{f:?}");
        assert_eq!(c.rope_freq_scale(), 1.0);
        let mut j = base();
        j["rope_scaling"] = json!({"type": "linear", "factor": 4.0});
        let c = Eagle3Config::parse(&j).unwrap();
        assert_eq!(c.rope_freq_scale(), 0.25);
        assert!(c.rope_factors().is_none());
    }

    #[test]
    fn vocab_map_round_trip() {
        // Draft id i is target token 2 i + 1.
        let offsets: Vec<i64> = (0..10).map(|i| i + 1).collect();
        let mask: Vec<bool> = (0..25).map(|t| t % 2 == 1 && t < 20).collect();
        let m = VocabMap::from_offsets(&offsets, 25, Some(&mask)).unwrap();
        assert_eq!((m.len(), m.n_vocab()), (10, 25));
        for d in 0..10 {
            let t = m.to_target(d);
            assert_eq!(t as usize, 2 * d + 1);
            assert_eq!(m.to_draft(t), Some(d));
        }
        for t in 0..25u32 {
            match m.to_draft(t) {
                Some(d) => assert_eq!(m.to_target(d), t),
                None => assert!(!mask[t as usize]),
            }
        }
        assert_eq!(m.to_draft(25), None);
        let probs: Vec<f32> = (0..10).map(|i| (i + 1) as f32 / 55.0).collect();
        let q = m.to_target_probs(&probs, 25);
        assert!(m.fits(25) && !m.fits(19));
        assert_eq!(q.len(), 25);
        assert!((q.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(q[7], probs[3]);
        assert_eq!(q[8], 0.0);
        let id = VocabMap::identity(5);
        assert_eq!((id.to_target(3), id.to_draft(3)), (3, Some(3)));
    }

    #[test]
    fn vocab_map_rejects_bad_maps() {
        assert!(VocabMap::from_offsets(&[0, 100], 10, None).is_err());
        assert!(VocabMap::from_offsets(&[-1], 10, None).is_err());
        // Two draft ids onto token 1.
        assert!(VocabMap::from_offsets(&[1, 0], 10, None).is_err());
        let mask = vec![true; 10];
        assert!(VocabMap::from_offsets(&[0, 0], 10, Some(&mask)).is_err());
        assert!(VocabMap::from_offsets(&[0, 0], 10, Some(&mask[..5])).is_err());
    }
}
