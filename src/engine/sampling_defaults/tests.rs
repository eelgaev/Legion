use super::*;
use crate::gguf::{GgufFile, GgufWriter, Value};

fn layer(t: f32, k: u32) -> SamplingLayer {
    SamplingLayer {
        temperature: Some(t),
        top_k: Some(k),
        ..Default::default()
    }
}

#[test]
fn operator_over_file_over_openai() {
    let model = ModelSampling {
        file: SamplingLayer {
            top_p: Some(0.95),
            ..layer(0.7, 20)
        },
        architecture: layer(0.1, 99),
    };
    let operator = SamplingLayer {
        temperature: Some(0.3),
        ..Default::default()
    };
    let d = SamplingDefaults::resolve(DefaultsMode::Model, &operator, &model);
    assert_eq!(
        d.temperature,
        Param {
            value: 0.3,
            source: Source::Operator
        }
    );
    assert_eq!(
        d.top_k,
        Param {
            value: 20,
            source: Source::Model
        }
    );
    assert_eq!(d.top_p.value, 0.95);
    // Not in the file: OpenAI's, not the architecture's (the file declares something).
    assert_eq!(
        d.min_p,
        Param {
            value: 0.0,
            source: Source::OpenAi
        }
    );
    assert_eq!(d.repetition_penalty.value, 1.0);
}

#[test]
fn architecture_only_when_the_file_declares_nothing() {
    let model = ModelSampling {
        file: SamplingLayer::default(),
        architecture: SamplingLayer::for_architecture("gemma4"),
    };
    let d = SamplingDefaults::resolve(DefaultsMode::Model, &SamplingLayer::default(), &model);
    assert_eq!(
        (d.temperature.value, d.top_k.value, d.top_p.value),
        (1.0, 64, 0.95)
    );
    assert_eq!(d.top_k.source, Source::Architecture);
    // An architecture without a table: OpenAI's.
    let none = ModelSampling {
        architecture: SamplingLayer::for_architecture("nemotron_h"),
        ..Default::default()
    };
    let d = SamplingDefaults::resolve(DefaultsMode::Model, &SamplingLayer::default(), &none);
    assert_eq!(d, SamplingDefaults::default());
}

#[test]
fn openai_mode_keeps_only_the_operator() {
    let model = ModelSampling {
        file: layer(0.7, 20),
        architecture: layer(0.1, 99),
    };
    let operator = SamplingLayer {
        top_p: Some(0.8),
        ..Default::default()
    };
    let d = SamplingDefaults::resolve(DefaultsMode::Openai, &operator, &model);
    assert_eq!(d.temperature.value, 1.0);
    assert_eq!(d.top_k.value, 0);
    assert_eq!(d.temperature.source, Source::OpenAi);
    assert_eq!(
        d.top_p,
        Param {
            value: 0.8,
            source: Source::Operator
        }
    );
    // OpenAI's defaults are SamplingParams::default()'s.
    let o = SamplingParams::default();
    let d = SamplingDefaults::default();
    assert_eq!(
        (
            d.temperature.value,
            d.top_p.value,
            d.top_k.value,
            d.min_p.value
        ),
        (o.temperature, o.top_p, o.top_k, o.min_p)
    );
    assert_eq!(
        (
            d.presence_penalty.value,
            d.frequency_penalty.value,
            d.repetition_penalty.value
        ),
        (
            o.presence_penalty,
            o.frequency_penalty,
            o.repetition_penalty
        )
    );
}

/// A tiny GGUF with the keys Qwen3.8-Flash-Next's GGUFs carry (types as there: top_k i32).
#[test]
fn reads_general_sampling_keys_from_a_gguf() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sampling.gguf");
    let mut w = GgufWriter::new();
    for (k, v) in [
        ("general.architecture", Value::String("qwen4exp".into())),
        ("general.sampling.top_k", Value::I32(20)),
        ("general.sampling.top_p", Value::F32(0.95)),
        ("general.sampling.temp", Value::F32(1.0)),
        ("general.sampling.min_p", Value::F32(0.05)),
        ("general.sampling.penalty_repeat", Value::F32(1.1)),
    ] {
        w.set(k, &v).unwrap();
    }
    w.write(&path).unwrap();
    let g = GgufFile::open(&path).unwrap();
    let m = ModelSampling::from_gguf(&g);
    assert_eq!(
        m.file,
        SamplingLayer {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(20),
            min_p: Some(0.05),
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: Some(1.1),
        }
    );
    assert_eq!(m.architecture, SamplingLayer::for_architecture("qwen4exp"));
    let d = SamplingDefaults::resolve(DefaultsMode::Model, &SamplingLayer::default(), &m);
    assert_eq!(d.top_k.source, Source::Model);
    assert!(
        d.describe().contains("top_k=20 (model file)"),
        "{}",
        d.describe()
    );
}

#[test]
fn ignores_out_of_range_and_absent_keys() {
    let l = SamplingLayer::from_metadata(|k| match k {
        "general.sampling.temp" => Some(5.0),
        "general.sampling.top_p" => Some(0.0),
        "general.sampling.top_k" => Some(-1.0),
        _ => None,
    });
    assert!(l.is_empty(), "{l:?}");
}

/// A `generation_config.json` as Hugging Face releases ship it; written back as GGUF keys and
/// read again, it gives the same layer.
#[test]
fn reads_generation_config() {
    let j = serde_json::json!({
        "bos_token_id": 1,
        "do_sample": true,
        "eos_token_id": [1, 106],
        "temperature": 1.0,
        "top_k": 64,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
        "transformers_version": "4.57.0"
    });
    let l = SamplingLayer::from_generation_config(&j);
    assert_eq!(
        l,
        SamplingLayer {
            temperature: Some(1.0),
            top_p: Some(0.95),
            top_k: Some(64),
            repetition_penalty: Some(1.05),
            ..Default::default()
        }
    );
    let meta: std::collections::HashMap<_, _> = l.metadata().into_iter().collect();
    assert_eq!(SamplingLayer::from_metadata(|k| meta.get(k)?.as_f64()), l);
    // transformers decodes greedily when do_sample is false
    let greedy = serde_json::json!({"do_sample": false, "temperature": 0.7, "top_k": 20});
    let l = SamplingLayer::from_generation_config(&greedy);
    assert_eq!((l.temperature, l.top_k), (Some(0.0), Some(20)));
}
