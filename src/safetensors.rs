//! Safetensors files and HF checkpoint directories.
//!
//! A safetensors file is an 8-byte little-endian header length, a JSON header and the raw tensor
//! bytes. [`SafetensorsFile`] maps one file and validates its header; [`Checkpoint`] opens a model
//! directory (single `model.safetensors` or a sharded `model.safetensors.index.json`) plus its
//! `config.json`. Tensor data is never copied: [`SafetensorsFile::data`] borrows the mapping.

use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde::Serialize;
use serde_json::Value;

/// The reference implementation refuses headers above 100 MB; so do we.
const MAX_HEADER_LEN: u64 = 100_000_000;

pub const INDEX_FILE: &str = "model.safetensors.index.json";
pub const SINGLE_FILE: &str = "model.safetensors";

#[derive(Debug, thiserror::Error)]
pub enum SafetensorsError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path}: {reason}")]
    Invalid { path: PathBuf, reason: String },
    #[error("{0}: no model.safetensors or model.safetensors.index.json")]
    NoCheckpoint(PathBuf),
    #[error("no tensor named `{0}`")]
    NoTensor(String),
}

/// Tensor element type. Unknown names are kept as [`Dtype::Other`] (no size check for those).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dtype {
    F64,
    F32,
    F16,
    BF16,
    F8E4M3,
    F8E5M2,
    /// Power-of-two scales of the OCP microscaling formats (MXFP4).
    F8E8M0,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
    Other(String),
}

impl Dtype {
    pub fn parse(s: &str) -> Dtype {
        match s {
            "F64" => Dtype::F64,
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "F8_E4M3" => Dtype::F8E4M3,
            "F8_E5M2" => Dtype::F8E5M2,
            "F8_E8M0" => Dtype::F8E8M0,
            "I64" => Dtype::I64,
            "I32" => Dtype::I32,
            "I16" => Dtype::I16,
            "I8" => Dtype::I8,
            "U64" => Dtype::U64,
            "U32" => Dtype::U32,
            "U16" => Dtype::U16,
            "U8" => Dtype::U8,
            "BOOL" => Dtype::Bool,
            other => Dtype::Other(other.to_string()),
        }
    }

    /// The name used in the safetensors header.
    pub fn name(&self) -> &str {
        match self {
            Dtype::F64 => "F64",
            Dtype::F32 => "F32",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::F8E4M3 => "F8_E4M3",
            Dtype::F8E5M2 => "F8_E5M2",
            Dtype::F8E8M0 => "F8_E8M0",
            Dtype::I64 => "I64",
            Dtype::I32 => "I32",
            Dtype::I16 => "I16",
            Dtype::I8 => "I8",
            Dtype::U64 => "U64",
            Dtype::U32 => "U32",
            Dtype::U16 => "U16",
            Dtype::U8 => "U8",
            Dtype::Bool => "BOOL",
            Dtype::Other(s) => s,
        }
    }

    /// Bytes per element, `None` for unknown dtypes.
    pub fn size(&self) -> Option<usize> {
        match self {
            Dtype::F64 | Dtype::I64 | Dtype::U64 => Some(8),
            Dtype::F32 | Dtype::I32 | Dtype::U32 => Some(4),
            Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => Some(2),
            Dtype::F8E4M3 | Dtype::F8E5M2 | Dtype::F8E8M0 | Dtype::I8 | Dtype::U8 | Dtype::Bool => {
                Some(1)
            }
            Dtype::Other(_) => None,
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl Serialize for Dtype {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Absolute byte range of the tensor in the file.
    pub data_range: Range<usize>,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        self.data_range.len()
    }
}

/// One memory-mapped safetensors file with a validated header.
pub struct SafetensorsFile {
    path: PathBuf,
    mmap: Mmap,
    metadata: BTreeMap<String, String>,
    /// Sorted by offset.
    tensors: Vec<TensorInfo>,
    by_name: BTreeMap<String, usize>,
}

impl SafetensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SafetensorsError> {
        let path = path.as_ref().to_path_buf();
        let io = |source| SafetensorsError::Io {
            path: path.clone(),
            source,
        };
        let file = File::open(&path).map_err(io)?;
        // SAFETY: the mapping is read-only; as with any mmap, truncating the file while it is
        // mapped is undefined, which model files on disk do not do.
        let mmap = unsafe { Mmap::map(&file) }.map_err(io)?;
        Self::from_mmap(path, mmap)
    }

    fn from_mmap(path: PathBuf, mmap: Mmap) -> Result<Self, SafetensorsError> {
        let (metadata, tensors) =
            parse_header(&mmap).map_err(|reason| SafetensorsError::Invalid {
                path: path.clone(),
                reason,
            })?;
        let by_name = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        Ok(Self {
            path,
            mmap,
            metadata,
            tensors,
            by_name,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Unmaps the file pages this mapping has touched (`MADV_DONTNEED`), so the page cache
    /// can drop them. A later read faults the same bytes back in from the file.
    pub fn release_pages(&self) {
        // SAFETY: the mapping is a read-only map of a model file that does not change, so a
        // dropped page reads back the same bytes; borrowed slices stay valid (same addresses).
        let _ = unsafe {
            self.mmap
                .unchecked_advise(memmap2::UncheckedAdvice::DontNeed)
        };
    }

    /// The `__metadata__` string map (empty when absent).
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Tensors in file order (by offset).
    pub fn tensors(&self) -> Vec<TensorInfo> {
        self.tensors.clone()
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).map(|&i| &self.tensors[i])
    }

    /// The raw bytes of one tensor, borrowed from the mapping.
    pub fn data(&self, name: &str) -> Result<&[u8], SafetensorsError> {
        let t = self
            .tensor(name)
            .ok_or_else(|| SafetensorsError::NoTensor(name.to_string()))?;
        Ok(&self.mmap[t.data_range.clone()])
    }
}

type Header = (BTreeMap<String, String>, Vec<TensorInfo>);

/// Parse and validate the header of a whole file image. Errors are plain reasons.
fn parse_header(buf: &[u8]) -> Result<Header, String> {
    let len_bytes: [u8; 8] = buf
        .get(..8)
        .ok_or("file shorter than the 8-byte header length")?
        .try_into()
        .unwrap();
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > MAX_HEADER_LEN {
        return Err(format!(
            "header length {header_len} exceeds {MAX_HEADER_LEN}"
        ));
    }
    let data_start = 8 + header_len as usize;
    let header = buf
        .get(8..data_start)
        .ok_or_else(|| format!("header length {header_len} runs past the end of the file"))?;
    let json: Value =
        serde_json::from_slice(header).map_err(|e| format!("header is not valid JSON: {e}"))?;
    let obj = json.as_object().ok_or("header is not a JSON object")?;
    let data_len = buf.len() - data_start;

    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            let m = v.as_object().ok_or("__metadata__ is not an object")?;
            for (k, v) in m {
                let s = v
                    .as_str()
                    .ok_or_else(|| format!("__metadata__ value for `{k}` is not a string"))?;
                metadata.insert(k.clone(), s.to_string());
            }
            continue;
        }
        let dtype = Dtype::parse(
            v.get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("tensor `{name}`: missing dtype"))?,
        );
        let shape = v
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("tensor `{name}`: missing shape"))?
            .iter()
            .map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| format!("tensor `{name}`: bad shape"))?;
        let offs = v
            .get("data_offsets")
            .and_then(Value::as_array)
            .filter(|a| a.len() == 2)
            .and_then(|a| Some((a[0].as_u64()?, a[1].as_u64()?)))
            .ok_or_else(|| format!("tensor `{name}`: bad data_offsets"))?;
        let (begin, end) = (offs.0 as usize, offs.1 as usize);
        if begin as u64 != offs.0 || end as u64 != offs.1 || begin > end {
            return Err(format!("tensor `{name}`: invalid data_offsets {offs:?}"));
        }
        if end > data_len {
            return Err(format!(
                "tensor `{name}`: data_offsets end {end} is past the data section ({data_len} bytes)"
            ));
        }
        if let Some(size) = dtype.size() {
            let expected = shape
                .iter()
                .try_fold(size, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| format!("tensor `{name}`: shape {shape:?} overflows"))?;
            if expected != end - begin {
                return Err(format!(
                    "tensor `{name}`: shape {shape:?} x {dtype} needs {expected} bytes, \
                     data_offsets span {}",
                    end - begin
                ));
            }
        }
        tensors.push(TensorInfo {
            name: name.clone(),
            dtype,
            shape,
            data_range: data_start + begin..data_start + end,
        });
    }

    tensors.sort_by_key(|t| (t.data_range.start, t.data_range.end));
    // Zero-length tensors may share an offset; everything else must not overlap.
    let mut prev_end = data_start;
    for t in &tensors {
        if t.data_range.start < prev_end {
            return Err(format!("tensor `{}`: data overlaps another tensor", t.name));
        }
        prev_end = prev_end.max(t.data_range.end);
    }
    Ok((metadata, tensors))
}

/// How a checkpoint's weights are quantized, as far as the reader can tell. Dequantization and
/// kernels live elsewhere; this only classifies, so a binder never guesses from tensor shapes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantMethod {
    /// No quantization (BF16/F16/F32 weights).
    None,
    /// FP8 E4M3 weights; `block` is the `[rows, cols]` scale block when the scales are blockwise.
    Fp8 { block: Option<[u32; 2]> },
    /// NVFP4: packed E2M1 with FP8 scales per `group_size` elements (16 in every release seen).
    NvFp4 { group_size: u32 },
    /// OCP MXFP4: packed E2M1 with E8M0 scales per 32 elements.
    MxFp4,
    /// A declared method that is recognised but that Legion does not load (GPTQ, AWQ, ...).
    Other(String),
}

impl fmt::Display for QuantMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantMethod::None => f.write_str("none"),
            QuantMethod::Fp8 { block: None } => f.write_str("fp8"),
            QuantMethod::Fp8 {
                block: Some([r, c]),
            } => write!(f, "fp8 (block {r}x{c})"),
            QuantMethod::NvFp4 { group_size } => write!(f, "nvfp4 (group {group_size})"),
            QuantMethod::MxFp4 => f.write_str("mxfp4"),
            QuantMethod::Other(s) => write!(f, "other ({s})"),
        }
    }
}

/// `quant_method` values that are real but not loadable here. Anything not listed here or
/// handled in [`from_config`] is an error.
const UNSUPPORTED_METHODS: &[&str] = &[
    "gptq",
    "awq",
    "bitsandbytes",
    "hqq",
    "aqlm",
    "eetq",
    "quanto",
    "torchao",
    "quark",
    "fbgemm_fp8",
    "mxfp8",
    "auto-round",
    "gptq_marlin",
    "awq_marlin",
];

type JsonMap = serde_json::Map<String, Value>;

/// Classify from `config.quantization_config`. `Ok(None)` when the config does not declare one.
fn from_config(config: &Value) -> Result<Option<QuantMethod>, String> {
    let q = match config.get("quantization_config") {
        None | Some(Value::Null) => return Ok(None),
        Some(q) => q
            .as_object()
            .ok_or("quantization_config is not an object")?,
    };
    let text = |k: &str| {
        q.get(k)
            .and_then(Value::as_str)
            .map(str::to_ascii_lowercase)
    };
    let method = text("quant_method").ok_or("quantization_config has no quant_method")?;
    let m = match method.as_str() {
        "fp8" => QuantMethod::Fp8 {
            block: block_pair(q.get("weight_block_size"))?,
        },
        "mxfp4" => QuantMethod::MxFp4,
        "modelopt" => {
            let algo =
                text("quant_algo").ok_or("modelopt quantization_config has no quant_algo")?;
            if algo == "fp8" {
                QuantMethod::Fp8 { block: None }
            } else if algo == "mxfp4" {
                QuantMethod::MxFp4
            } else if algo.starts_with("nvfp4") {
                QuantMethod::NvFp4 {
                    group_size: group_size(q)?,
                }
            } else {
                QuantMethod::Other(format!("modelopt:{algo}"))
            }
        }
        "compressed-tensors" => {
            let format = text("format").unwrap_or_default();
            let weights = first_group_weights(q);
            let is_float =
                weights.and_then(|w| w.get("type")).and_then(Value::as_str) == Some("float");
            let bits = weights
                .and_then(|w| w.get("num_bits"))
                .and_then(Value::as_u64);
            if format == "nvfp4-pack-quantized" {
                QuantMethod::NvFp4 {
                    group_size: group_size(q)?,
                }
            } else if format == "float-quantized" && is_float && bits == Some(8) {
                QuantMethod::Fp8 {
                    block: block_structure(weights.and_then(|w| w.get("block_structure")))?,
                }
            } else {
                QuantMethod::Other(format!("compressed-tensors:{format}"))
            }
        }
        m if UNSUPPORTED_METHODS.contains(&m) => QuantMethod::Other(m.to_string()),
        m => return Err(format!("unknown quantization_config.quant_method `{m}`")),
    };
    Ok(Some(m))
}

fn positive_u32(v: &Value) -> Option<u32> {
    v.as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .filter(|&n| n > 0)
}

/// `[rows, cols]` as in Qwen's `weight_block_size`.
fn block_pair(v: Option<&Value>) -> Result<Option<[u32; 2]>, String> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_array()
            .filter(|a| a.len() == 2)
            .and_then(|a| Some([positive_u32(&a[0])?, positive_u32(&a[1])?]))
            .map(Some)
            .ok_or_else(|| "weight_block_size must be two positive integers".to_string()),
    }
}

/// compressed-tensors `block_structure`: a `[r, c]` list or an `"rxc"` string.
fn block_structure(v: Option<&Value>) -> Result<Option<[u32; 2]>, String> {
    if let Some(Value::String(s)) = v {
        let parsed = s
            .split_once('x')
            .and_then(|(r, c)| Some([r.trim().parse().ok()?, c.trim().parse().ok()?]))
            .filter(|b: &[u32; 2]| b.iter().all(|&n| n > 0));
        return parsed
            .map(Some)
            .ok_or_else(|| format!("bad block_structure `{s}`"));
    }
    block_pair(v)
}

fn first_group_weights(q: &JsonMap) -> Option<&Value> {
    q.get("config_groups")?
        .as_object()?
        .values()
        .find_map(|g| g.get("weights"))
}

/// `group_size` of an FP4 config: top level, else the weights of the first config group, else 16.
fn group_size(q: &JsonMap) -> Result<u32, String> {
    let v = q
        .get("group_size")
        .or_else(|| first_group_weights(q).and_then(|w| w.get("group_size")));
    match v {
        None | Some(Value::Null) => Ok(16),
        Some(v) => positive_u32(v).ok_or_else(|| "group_size must be a positive integer".into()),
    }
}

/// Classify from tensor names alone, for checkpoints whose config is silent. Ambiguous names are
/// an error rather than a guess.
fn from_tensor_names(tensors: &[TensorInfo]) -> Result<QuantMethod, String> {
    let by_name: BTreeMap<&str, &TensorInfo> =
        tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let has = |suffix: &str| by_name.keys().any(|n| n.ends_with(suffix));
    let mx = by_name.keys().any(|n| {
        n.strip_suffix("_blocks")
            .is_some_and(|b| by_name.contains_key(format!("{b}_scales").as_str()))
    });
    if mx {
        return Ok(QuantMethod::MxFp4);
    }
    if has(".weight_scale_2") || (has(".weight_packed") && has(".weight_global_scale")) {
        return Ok(QuantMethod::NvFp4 { group_size: 16 });
    }
    if has(".weight_scale_inv") {
        return Ok(QuantMethod::Fp8 { block: None });
    }
    // Per-tensor or per-row scales next to a weight: the weight dtype decides.
    let scaled: Vec<&str> = by_name
        .keys()
        .filter_map(|n| {
            n.strip_suffix(".weight_scale")
                .or_else(|| n.strip_suffix(".input_scale"))
        })
        .collect();
    if scaled.is_empty() {
        return Ok(QuantMethod::None);
    }
    let all_fp8 = scaled.iter().all(|base| {
        by_name
            .get(format!("{base}.weight").as_str())
            .is_some_and(|t| t.dtype == Dtype::F8E4M3)
    });
    if all_fp8 {
        Ok(QuantMethod::Fp8 { block: None })
    } else {
        Err(format!(
            "`{}` has a quantization scale but its weight is not F8_E4M3, and config.json \
             declares no quantization_config",
            scaled[0]
        ))
    }
}

/// Config first, tensor names second. A declared method that is not understood is an error.
/// Errors are plain reasons.
pub fn detect_quant_method(
    config: Option<&Value>,
    tensors: &[TensorInfo],
) -> Result<QuantMethod, String> {
    if let Some(m) = config.map(from_config).transpose()?.flatten() {
        return Ok(m);
    }
    from_tensor_names(tensors)
}

/// A model directory: one or more safetensors shards plus `config.json`.
pub struct Checkpoint {
    dir: PathBuf,
    config: Option<Value>,
    /// Shards in file-name order.
    shards: Vec<SafetensorsFile>,
    /// Tensor name -> shard index.
    tensor_shard: BTreeMap<String, usize>,
    /// `metadata.total_size` from the index, if there was one.
    index_total_size: Option<u64>,
}

impl Checkpoint {
    /// Open `dir`: `model.safetensors.index.json` if present (every shard it names is opened and
    /// checked against `weight_map`), else `model.safetensors`. `config.json` is optional.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, SafetensorsError> {
        let dir = dir.as_ref().to_path_buf();
        let invalid = |path: &Path, reason: String| SafetensorsError::Invalid {
            path: path.to_path_buf(),
            reason,
        };
        let read = |path: &Path| {
            std::fs::read(path).map_err(|source| SafetensorsError::Io {
                path: path.to_path_buf(),
                source,
            })
        };

        let config_path = dir.join("config.json");
        let config = if config_path.exists() {
            let bytes = read(&config_path)?;
            Some(serde_json::from_slice(&bytes).map_err(|e| invalid(&config_path, e.to_string()))?)
        } else {
            None
        };

        let index_path = dir.join(INDEX_FILE);
        let single_path = dir.join(SINGLE_FILE);
        let mut shards = Vec::new();
        let mut tensor_shard = BTreeMap::new();
        let mut index_total_size = None;

        if index_path.exists() {
            let index: Value = serde_json::from_slice(&read(&index_path)?)
                .map_err(|e| invalid(&index_path, e.to_string()))?;
            let weight_map = index
                .get("weight_map")
                .and_then(Value::as_object)
                .ok_or_else(|| invalid(&index_path, "missing weight_map".into()))?;
            index_total_size = index
                .get("metadata")
                .and_then(|m| m.get("total_size"))
                .and_then(Value::as_u64);
            let mut files: Vec<&str> = Vec::new();
            for (tensor, file) in weight_map {
                let file = file.as_str().ok_or_else(|| {
                    invalid(
                        &index_path,
                        format!("weight_map[`{tensor}`] is not a string"),
                    )
                })?;
                if file.contains('/') || file.contains('\\') || file.starts_with('.') {
                    return Err(invalid(
                        &index_path,
                        format!("shard name `{file}` must be a plain file name"),
                    ));
                }
                if !files.contains(&file) {
                    files.push(file);
                }
            }
            files.sort_unstable();
            for file in &files {
                shards.push(SafetensorsFile::open(dir.join(file))?);
            }
            for (tensor, file) in weight_map {
                let idx = files
                    .iter()
                    .position(|f| Some(*f) == file.as_str())
                    .unwrap();
                if shards[idx].tensor(tensor).is_none() {
                    return Err(invalid(
                        &index_path,
                        format!("`{tensor}` is not in {}", files[idx]),
                    ));
                }
                tensor_shard.insert(tensor.clone(), idx);
            }
        } else if single_path.exists() {
            let shard = SafetensorsFile::open(&single_path)?;
            for t in &shard.tensors {
                tensor_shard.insert(t.name.clone(), 0);
            }
            shards.push(shard);
        } else {
            return Err(SafetensorsError::NoCheckpoint(dir));
        }

        Ok(Self {
            dir,
            config,
            shards,
            tensor_shard,
            index_total_size,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config(&self) -> Option<&Value> {
        self.config.as_ref()
    }

    pub fn shards(&self) -> &[SafetensorsFile] {
        &self.shards
    }

    /// [`SafetensorsFile::release_pages`] on every shard.
    pub fn release_pages(&self) {
        for s in &self.shards {
            s.release_pages();
        }
    }

    pub fn index_total_size(&self) -> Option<u64> {
        self.index_total_size
    }

    /// Every tensor in the checkpoint, sorted by name. With an index, tensors in a shard that
    /// the `weight_map` does not list are ignored.
    pub fn tensors(&self) -> Vec<TensorInfo> {
        self.tensor_shard
            .iter()
            .map(|(name, &i)| self.shards[i].tensor(name).unwrap().clone())
            .collect()
    }

    /// The quantization method: `config.json`'s `quantization_config` if present, else a sniff of
    /// the tensor names. An unknown declared method or ambiguous names are errors.
    pub fn quant_method(&self) -> Result<QuantMethod, SafetensorsError> {
        detect_quant_method(self.config.as_ref(), &self.tensors()).map_err(|reason| {
            SafetensorsError::Invalid {
                path: self.dir.join("config.json"),
                reason,
            }
        })
    }

    /// Index into [`Self::shards`] of the shard holding `name`, with its info there.
    pub fn locate(&self, name: &str) -> Option<(usize, &TensorInfo)> {
        let &i = self.tensor_shard.get(name)?;
        Some((i, self.shards[i].tensor(name)?))
    }

    pub fn data(&self, name: &str) -> Result<&[u8], SafetensorsError> {
        let &i = self
            .tensor_shard
            .get(name)
            .ok_or_else(|| SafetensorsError::NoTensor(name.to_string()))?;
        self.shards[i].data(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a safetensors file image: `(name, dtype, shape, bytes)`.
    fn image(tensors: &[(&str, &str, &[usize], Vec<u8>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let begin = data.len();
            data.extend_from_slice(bytes);
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [begin, data.len()]}),
            );
        }
        raw_image(&serde_json::to_vec(&header).unwrap(), &data)
    }

    fn raw_image(header: &[u8], data: &[u8]) -> Vec<u8> {
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header);
        out.extend_from_slice(data);
        out
    }

    fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }

    fn open_err(bytes: &[u8]) -> String {
        let dir = tempfile::tempdir().unwrap();
        let p = write(dir.path(), "x.safetensors", bytes);
        match SafetensorsFile::open(p) {
            Ok(_) => panic!("expected an error"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn reads_tensors_and_data() {
        let dir = tempfile::tempdir().unwrap();
        let a: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let p = write(
            dir.path(),
            "m.safetensors",
            &image(&[
                ("a", "F32", &[2, 3], a.clone()),
                ("scale", "F8_E4M3", &[4], vec![1, 2, 3, 4]),
                ("odd", "F6_E3M2", &[3], vec![9; 3]),
            ]),
        );
        let f = SafetensorsFile::open(&p).unwrap();
        assert_eq!(f.metadata()["format"], "pt");
        let ts = f.tensors();
        assert_eq!(ts.len(), 3);
        assert_eq!(ts[0].name, "a");
        assert_eq!(ts[0].dtype, Dtype::F32);
        assert_eq!(ts[0].shape, vec![2, 3]);
        assert_eq!(ts[0].numel(), 6);
        assert_eq!(f.data("a").unwrap(), &a[..]);
        assert_eq!(f.data("scale").unwrap(), &[1, 2, 3, 4]);
        assert_eq!(
            f.tensor("odd").unwrap().dtype,
            Dtype::Other("F6_E3M2".into())
        );
        assert!(matches!(f.data("nope"), Err(SafetensorsError::NoTensor(_))));
    }

    #[test]
    fn scalar_and_empty_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let p = write(
            dir.path(),
            "m.safetensors",
            &image(&[
                ("s", "F32", &[], vec![0; 4]),
                ("e", "BF16", &[0, 5], vec![]),
            ]),
        );
        let f = SafetensorsFile::open(&p).unwrap();
        assert_eq!(f.data("s").unwrap().len(), 4);
        assert!(f.data("e").unwrap().is_empty());
    }

    #[test]
    fn rejects_corrupt_headers() {
        assert!(open_err(&[1, 2, 3]).contains("shorter"));
        assert!(open_err(&u64::MAX.to_le_bytes()).contains("exceeds"));
        assert!(open_err(&raw_image(b"{\"a\":", b"")[..12]).contains("past the end"));
        assert!(open_err(&raw_image(b"{not json", b"")).contains("not valid JSON"));
        assert!(open_err(&raw_image(b"[1]", b"")).contains("not a JSON object"));
        assert!(open_err(&raw_image(b"{\"a\":{\"dtype\":\"F32\"}}", b"")).contains("shape"));
    }

    #[test]
    fn rejects_bad_offsets() {
        let h = |s: &str| raw_image(s.as_bytes(), &[0; 8]);
        // end past the data section
        let e = open_err(&h(
            r#"{"a":{"dtype":"U8","shape":[16],"data_offsets":[0,16]}}"#,
        ));
        assert!(e.contains("past the data section"), "{e}");
        // begin > end
        let e = open_err(&h(
            r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[4,2]}}"#,
        ));
        assert!(e.contains("invalid data_offsets"), "{e}");
        // overlap
        let e = open_err(&h(
            r#"{"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"U8","shape":[4],"data_offsets":[2,6]}}"#,
        ));
        assert!(e.contains("overlaps"), "{e}");
        // shape x dtype mismatch
        let e = open_err(&h(
            r#"{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#,
        ));
        assert!(e.contains("needs 12 bytes"), "{e}");
        // overflow
        let e = open_err(&h(
            r#"{"a":{"dtype":"F32","shape":[4294967296,4294967296,4294967296],"data_offsets":[0,0]}}"#,
        ));
        assert!(e.contains("overflows"), "{e}");
    }

    #[test]
    fn single_file_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "config.json", br#"{"model_type":"toy"}"#);
        write(
            dir.path(),
            SINGLE_FILE,
            &image(&[("w", "U8", &[3], vec![7, 8, 9])]),
        );
        let c = Checkpoint::open(dir.path()).unwrap();
        assert_eq!(c.config().unwrap()["model_type"], "toy");
        assert_eq!(c.shards().len(), 1);
        assert_eq!(c.data("w").unwrap(), &[7, 8, 9]);
        assert!(Checkpoint::open(tempfile::tempdir().unwrap().path()).is_err());
    }

    #[test]
    fn sharded_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &image(&[("l0.weight", "U8", &[2], vec![1, 2])]),
        );
        write(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &image(&[
                ("l1.weight", "U8", &[2], vec![3, 4]),
                ("l1.weight_scale", "F8_E4M3", &[1], vec![5]),
            ]),
        );
        let index = serde_json::json!({
            "metadata": {"total_size": 5},
            "weight_map": {
                "l0.weight": "model-00001-of-00002.safetensors",
                "l1.weight": "model-00002-of-00002.safetensors",
                "l1.weight_scale": "model-00002-of-00002.safetensors",
            }
        });
        write(dir.path(), INDEX_FILE, index.to_string().as_bytes());
        let c = Checkpoint::open(dir.path()).unwrap();
        assert!(c.config().is_none());
        assert_eq!(c.shards().len(), 2);
        assert_eq!(c.index_total_size(), Some(5));
        let names: Vec<_> = c.tensors().into_iter().map(|t| t.name).collect();
        assert_eq!(names, ["l0.weight", "l1.weight", "l1.weight_scale"]);
        assert_eq!(c.data("l1.weight").unwrap(), &[3, 4]);
        assert_eq!(c.data("l1.weight_scale").unwrap(), &[5]);
    }

    #[test]
    fn index_errors() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "a.safetensors",
            &image(&[("x", "U8", &[1], vec![1])]),
        );
        let bad = |map: serde_json::Value| {
            write(
                dir.path(),
                INDEX_FILE,
                serde_json::json!({ "weight_map": map })
                    .to_string()
                    .as_bytes(),
            );
            Checkpoint::open(dir.path()).err().unwrap().to_string()
        };
        assert!(bad(serde_json::json!({"y": "a.safetensors"})).contains("not in a.safetensors"));
        assert!(
            bad(serde_json::json!({"x": "missing.safetensors"})).contains("missing.safetensors")
        );
        assert!(bad(serde_json::json!({"x": "../a.safetensors"})).contains("plain file name"));
    }

    // ---- quantization detection -------------------------------------------------------------
    // The config shapes below are trimmed from public releases: openai/gpt-oss-20b (mxfp4),
    // Qwen/Qwen3-8B-FP8 (fp8, 128x128 blocks), nvidia/Qwen3-8B-NVFP4 and nvidia/Qwen3-8B-FP8
    // (ModelOpt), RedHatAI/Qwen3-8B-NVFP4 (compressed-tensors).

    fn ti(name: &str, dtype: Dtype) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dtype,
            shape: vec![1],
            data_range: 0..0,
        }
    }

    fn detect(
        config: Option<serde_json::Value>,
        tensors: &[TensorInfo],
    ) -> Result<QuantMethod, String> {
        detect_quant_method(config.as_ref(), tensors)
    }

    fn qc(q: serde_json::Value) -> Option<serde_json::Value> {
        Some(serde_json::json!({"model_type": "x", "quantization_config": q}))
    }

    #[test]
    fn dtype_f8e8m0() {
        assert_eq!(Dtype::parse("F8_E8M0"), Dtype::F8E8M0);
        assert_eq!(Dtype::F8E8M0.name(), "F8_E8M0");
        assert_eq!(Dtype::F8E8M0.size(), Some(1));
        // I8 and U8 stay raw bytes with a known size.
        assert_eq!(Dtype::parse("I8"), Dtype::I8);
        assert_eq!(Dtype::parse("U8").size(), Some(1));
        // The size check now applies to E8M0 tensors.
        let ok = image(&[("s", "F8_E8M0", &[4], vec![127; 4])]);
        assert!(parse_header(&ok).is_ok());
        let bad = raw_image(
            br#"{"s":{"dtype":"F8_E8M0","shape":[4],"data_offsets":[0,3]}}"#,
            &[0; 3],
        );
        assert!(parse_header(&bad).unwrap_err().contains("needs 4 bytes"));
    }

    #[test]
    fn config_absent_or_null_is_not_declared() {
        assert_eq!(detect(None, &[]), Ok(QuantMethod::None));
        let c = Some(serde_json::json!({"quantization_config": null, "model_type": "x"}));
        assert_eq!(
            detect(c, &[ti("a.weight", Dtype::BF16)]),
            Ok(QuantMethod::None)
        );
    }

    #[test]
    fn qwen_fp8_blockwise() {
        let c = qc(serde_json::json!({
            "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
            "weight_block_size": [128, 128]
        }));
        let m = detect(c, &[]).unwrap();
        assert_eq!(
            m,
            QuantMethod::Fp8 {
                block: Some([128, 128])
            }
        );
        assert_eq!(m.to_string(), "fp8 (block 128x128)");
        // No block size: per-tensor or per-row scales.
        let c = qc(serde_json::json!({"quant_method": "fp8", "activation_scheme": "static"}));
        assert_eq!(detect(c, &[]), Ok(QuantMethod::Fp8 { block: None }));
        let c = qc(serde_json::json!({"quant_method": "fp8", "weight_block_size": [128]}));
        assert!(detect(c, &[]).unwrap_err().contains("weight_block_size"));
    }

    #[test]
    fn modelopt_nvfp4() {
        let c = qc(serde_json::json!({
            "config_groups": {"group_0": {
                "input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                "targets": ["Linear"]
            }},
            "ignore": ["lm_head"], "quant_algo": "NVFP4",
            "kv_cache_scheme": {"dynamic": false, "num_bits": 8, "type": "float"},
            "producer": {"name": "modelopt", "version": "0.35.0"}, "quant_method": "modelopt"
        }));
        assert_eq!(detect(c, &[]), Ok(QuantMethod::NvFp4 { group_size: 16 }));
        // The declared group size wins over the default.
        let c = qc(serde_json::json!({
            "quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 32
        }));
        assert_eq!(detect(c, &[]), Ok(QuantMethod::NvFp4 { group_size: 32 }));
        // A config that says NVFP4 beats tensor names that look like plain BF16.
        let c = qc(serde_json::json!({"quant_method": "modelopt", "quant_algo": "NVFP4"}));
        assert_eq!(
            detect(c, &[ti("m.layers.0.q_proj.weight", Dtype::BF16)]),
            Ok(QuantMethod::NvFp4 { group_size: 16 })
        );
    }

    #[test]
    fn modelopt_fp8_and_other_algos() {
        let c = qc(serde_json::json!({
            "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "float"}}},
            "ignore": ["lm_head"], "quant_algo": "FP8", "quant_method": "modelopt"
        }));
        assert_eq!(detect(c, &[]), Ok(QuantMethod::Fp8 { block: None }));
        let c = qc(serde_json::json!({"quant_method": "modelopt", "quant_algo": "W4A8_AWQ"}));
        assert_eq!(
            detect(c, &[]),
            Ok(QuantMethod::Other("modelopt:w4a8_awq".into()))
        );
        let c = qc(serde_json::json!({"quant_method": "modelopt"}));
        assert!(detect(c, &[]).unwrap_err().contains("quant_algo"));
    }

    #[test]
    fn compressed_tensors_variants() {
        let c = qc(serde_json::json!({
            "config_groups": {"group_0": {
                "format": "nvfp4-pack-quantized",
                "weights": {"group_size": 16, "num_bits": 4, "strategy": "tensor_group", "type": "float"},
                "targets": ["Linear"]
            }},
            "format": "nvfp4-pack-quantized", "ignore": ["lm_head"],
            "quant_method": "compressed-tensors", "quantization_status": "compressed"
        }));
        assert_eq!(detect(c, &[]), Ok(QuantMethod::NvFp4 { group_size: 16 }));

        let c = qc(serde_json::json!({
            "config_groups": {"group_0": {
                "weights": {"num_bits": 8, "type": "float", "strategy": "block",
                            "block_structure": [128, 128]},
                "targets": ["Linear"]
            }},
            "format": "float-quantized", "quant_method": "compressed-tensors"
        }));
        assert_eq!(
            detect(c, &[]),
            Ok(QuantMethod::Fp8 {
                block: Some([128, 128])
            })
        );

        let c = qc(serde_json::json!({
            "config_groups": {"group_0": {
                "weights": {"num_bits": 8, "type": "float", "block_structure": "128x128"}
            }},
            "format": "float-quantized", "quant_method": "compressed-tensors"
        }));
        assert_eq!(
            detect(c, &[]),
            Ok(QuantMethod::Fp8 {
                block: Some([128, 128])
            })
        );

        let c = qc(serde_json::json!({
            "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "int"}}},
            "format": "pack-quantized", "quant_method": "compressed-tensors"
        }));
        assert_eq!(
            detect(c, &[]),
            Ok(QuantMethod::Other(
                "compressed-tensors:pack-quantized".into()
            ))
        );
    }

    #[test]
    fn gpt_oss_mxfp4() {
        let c = qc(serde_json::json!({
            "modules_to_not_convert": [
                "model.layers.*.self_attn", "model.layers.*.mlp.router",
                "model.embed_tokens", "lm_head"
            ],
            "quant_method": "mxfp4"
        }));
        let m = detect(c, &[]).unwrap();
        assert_eq!(m, QuantMethod::MxFp4);
        assert_eq!(m.to_string(), "mxfp4");
    }

    #[test]
    fn recognised_but_unsupported_method() {
        let c = qc(serde_json::json!({"quant_method": "gptq", "bits": 4, "group_size": 128}));
        assert_eq!(detect(c, &[]), Ok(QuantMethod::Other("gptq".into())));
    }

    #[test]
    fn unknown_method_is_an_error() {
        let c = qc(serde_json::json!({"quant_method": "frobnicate"}));
        let e = detect(c, &[ti("a.weight", Dtype::BF16)]).unwrap_err();
        assert!(e.contains("frobnicate"), "{e}");
        let c = qc(serde_json::json!({"bits": 4}));
        assert!(detect(c, &[]).unwrap_err().contains("quant_method"));
        let c = qc(serde_json::json!("fp8"));
        assert!(detect(c, &[]).unwrap_err().contains("not an object"));
    }

    #[test]
    fn sniff_names() {
        let nvfp4 = [
            ti("l.q_proj.weight", Dtype::U8),
            ti("l.q_proj.weight_scale", Dtype::F8E4M3),
            ti("l.q_proj.weight_scale_2", Dtype::F32),
            ti("l.q_proj.input_scale", Dtype::F32),
        ];
        assert_eq!(
            detect(None, &nvfp4),
            Ok(QuantMethod::NvFp4 { group_size: 16 })
        );
        // A config without quantization_config falls through to the names too.
        let plain = Some(serde_json::json!({"model_type": "x"}));
        assert_eq!(
            detect(plain, &nvfp4),
            Ok(QuantMethod::NvFp4 { group_size: 16 })
        );

        let ct = [
            ti("l.q.weight_packed", Dtype::U8),
            ti("l.q.weight_scale", Dtype::F8E4M3),
            ti("l.q.weight_global_scale", Dtype::F32),
        ];
        assert_eq!(detect(None, &ct), Ok(QuantMethod::NvFp4 { group_size: 16 }));

        let mx = [
            ti("m.l.mlp.experts.gate_up_proj_blocks", Dtype::U8),
            ti("m.l.mlp.experts.gate_up_proj_scales", Dtype::U8),
            ti("m.l.mlp.router.weight", Dtype::BF16),
        ];
        assert_eq!(detect(None, &mx), Ok(QuantMethod::MxFp4));
        // `_blocks` without its `_scales` is not MXFP4.
        assert_eq!(detect(None, &mx[..1]), Ok(QuantMethod::None));

        let block_fp8 = [
            ti("l.q.weight", Dtype::F8E4M3),
            ti("l.q.weight_scale_inv", Dtype::BF16),
        ];
        assert_eq!(
            detect(None, &block_fp8),
            Ok(QuantMethod::Fp8 { block: None })
        );

        let tensor_fp8 = [
            ti("l.q.weight", Dtype::F8E4M3),
            ti("l.q.weight_scale", Dtype::F32),
        ];
        assert_eq!(
            detect(None, &tensor_fp8),
            Ok(QuantMethod::Fp8 { block: None })
        );

        assert_eq!(
            detect(None, &[ti("l.q.weight", Dtype::BF16)]),
            Ok(QuantMethod::None)
        );
    }

    #[test]
    fn sniff_ambiguous_scale_is_an_error() {
        // A scale next to a non-FP8 weight with no config: do not guess.
        let t = [
            ti("l.q.weight", Dtype::U8),
            ti("l.q.weight_scale", Dtype::F32),
        ];
        let e = detect(None, &t).unwrap_err();
        assert!(e.contains("l.q"), "{e}");
    }

    #[test]
    fn checkpoint_quant_method() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            SINGLE_FILE,
            &image(&[
                ("l.weight", "U8", &[2], vec![1, 2]),
                ("l.weight_scale", "F8_E4M3", &[1], vec![5]),
                ("l.weight_scale_2", "F32", &[1], vec![0; 4]),
            ]),
        );
        // No config.json: names.
        let c = Checkpoint::open(dir.path()).unwrap();
        assert_eq!(
            c.quant_method().unwrap(),
            QuantMethod::NvFp4 { group_size: 16 }
        );
        // Config wins.
        write(
            dir.path(),
            "config.json",
            br#"{"quantization_config":{"quant_method":"fp8"}}"#,
        );
        let c = Checkpoint::open(dir.path()).unwrap();
        assert_eq!(c.quant_method().unwrap(), QuantMethod::Fp8 { block: None });
        // Unknown declared method surfaces as an error naming config.json.
        write(
            dir.path(),
            "config.json",
            br#"{"quantization_config":{"quant_method":"nope"}}"#,
        );
        let e = Checkpoint::open(dir.path())
            .unwrap()
            .quant_method()
            .unwrap_err();
        assert!(e.to_string().contains("config.json") && e.to_string().contains("nope"));
    }
}
