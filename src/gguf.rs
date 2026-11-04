//! GGUF files through ggml's own `gguf_*` reader and writer.
//!
//! [`GgufFile`] reads metadata and tensor infos without touching tensor data (the data is
//! mapped or streamed later by the model loader). [`GgufWriter`] builds small files, mainly for
//! tests and tooling.

use std::ffi::{CStr, CString};
use std::fmt;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use crate::ggml::{self, ffi};

mod companion;
mod split;
mod synth;

pub use companion::companion_path;

pub use synth::{Compute, FileRange, Nvfp4Source, Nvfp4Stack, Recipe};

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("{0}: not a readable GGUF file (ggml printed the reason to stderr)")]
    Open(PathBuf),
    #[error("{path}: invalid split GGUF: {reason}")]
    Split { path: PathBuf, reason: String },
    #[error("{0}: failed to write GGUF file")]
    Write(PathBuf),
    #[error("string contains a NUL byte: {0:?}")]
    Nul(String),
    #[error("tensor `{name}`: expected {expected} bytes of data, got {got}")]
    TensorSize {
        name: String,
        expected: usize,
        got: usize,
    },
    #[error("unsupported metadata value for `{0}`: {1}")]
    Unsupported(String, &'static str),
    #[error("{path}: {reason}")]
    Checkpoint { path: PathBuf, reason: String },
    #[error("{path}: invalid companion GGUF: {reason}")]
    Companion { path: PathBuf, reason: String },
}

/// A metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) => u64::try_from(v).ok(),
            Value::I16(v) => u64::try_from(v).ok(),
            Value::I32(v) => u64::try_from(v).ok(),
            Value::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Value::F32(v) => Some(v.into()),
            Value::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(v) => Some(v),
            _ => None,
        }
    }

    /// GGUF type name, e.g. `u32`, `string`, `array[string]`.
    pub fn type_name(&self) -> String {
        match self {
            Value::U8(_) => "u8".into(),
            Value::I8(_) => "i8".into(),
            Value::U16(_) => "u16".into(),
            Value::I16(_) => "i16".into(),
            Value::U32(_) => "u32".into(),
            Value::I32(_) => "i32".into(),
            Value::U64(_) => "u64".into(),
            Value::I64(_) => "i64".into(),
            Value::F32(_) => "f32".into(),
            Value::F64(_) => "f64".into(),
            Value::Bool(_) => "bool".into(),
            Value::String(_) => "string".into(),
            Value::Array(v) => match v.first() {
                Some(first) => format!("array[{}]", first.type_name()),
                None => "array".into(),
            },
        }
    }
}

/// Short form: strings in quotes (cut at 80 chars), arrays longer than 8 elided.
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::U8(v) => write!(f, "{v}"),
            Value::I8(v) => write!(f, "{v}"),
            Value::U16(v) => write!(f, "{v}"),
            Value::I16(v) => write!(f, "{v}"),
            Value::U32(v) => write!(f, "{v}"),
            Value::I32(v) => write!(f, "{v}"),
            Value::U64(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}"),
            Value::F32(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::String(s) if s.chars().count() > 80 => {
                let head: String = s.chars().take(80).collect();
                write!(f, "{head:?}... ({} bytes)", s.len())
            }
            Value::String(s) => write!(f, "{s:?}"),
            Value::Array(v) => {
                write!(f, "[")?;
                for (i, x) in v.iter().take(8).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{x}")?;
                }
                if v.len() > 8 {
                    write!(f, ", ... {} total", v.len())?;
                }
                write!(f, "]")
            }
        }
    }
}

/// Plain JSON: strings as strings, arrays as arrays, numbers as numbers.
impl serde::Serialize for Value {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        match self {
            Value::U8(v) => s.serialize_u8(*v),
            Value::I8(v) => s.serialize_i8(*v),
            Value::U16(v) => s.serialize_u16(*v),
            Value::I16(v) => s.serialize_i16(*v),
            Value::U32(v) => s.serialize_u32(*v),
            Value::I32(v) => s.serialize_i32(*v),
            Value::U64(v) => s.serialize_u64(*v),
            Value::I64(v) => s.serialize_i64(*v),
            Value::F32(v) => s.serialize_f32(*v),
            Value::F64(v) => s.serialize_f64(*v),
            Value::Bool(v) => s.serialize_bool(*v),
            Value::String(v) => s.serialize_str(v),
            Value::Array(v) => {
                let mut seq = s.serialize_seq(Some(v.len()))?;
                for x in v {
                    seq.serialize_element(x)?;
                }
                seq.end()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub ggml_type: ffi::ggml_type,
    /// ggml order: `shape[0]` is the innermost (contiguous) dimension. Trailing 1s are trimmed.
    pub shape: Vec<u64>,
    /// Index into [`GgufFile::shard_paths`], in shard order (zero for a single file).
    pub file_index: usize,
    /// Absolute byte offset of the tensor data in the file.
    pub offset: u64,
    pub size: u64,
}

impl TensorInfo {
    pub fn type_name(&self) -> &'static str {
        ggml::type_name(self.ggml_type)
    }

    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// An open GGUF model (metadata only), including all sibling files of a split model.
/// Metadata and header getters refer to the first shard; tensor getters cover all shards.
pub struct GgufFile {
    ctx: NonNull<ffi::gguf_context>,
    path: PathBuf,
    siblings: Vec<GgufFile>,
    /// Tensors of a synthesized model ([`GgufFile::synthetic`]); `ctx` then holds metadata only.
    synth: Option<Box<synth::Synthetic>>,
}

// SAFETY: the context is never mutated after loading; ggml's getters are read-only.
unsafe impl Send for GgufFile {}
unsafe impl Sync for GgufFile {}

impl Drop for GgufFile {
    fn drop(&mut self) {
        // SAFETY: ctx came from gguf_init_from_file and is freed once.
        unsafe { ffi::gguf_free(self.ctx.as_ptr()) }
    }
}

fn cstring(s: &str) -> Result<CString, GgufError> {
    CString::new(s).map_err(|_| GgufError::Nul(s.to_string()))
}

/// # Safety
/// `p` must be null or a valid NUL-terminated string.
unsafe fn owned(p: *const std::os::raw::c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

impl GgufFile {
    /// Opens a GGUF file (any shard of a split model) with the companion files its metadata
    /// names next to it (DeepSeek-V4.1-Flash's Engram tables, [`companion_path`]), or a Hugging
    /// Face safetensors checkpoint directory, which is synthesized as the GGUF llama.cpp's
    /// converter would write ([`crate::hf`]).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        Self::open_with(path, &crate::hf::OpenOptions::default())
    }

    /// [`Self::open`], with options for safetensors checkpoints.
    pub fn open_with(
        path: impl AsRef<Path>,
        hf: &crate::hf::OpenOptions,
    ) -> Result<Self, GgufError> {
        let path = path.as_ref();
        if path.is_dir() {
            return crate::hf::open(path, hf);
        }
        companion::attach(split::open(path)?)
    }

    fn open_single(path: &Path) -> Result<Self, GgufError> {
        let path = path.to_path_buf();
        let c = cstring(&path.to_string_lossy())?;
        let params = ffi::gguf_init_params {
            no_alloc: true,
            ctx: std::ptr::null_mut(),
        };
        // SAFETY: valid C string; no_alloc with a null ctx only reads metadata.
        let ctx = unsafe { ffi::gguf_init_from_file(c.as_ptr(), params) };
        let ctx = NonNull::new(ctx).ok_or_else(|| GgufError::Open(path.clone()))?;
        Ok(Self {
            ctx,
            path,
            siblings: Vec::new(),
            synth: None,
        })
    }

    /// Path of the first shard (or the original path for a single-file model).
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn raw(&self) -> *const ffi::gguf_context {
        self.ctx.as_ptr()
    }

    pub fn version(&self) -> u32 {
        unsafe { ffi::gguf_get_version(self.raw()) }
    }

    pub fn alignment(&self) -> usize {
        unsafe { ffi::gguf_get_alignment(self.raw()) }
    }

    /// Byte offset where tensor data starts.
    pub fn data_offset(&self) -> u64 {
        unsafe { ffi::gguf_get_data_offset(self.raw()) as u64 }
    }

    pub fn kv_count(&self) -> usize {
        unsafe { ffi::gguf_get_n_kv(self.raw()) as usize }
    }

    pub fn keys(&self) -> Vec<String> {
        (0..self.kv_count() as i64)
            .map(|i| unsafe { owned(ffi::gguf_get_key(self.raw(), i)) })
            .collect()
    }

    pub fn get(&self, key: &str) -> Option<Value> {
        let c = CString::new(key).ok()?;
        let id = unsafe { ffi::gguf_find_key(self.raw(), c.as_ptr()) };
        (id >= 0).then(|| self.value(id)).flatten()
    }

    pub fn get_str(&self, key: &str) -> Option<String> {
        match self.get(key)? {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64()
    }

    /// `general.architecture`, e.g. `qwen3next`.
    pub fn architecture(&self) -> Option<String> {
        self.get_str("general.architecture")
    }

    /// All metadata in file order. Values of unsupported types (nested arrays) are skipped.
    pub fn metadata(&self) -> Vec<(String, Value)> {
        (0..self.kv_count() as i64)
            .filter_map(|i| {
                let key = unsafe { owned(ffi::gguf_get_key(self.raw(), i)) };
                self.value(i).map(|v| (key, v))
            })
            .collect()
    }

    fn value(&self, id: i64) -> Option<Value> {
        let ctx = self.raw();
        // SAFETY: id is a valid key index; each getter matches the key's type.
        unsafe {
            let ty = ffi::gguf_get_kv_type(ctx, id);
            if ty == ffi::gguf_type_GGUF_TYPE_ARRAY {
                return self.array(id);
            }
            Some(match ty {
                ffi::gguf_type_GGUF_TYPE_UINT8 => Value::U8(ffi::gguf_get_val_u8(ctx, id)),
                ffi::gguf_type_GGUF_TYPE_INT8 => Value::I8(ffi::gguf_get_val_i8(ctx, id)),
                ffi::gguf_type_GGUF_TYPE_UINT16 => Value::U16(ffi::gguf_get_val_u16(ctx, id)),
                ffi::gguf_type_GGUF_TYPE_INT16 => Value::I16(ffi::gguf_get_val_i16(ctx, id)),
                ffi::gguf_type_GGUF_TYPE_UINT32 => Value::U32(ffi::gguf_get_val_u32(ctx, id)),
                ffi::gguf_type_GGUF_TYPE_INT32 => Value::I32(ffi::gguf_get_val_i32(ctx, id)),
                ffi::gguf_type_GGUF_TYPE_UINT64 => Value::U64(ffi::gguf_get_val_u64(ctx, id)),
                ffi::gguf_type_GGUF_TYPE_INT64 => Value::I64(ffi::gguf_get_val_i64(ctx, id)),
                ffi::gguf_type_GGUF_TYPE_FLOAT32 => Value::F32(ffi::gguf_get_val_f32(ctx, id)),
                ffi::gguf_type_GGUF_TYPE_FLOAT64 => Value::F64(ffi::gguf_get_val_f64(ctx, id)),
                ffi::gguf_type_GGUF_TYPE_BOOL => Value::Bool(ffi::gguf_get_val_bool(ctx, id)),
                ffi::gguf_type_GGUF_TYPE_STRING => {
                    Value::String(owned(ffi::gguf_get_val_str(ctx, id)))
                }
                _ => return None,
            })
        }
    }

    fn array(&self, id: i64) -> Option<Value> {
        let ctx = self.raw();
        // SAFETY: id is an array key; reads stay within gguf_get_arr_n elements of its type.
        unsafe {
            let ty = ffi::gguf_get_arr_type(ctx, id);
            let n = ffi::gguf_get_arr_n(ctx, id);
            if ty == ffi::gguf_type_GGUF_TYPE_STRING {
                let v = (0..n)
                    .map(|i| Value::String(owned(ffi::gguf_get_arr_str(ctx, id, i))))
                    .collect();
                return Some(Value::Array(v));
            }
            let data = ffi::gguf_get_arr_data(ctx, id);
            fn read<T: Copy>(
                data: *const std::ffi::c_void,
                n: usize,
                f: fn(T) -> Value,
            ) -> Vec<Value> {
                let p = data as *const T;
                (0..n)
                    .map(|i| f(unsafe { p.add(i).read_unaligned() }))
                    .collect()
            }
            let v = match ty {
                ffi::gguf_type_GGUF_TYPE_UINT8 => read(data, n, Value::U8),
                ffi::gguf_type_GGUF_TYPE_INT8 => read(data, n, Value::I8),
                ffi::gguf_type_GGUF_TYPE_UINT16 => read(data, n, Value::U16),
                ffi::gguf_type_GGUF_TYPE_INT16 => read(data, n, Value::I16),
                ffi::gguf_type_GGUF_TYPE_UINT32 => read(data, n, Value::U32),
                ffi::gguf_type_GGUF_TYPE_INT32 => read(data, n, Value::I32),
                ffi::gguf_type_GGUF_TYPE_UINT64 => read(data, n, Value::U64),
                ffi::gguf_type_GGUF_TYPE_INT64 => read(data, n, Value::I64),
                ffi::gguf_type_GGUF_TYPE_FLOAT32 => read(data, n, Value::F32),
                ffi::gguf_type_GGUF_TYPE_FLOAT64 => read(data, n, Value::F64),
                ffi::gguf_type_GGUF_TYPE_BOOL => read::<u8>(data, n, |b| Value::Bool(b != 0)),
                _ => return None,
            };
            Some(Value::Array(v))
        }
    }

    pub fn tensor_count(&self) -> usize {
        if let Some(s) = &self.synth {
            return s.tensors.len();
        }
        self.shards().map(Self::local_tensor_count).sum()
    }

    fn local_tensor_count(&self) -> usize {
        unsafe { ffi::gguf_get_n_tensors(self.raw()) as usize }
    }

    pub fn tensors(&self) -> Vec<TensorInfo> {
        if let Some(s) = &self.synth {
            return s.tensors.clone();
        }
        self.shards()
            .enumerate()
            .flat_map(|(file_index, shard)| {
                (0..shard.local_tensor_count() as i64).map(move |i| shard.tensor_at(i, file_index))
            })
            .collect()
    }

    pub fn tensor(&self, name: &str) -> Option<TensorInfo> {
        if let Some(s) = &self.synth {
            return s.by_name.get(name).map(|&i| s.tensors[i].clone());
        }
        let c = CString::new(name).ok()?;
        self.shards().enumerate().find_map(|(file_index, shard)| {
            let id = unsafe { ffi::gguf_find_tensor(shard.raw(), c.as_ptr()) };
            (id >= 0).then(|| shard.tensor_at(id, file_index))
        })
    }

    fn tensor_at(&self, id: i64, file_index: usize) -> TensorInfo {
        let ctx = self.raw();
        // SAFETY: id < n_tensors; ne points at GGML_MAX_DIMS elements.
        unsafe {
            let ne = std::slice::from_raw_parts(
                ffi::gguf_get_tensor_ne(ctx, id),
                ffi::GGML_MAX_DIMS as usize,
            );
            let mut shape: Vec<u64> = ne.iter().map(|&d| d as u64).collect();
            while shape.len() > 1 && shape.last() == Some(&1) {
                shape.pop();
            }
            TensorInfo {
                name: owned(ffi::gguf_get_tensor_name(ctx, id)),
                ggml_type: ffi::gguf_get_tensor_type(ctx, id),
                shape,
                file_index,
                offset: self.data_offset() + ffi::gguf_get_tensor_offset(ctx, id) as u64,
                size: ffi::gguf_get_tensor_size(ctx, id) as u64,
            }
        }
    }

    /// Sum of all tensor sizes in bytes.
    pub fn tensor_bytes(&self) -> u64 {
        self.tensors().iter().map(|t| t.size).sum()
    }
}

/// Builds a GGUF file in memory and writes it out.
pub struct GgufWriter {
    ctx: NonNull<ffi::gguf_context>,
    /// One ggml context per tensor, holding only its descriptor.
    tensor_ctxs: Vec<NonNull<ffi::ggml_context>>,
    /// Tensor data, kept alive until the file is written.
    data: Vec<Box<[u8]>>,
}

impl Drop for GgufWriter {
    fn drop(&mut self) {
        unsafe {
            ffi::gguf_free(self.ctx.as_ptr());
            for c in &self.tensor_ctxs {
                ffi::ggml_free(c.as_ptr());
            }
        }
    }
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufWriter {
    pub fn new() -> Self {
        let ctx = unsafe { ffi::gguf_init_empty() };
        Self {
            ctx: NonNull::new(ctx).expect("gguf_init_empty returned null"),
            tensor_ctxs: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn set(&mut self, key: &str, value: &Value) -> Result<&mut Self, GgufError> {
        let k = cstring(key)?;
        let ctx = self.ctx.as_ptr();
        // SAFETY: valid context and C strings; gguf copies keys and values.
        unsafe {
            match value {
                Value::U8(v) => ffi::gguf_set_val_u8(ctx, k.as_ptr(), *v),
                Value::I8(v) => ffi::gguf_set_val_i8(ctx, k.as_ptr(), *v),
                Value::U16(v) => ffi::gguf_set_val_u16(ctx, k.as_ptr(), *v),
                Value::I16(v) => ffi::gguf_set_val_i16(ctx, k.as_ptr(), *v),
                Value::U32(v) => ffi::gguf_set_val_u32(ctx, k.as_ptr(), *v),
                Value::I32(v) => ffi::gguf_set_val_i32(ctx, k.as_ptr(), *v),
                Value::U64(v) => ffi::gguf_set_val_u64(ctx, k.as_ptr(), *v),
                Value::I64(v) => ffi::gguf_set_val_i64(ctx, k.as_ptr(), *v),
                Value::F32(v) => ffi::gguf_set_val_f32(ctx, k.as_ptr(), *v),
                Value::F64(v) => ffi::gguf_set_val_f64(ctx, k.as_ptr(), *v),
                Value::Bool(v) => ffi::gguf_set_val_bool(ctx, k.as_ptr(), *v),
                Value::String(s) => ffi::gguf_set_val_str(ctx, k.as_ptr(), cstring(s)?.as_ptr()),
                Value::Array(items) => self.set_array(&k, key, items)?,
            }
        }
        Ok(self)
    }

    unsafe fn set_array(&mut self, k: &CStr, key: &str, items: &[Value]) -> Result<(), GgufError> {
        let ctx = self.ctx.as_ptr();
        if items.iter().all(|v| matches!(v, Value::String(_))) && !items.is_empty() {
            let owned: Vec<CString> = items
                .iter()
                .map(|v| cstring(v.as_str().unwrap_or_default()))
                .collect::<Result<_, _>>()?;
            let mut ptrs: Vec<*const std::os::raw::c_char> =
                owned.iter().map(|s| s.as_ptr()).collect();
            unsafe { ffi::gguf_set_arr_str(ctx, k.as_ptr(), ptrs.as_mut_ptr(), ptrs.len()) };
            return Ok(());
        }
        macro_rules! raw {
            ($variant:ident, $t:ty, $gt:ident) => {{
                let v: Option<Vec<$t>> = items
                    .iter()
                    .map(|x| match x {
                        Value::$variant(y) => Some(*y as $t),
                        _ => None,
                    })
                    .collect();
                v.map(|v| {
                    (
                        ffi::$gt,
                        v.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<u8>>(),
                        v.len(),
                    )
                })
            }};
        }
        let encoded = match items.first() {
            None => Some((ffi::gguf_type_GGUF_TYPE_INT32, Vec::new(), 0)),
            Some(Value::U8(_)) => raw!(U8, u8, gguf_type_GGUF_TYPE_UINT8),
            Some(Value::I8(_)) => raw!(I8, i8, gguf_type_GGUF_TYPE_INT8),
            Some(Value::U16(_)) => raw!(U16, u16, gguf_type_GGUF_TYPE_UINT16),
            Some(Value::I16(_)) => raw!(I16, i16, gguf_type_GGUF_TYPE_INT16),
            Some(Value::U32(_)) => raw!(U32, u32, gguf_type_GGUF_TYPE_UINT32),
            Some(Value::I32(_)) => raw!(I32, i32, gguf_type_GGUF_TYPE_INT32),
            Some(Value::U64(_)) => raw!(U64, u64, gguf_type_GGUF_TYPE_UINT64),
            Some(Value::I64(_)) => raw!(I64, i64, gguf_type_GGUF_TYPE_INT64),
            Some(Value::F32(_)) => raw!(F32, f32, gguf_type_GGUF_TYPE_FLOAT32),
            Some(Value::F64(_)) => raw!(F64, f64, gguf_type_GGUF_TYPE_FLOAT64),
            Some(Value::Bool(_)) => {
                let v: Option<Vec<u8>> = items.iter().map(|x| x.as_bool().map(u8::from)).collect();
                v.map(|v| (ffi::gguf_type_GGUF_TYPE_BOOL, v.clone(), v.len()))
            }
            _ => None,
        };
        let Some((ty, bytes, n)) = encoded else {
            return Err(GgufError::Unsupported(key.into(), "mixed or nested array"));
        };
        unsafe { ffi::gguf_set_arr_data(ctx, k.as_ptr(), ty, bytes.as_ptr().cast(), n) };
        Ok(())
    }

    /// Adds a tensor. `shape` is in ggml order (innermost first); `data` must be exactly the
    /// tensor's size in bytes for `ty`.
    pub fn add_tensor(
        &mut self,
        name: &str,
        ty: ffi::ggml_type,
        shape: &[i64],
        data: &[u8],
    ) -> Result<&mut Self, GgufError> {
        assert!(!shape.is_empty() && shape.len() <= ffi::GGML_MAX_DIMS as usize);
        let cname = cstring(name)?;
        // SAFETY: a fresh no_alloc context sized for exactly one tensor descriptor.
        unsafe {
            let params = ffi::ggml_init_params {
                mem_size: ffi::ggml_tensor_overhead(),
                mem_buffer: std::ptr::null_mut(),
                no_alloc: true,
            };
            let tctx = NonNull::new(ffi::ggml_init(params)).expect("ggml_init failed");
            self.tensor_ctxs.push(tctx);
            let t = ffi::ggml_new_tensor(tctx.as_ptr(), ty, shape.len() as i32, shape.as_ptr());
            ffi::ggml_set_name(t, cname.as_ptr());
            let expected = ffi::ggml_nbytes(t);
            if expected != data.len() {
                return Err(GgufError::TensorSize {
                    name: name.into(),
                    expected,
                    got: data.len(),
                });
            }
            ffi::gguf_add_tensor(self.ctx.as_ptr(), t);
            let owned: Box<[u8]> = data.into();
            ffi::gguf_set_tensor_data(self.ctx.as_ptr(), cname.as_ptr(), owned.as_ptr().cast());
            self.data.push(owned);
        }
        Ok(self)
    }

    /// Gives up the context as metadata for [`GgufFile::synthetic`]. The writer must hold no
    /// tensors.
    fn into_metadata(self) -> NonNull<ffi::gguf_context> {
        assert!(
            self.tensor_ctxs.is_empty(),
            "a metadata-only writer holds no tensors"
        );
        let this = std::mem::ManuallyDrop::new(self);
        this.ctx
    }

    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), GgufError> {
        let path = path.as_ref();
        let c = cstring(&path.to_string_lossy())?;
        // SAFETY: all tensor data is alive in self.data.
        let ok = unsafe { ffi::gguf_write_to_file(self.ctx.as_ptr(), c.as_ptr(), false) };
        if ok {
            Ok(())
        } else {
            Err(GgufError::Write(path.to_path_buf()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_serializes_as_plain_json() {
        assert_eq!(serde_json::to_string(&Value::U8(255)).unwrap(), "255");
        assert_eq!(serde_json::to_string(&Value::I8(-1)).unwrap(), "-1");
        assert_eq!(serde_json::to_string(&Value::U32(7)).unwrap(), "7");
        assert_eq!(serde_json::to_string(&Value::I64(-2)).unwrap(), "-2");
        assert_eq!(serde_json::to_string(&Value::F32(1.5)).unwrap(), "1.5");
        assert_eq!(serde_json::to_string(&Value::F64(0.25)).unwrap(), "0.25");
        assert_eq!(serde_json::to_string(&Value::Bool(true)).unwrap(), "true");
        assert_eq!(
            serde_json::to_string(&Value::String("é\n".into())).unwrap(),
            "\"é\\n\""
        );
        assert_eq!(serde_json::to_string(&Value::Array(vec![])).unwrap(), "[]");
        assert_eq!(
            serde_json::to_string(&Value::Array(vec![Value::U8(1), Value::U8(2)])).unwrap(),
            "[1,2]"
        );
        assert_eq!(
            serde_json::to_string(&Value::Array(vec![
                Value::String("a".into()),
                Value::String("b".into()),
            ]))
            .unwrap(),
            "[\"a\",\"b\"]"
        );
    }
}
