//! Legion: an inference engine for NVIDIA gaming PCs, DGX Spark (and Spark clusters) and the
//! IBM AC922. See `docs/architecture.md` for the module map.

pub mod api;
pub mod bench;
pub mod chat_template;
pub mod cpu;
pub mod dashboard;
pub mod engine;
pub mod ggml;
pub mod gguf;
pub mod grammar;
pub mod hf;
pub mod hw;
pub mod kv;
pub mod load_io;
pub mod model;
pub mod node;
pub mod ptx;
pub mod residency;
pub mod safetensors;
pub mod server;
pub mod spec;
pub mod tokenizer;
pub mod tp_net;
pub mod vision;
pub mod weights;
