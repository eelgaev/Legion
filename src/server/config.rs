//! Server settings. Each server feature adds its fields here, with a default that keeps the
//! feature off (or today's behaviour), and its CLI flags in `src/cli/serve.rs`.

use std::path::PathBuf;

/// Files used to serve HTTPS. The certificate file may contain the full certificate chain;
/// the private key may be PKCS#8 or RSA.
#[derive(Debug, Clone)]
pub struct TlsConfig {
    pub cert: PathBuf,
    pub key: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// TLS certificate chain and private key (`--tls-cert` and `--tls-key`). `None` keeps the
    /// listener on plain HTTP.
    pub tls: Option<TlsConfig>,
    /// Used when an OpenAI chat request sets neither `max_tokens` nor `max_completion_tokens`.
    pub default_max_tokens: u32,
    /// Accepted API keys (`--api-key`, `LEGION_API_KEY`); a request may present any one of
    /// them. Empty means auth is off and every request is served without a key.
    pub api_keys: Vec<String>,
    /// Allowed CORS origins (`--cors`, repeatable); `*` allows any origin. Empty means no
    /// CORS at all: no layer, no headers, `OPTIONS` handled by the router as before.
    pub cors_origins: Vec<String>,
    /// Serve the built-in web UI at `/` (`--no-webui` turns it off).
    pub webui: bool,
    /// Maximum request body size in megabytes (`--max-request-mb`). Bodies over the limit are
    /// refused with 413 before being buffered in full.
    pub max_request_mb: usize,
    /// Seconds to let in-flight requests finish after SIGINT/SIGTERM before exiting anyway
    /// (`--shutdown-timeout-secs`).
    pub shutdown_timeout_secs: u64,
    /// Names the model is served under (`--served-model-name`, repeatable). When non-empty,
    /// `/v1/models` lists these instead of the engine id, each resolves to the engine's model,
    /// and responses echo the requested name. Empty means the engine id is used as is.
    pub served_model_names: Vec<String>,
    /// Whose defaults fill the sampling parameters a request omits (`--sampling-defaults`).
    pub sampling_mode: crate::engine::sampling_defaults::DefaultsMode,
    /// The operator's per-parameter defaults (`--default-temperature`, ...), over the model's.
    pub sampling_overrides: crate::engine::sampling_defaults::SamplingLayer,
    /// The GPUs' hardware counters, exported in `/metrics` (`--hw-counters`, #391).
    pub hw_counters: Option<std::sync::Arc<crate::hw::counters::HwCounters>>,
    /// LLM proxies to register with (`--registry-api`, #622); no proxies is off.
    pub registry: crate::server::registry::RegistryConfig,
    /// What identifies a client for the engine queue's fair share (`--fair-share`, #625).
    pub fair_share: crate::server::fair_share::FairShare,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            tls: None,
            default_max_tokens: 4096,
            api_keys: Vec::new(),
            cors_origins: Vec::new(),
            webui: true,
            max_request_mb: 64,
            shutdown_timeout_secs: 30,
            served_model_names: Vec::new(),
            sampling_mode: Default::default(),
            sampling_overrides: Default::default(),
            hw_counters: None,
            registry: Default::default(),
            fair_share: Default::default(),
        }
    }
}
