//! `--registry-api`: registers the server with an LLM proxy (#622). The proxy is
//! simple-llm-proxy's dynamic `POST /register` route: the body names the address and port the
//! proxy reaches this server on and the key it sends as `Authorization: Bearer <key>`; the
//! proxy then checks `GET /health` and `GET /models` before it answers 201.
//!
//! Registration runs in a background task per proxy, started once the listener is bound (the
//! model is loaded by then), so the proxy's checks pass. It never fails or blocks the server:
//! errors are logged and retried with a doubling backoff, and after a success the task
//! registers again every `--registry-interval`, so a proxy that lost its host list picks the
//! server up again.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use reqwest::Url;
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;

/// Per-request timeout of a registration. The proxy checks `/health` and `/models` over https
/// and then http with 5 s each before it answers.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// `--registry-api` and friends. The default (no proxies) is off.
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// The proxies' `/register` URLs (`--registry-api`, repeatable, see [`register_url`]).
    /// Empty: no registration.
    pub proxies: Vec<Url>,
    /// The address to advertise (`--registry-host`); `None` picks one (see [`fixed_host`]).
    pub host: Option<IpAddr>,
    /// Time between registrations after a success (`--registry-interval`).
    pub interval: Duration,
    /// Retry delay after the first failure, doubled per further failure.
    pub backoff_min: Duration,
    /// The longest retry delay.
    pub backoff_max: Duration,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            proxies: Vec::new(),
            host: None,
            interval: Duration::from_secs(300),
            backoff_min: Duration::from_secs(1),
            backoff_max: Duration::from_secs(60),
        }
    }
}

/// The JSON body of `POST /register`.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterBody {
    /// An IP address (the proxy refuses host names).
    pub host: String,
    pub port: u16,
    /// The key the proxy sends to this server; empty when the server has none.
    pub api_key: String,
}

impl std::fmt::Debug for RegisterBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegisterBody")
            .field("host", &self.host)
            .field("port", &self.port)
            .field(
                "api_key",
                &if self.api_key.is_empty() { "" } else { "<set>" },
            )
            .finish()
    }
}

/// The proxy's answer to a successful registration (`201 {"status","url","models"}`).
#[derive(Debug, Default, Deserialize)]
struct Registered {
    #[serde(default)]
    url: String,
    #[serde(default)]
    models: u64,
}

/// Parses a `--registry-api` value into the proxy's `/register` URL: `http(s)://proxy[:port]`
/// with or without a path prefix and with or without a trailing `/register`.
pub fn register_url(value: &str) -> Result<Url, String> {
    let mut url = Url::parse(value).map_err(|e| format!("not a URL: {e}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(format!("scheme `{}` is not http or https", url.scheme()));
    }
    if url.host().is_none() {
        return Err("the URL has no host".into());
    }
    let path = url.path().trim_end_matches('/');
    let path = if path.ends_with("/register") {
        path.to_string()
    } else {
        format!("{path}/register")
    };
    url.set_path(&path);
    url.set_query(None);
    url.set_fragment(None);
    Ok(url)
}

/// The key the proxy is given: the first `--api-key`, or empty when the server has none.
pub fn api_key(keys: &[String]) -> String {
    keys.first().cloned().unwrap_or_default()
}

/// The address to advertise when no route lookup is needed: `--registry-host`, else the bound
/// address when it is a specific one. `None` for a wildcard bind (`0.0.0.0`, `::`).
pub fn fixed_host(explicit: Option<IpAddr>, bound: IpAddr) -> Option<IpAddr> {
    explicit.or_else(|| (!bound.is_unspecified()).then_some(bound))
}

/// The local address of the route to the proxy: a UDP socket connected to it (which sends
/// nothing) reports the source address the kernel picks.
pub async fn route_local_ip(proxy: &Url) -> std::io::Result<IpAddr> {
    let port = proxy.port_or_known_default().unwrap_or(80);
    let name = proxy
        .host_str()
        .ok_or_else(|| std::io::Error::other("the proxy URL has no host"))?;
    // An IPv6 literal comes bracketed.
    let literal = name.trim_start_matches('[').trim_end_matches(']');
    let target: SocketAddr = match literal.parse::<IpAddr>() {
        Ok(ip) => (ip, port).into(),
        Err(_) => tokio::net::lookup_host((name, port))
            .await?
            .next()
            .ok_or_else(|| std::io::Error::other(format!("{name} has no address")))?,
    };
    let any: SocketAddr = if target.is_ipv4() {
        (Ipv4Addr::UNSPECIFIED, 0).into()
    } else {
        (Ipv6Addr::UNSPECIFIED, 0).into()
    };
    let socket = std::net::UdpSocket::bind(any)?;
    socket.connect(target)?;
    let ip = socket.local_addr()?.ip();
    if ip.is_unspecified() {
        return Err(std::io::Error::other("no local address routes to it"));
    }
    Ok(ip)
}

/// Starts one registration task per proxy for a server bound to `bound`. Nothing with no
/// proxies. The tasks run until the process exits.
pub fn start(config: &super::ServerConfig, bound: SocketAddr) -> Vec<JoinHandle<()>> {
    let registry = &config.registry;
    if registry.proxies.is_empty() {
        return Vec::new();
    }
    let client = match reqwest::Client::builder().timeout(REQUEST_TIMEOUT).build() {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("--registry-api: cannot build the HTTP client: {e}");
            return Vec::new();
        }
    };
    let key = api_key(&config.api_keys);
    registry
        .proxies
        .iter()
        .map(|proxy| {
            let task = Task {
                client: client.clone(),
                proxy: proxy.clone(),
                bound,
                api_key: key.clone(),
                config: registry.clone(),
            };
            tokio::spawn(task.run())
        })
        .collect()
}

struct Task {
    client: reqwest::Client,
    proxy: Url,
    bound: SocketAddr,
    api_key: String,
    config: RegistryConfig,
}

impl Task {
    async fn run(self) {
        let mut delay = self.config.backoff_min;
        // Log a success at info the first time and after a failure; routine re-registrations
        // at debug.
        let mut announced = false;
        loop {
            match self.register().await {
                Ok(r) => {
                    if announced {
                        tracing::debug!(proxy = %self.proxy, url = %r.url, models = r.models, "registry: registered again");
                    } else {
                        tracing::info!(proxy = %self.proxy, url = %r.url, models = r.models, "registry: registered with the proxy");
                        announced = true;
                    }
                    delay = self.config.backoff_min;
                    tokio::time::sleep(self.config.interval).await;
                }
                Err(e) => {
                    tracing::warn!(proxy = %self.proxy, retry_in = ?delay, "registry: {e}");
                    announced = false;
                    tokio::time::sleep(delay).await;
                    delay = (delay * 2).min(self.config.backoff_max);
                }
            }
        }
    }

    /// One `POST /register`; the error text says what failed (and the proxy's own message).
    async fn register(&self) -> Result<Registered, String> {
        let host = match fixed_host(self.config.host, self.bound.ip()) {
            Some(ip) => ip,
            None => route_local_ip(&self.proxy).await.map_err(|e| {
                format!("cannot find an address to advertise ({e}); pass --registry-host <IP>")
            })?,
        };
        let body = RegisterBody {
            host: host.to_string(),
            port: self.bound.port(),
            api_key: self.api_key.clone(),
        };
        let resp = self
            .client
            .post(self.proxy.clone())
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(serde_json::to_vec(&body).expect("the body always serializes"))
            .send()
            .await
            .map_err(|e| format!("cannot reach the proxy: {}", error_chain(&e)))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(format!(
                "the proxy refused {host}:{}: {status}: {}",
                body.port,
                text.trim()
            ));
        }
        Ok(serde_json::from_str(&text).unwrap_or_default())
    }
}

/// An error with its sources, e.g. "error sending request: connection refused".
fn error_chain(e: &dyn std::error::Error) -> String {
    let mut s = e.to_string();
    let mut source = e.source();
    while let Some(inner) = source {
        s.push_str(": ");
        s.push_str(&inner.to_string());
        source = inner.source();
    }
    s
}

#[cfg(test)]
mod tests;
