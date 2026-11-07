//! `legion top`: the live dashboard of a running `legion serve`, from its `/v1/stats/stream`
//! and `/v1/logs/stream` (#1198). `q` quits the client only; the server is not touched.

use std::io::IsTerminal;
use std::sync::Arc;

use clap::Args;
use legion::dashboard::client::{self, DEFAULT_URL};
use legion::dashboard::term::{self, Feed};
use legion::dashboard::ui::{App, Conn};

#[derive(Args)]
pub struct TopArgs {
    /// The server's base URL. https works; add --insecure for a self-signed certificate.
    #[arg(long, default_value = DEFAULT_URL)]
    url: String,
    /// The server's API key (`legion serve --api-key`), sent as `Authorization: Bearer`.
    #[arg(long, env = "LEGION_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
    /// Accept any TLS certificate (a self-signed one).
    #[arg(long)]
    insecure: bool,
    /// How often the server sends a snapshot, in milliseconds (100-5000).
    #[arg(long, default_value_t = 250, value_parser = clap::value_parser!(u64).range(100..=5000))]
    interval_ms: u64,
}

pub async fn run(args: TopArgs) -> anyhow::Result<()> {
    if !std::io::stdout().is_terminal() {
        anyhow::bail!("legion top draws on a terminal; stdout is not one");
    }
    let url = client::base_url(&args.url);
    let feed = Arc::new(Feed::default());
    client::spawn(
        client::Options {
            url: url.clone(),
            api_key: args.api_key.filter(|k| !k.is_empty()),
            insecure: args.insecure,
            interval_ms: args.interval_ms,
        },
        feed.clone(),
    )
    .map_err(|e| anyhow::anyhow!("cannot build the HTTP client: {e}"))?;
    // `q` and Ctrl-C both quit the client.
    tokio::task::spawn_blocking(move || {
        let mut app = App::new(url, Conn::Connecting);
        term::run(&mut app, |app| app.ingest(feed.take()))
    })
    .await??;
    Ok(())
}
