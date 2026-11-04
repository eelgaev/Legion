//! `legion comm-bench`: the multi-GPU communication micro-benchmarks (a development tool,
//! #668), so a build tree's release binary can measure the box it runs on. Choose the GPUs with
//! `CUDA_VISIBLE_DEVICES` and the CPU side with `numactl`.

use clap::{Args, Subcommand};
use legion::hw::comm_bench::{self, Chain, Shape};

#[derive(Args)]
pub struct CommBenchArgs {
    #[command(subcommand)]
    what: What,
}

#[derive(Subcommand)]
enum What {
    /// One all-reduce through GLM's parallel-decode exchange (`--glm-parallel-decode`) over all
    /// visible GPUs: microseconds per reduce of `[elements, tokens]` F32, timed as graphs of
    /// `--chain` reduces in a row (a GLM token makes about 90), next to a graph of as many adds
    /// (a node's cost without communication).
    Exchange {
        /// F32 values per token, comma separated (an FP16 vector of 2x as many has the same
        /// bytes).
        #[arg(long, value_delimiter = ',', default_values_t = [2048, 3072, 4096, 6144, 8192, 12288])]
        elements: Vec<i64>,
        /// Tokens per reduce, comma separated.
        #[arg(long, value_delimiter = ',', default_values_t = [1])]
        tokens: Vec<i64>,
        /// Reduces per graph.
        #[arg(long, default_value_t = 90)]
        chain: usize,
        /// Untimed graph runs first.
        #[arg(long, default_value_t = 10)]
        warmup: usize,
        /// Timed graph runs.
        #[arg(long, default_value_t = 100)]
        iters: usize,
        /// Per GPU, the NUMA node of its exchange inbox, comma separated (default: the GPU's own
        /// node, as the engine places it).
        #[arg(long, value_delimiter = ',')]
        inbox_nodes: Vec<u32>,
        /// Which GPUs write straight into another GPU's VRAM inbox, as `serve
        /// --glm-exchange-peer`: pairs (default; the GPUs of one NUMA node), all, off.
        #[arg(long, value_enum, default_value_t = legion::model::glm5next::par::exchange::PeerWrites::Pairs)]
        peer: legion::model::glm5next::par::exchange::PeerWrites,
        /// Bytes of a VRAM inbox slot: bigger payloads take the host inboxes (default: as big as
        /// the biggest payload; serve uses a 512-token reduce's, 8 MiB for GLM).
        #[arg(long, value_name = "BYTES")]
        peer_slot: Option<usize>,
        /// Reduces of this many bytes or more go as a reduce-scatter and an all-gather (#686;
        /// default: as serve).
        #[arg(long, value_name = "BYTES", default_value_t = legion::model::glm5next::par::exchange::SCATTER_MIN)]
        scatter_min: usize,
        /// Run on this many CPU threads instead of the GPUs (checks the plumbing).
        #[arg(long)]
        cpu: Option<usize>,
    },
    /// The `cudaMemcpyPeer` baseline between every ordered pair of visible GPUs, and a round
    /// in which every GPU sends to every other one at once.
    Peer {
        /// Bytes per copy, comma separated.
        #[arg(long, value_delimiter = ',', default_values_t = [8192, 12288, 16384, 24576, 32768, 49152])]
        bytes: Vec<usize>,
        /// Timed copies (and rounds) per point.
        #[arg(long, default_value_t = 200)]
        iters: usize,
    },
    /// Each visible GPU reading host memory of each NUMA node: random 3 MiB expert-sized rows
    /// read in place by a kernel, and DMA copies of them, in GB/s.
    HostRead {
        /// NUMA nodes, comma separated.
        #[arg(long, value_delimiter = ',', default_values_t = [0, 8])]
        nodes: Vec<u32>,
        /// Measurements per GPU and node.
        #[arg(long, default_value_t = 3)]
        reps: usize,
    },
}

/// Set before CUDA starts, while the process has one thread: what `serve --glm-parallel-decode`
/// sets for its exchange kernels (eager module loading, no ggml CUDA graphs), so the reduce is
/// timed as the engine runs it. An explicit setting wins.
pub fn prepare_env() {
    // SAFETY (both): called from `main` before the runtime starts: no other thread reads the
    // environment.
    if std::env::var_os("CUDA_MODULE_LOADING").is_none() {
        unsafe { std::env::set_var("CUDA_MODULE_LOADING", "EAGER") };
    }
    if std::env::var_os("GGML_CUDA_DISABLE_GRAPHS").is_none() {
        unsafe { std::env::set_var("GGML_CUDA_DISABLE_GRAPHS", "1") };
    }
}

pub fn run(args: CommBenchArgs) -> anyhow::Result<()> {
    match args.what {
        What::Exchange {
            elements,
            tokens,
            chain,
            warmup,
            iters,
            inbox_nodes,
            peer,
            peer_slot,
            scatter_min,
            cpu,
        } => {
            let devs = devices(cpu, inbox_nodes, peer, peer_slot, scatter_min)?;
            let shapes: Vec<Shape> = tokens
                .iter()
                .flat_map(|&n| elements.iter().map(move |&e| Shape { e, n }))
                .collect();
            let r = comm_bench::exchange_bench(
                &devs,
                &shapes,
                &[Chain::Reduce, Chain::Add],
                chain,
                warmup,
                iters,
            )
            .map_err(anyhow::Error::msg)?;
            println!("kind\telements\ttokens\tbytes\tchain\tp10_us\tp50_us\tp90_us\tgraph_p50_us");
            for x in &r {
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.1}",
                    match x.chain {
                        Chain::Reduce => "reduce",
                        Chain::Add => "add",
                    },
                    x.shape.e,
                    x.shape.n,
                    x.bytes(),
                    x.len,
                    x.per_node_us(0.1),
                    x.per_node_us(0.5),
                    x.per_node_us(0.9),
                    comm_bench::quantile(&x.runs_us, 0.5),
                );
            }
        }
        What::Peer { bytes, iters } => peer(&bytes, iters)?,
        What::HostRead { nodes, reps } => host_read(&nodes, reps)?,
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_count() -> anyhow::Result<usize> {
    let n = legion::ggml::devices()
        .into_iter()
        .filter(|d| d.name().starts_with("CUDA"))
        .count();
    if n == 0 {
        anyhow::bail!("comm-bench: no CUDA devices");
    }
    Ok(n)
}

#[cfg(feature = "cuda")]
fn devices(
    cpu: Option<usize>,
    inbox_nodes: Vec<u32>,
    peers: legion::model::glm5next::par::exchange::PeerWrites,
    peer_slot: Option<usize>,
    scatter_min: usize,
) -> anyhow::Result<comm_bench::Devices> {
    if let Some(n) = cpu {
        return Ok(comm_bench::Devices::Cpu(n));
    }
    let n = cuda_count()?;
    let nodes = if inbox_nodes.is_empty() {
        (0..n)
            .map(|d| legion::residency::cuda::device_numa_node(d).unwrap_or(0))
            .collect()
    } else {
        if inbox_nodes.len() != n {
            anyhow::bail!("--inbox-nodes: one node per visible GPU ({n})");
        }
        inbox_nodes
    };
    println!(
        "# {n} GPUs, exchange inboxes on nodes {nodes:?}, peer writes {peers:?}, peer slot \
         {peer_slot:?}, scattered reduces from {scatter_min} bytes"
    );
    Ok(comm_bench::Devices::Cuda {
        nodes,
        peers,
        peer_slot,
        scatter_min,
    })
}

#[cfg(not(feature = "cuda"))]
fn devices(
    cpu: Option<usize>,
    _inbox_nodes: Vec<u32>,
    _peers: legion::model::glm5next::par::exchange::PeerWrites,
    _peer_slot: Option<usize>,
    _scatter_min: usize,
) -> anyhow::Result<comm_bench::Devices> {
    match cpu {
        Some(n) => Ok(comm_bench::Devices::Cpu(n)),
        None => anyhow::bail!("comm-bench exchange: GPUs need a CUDA build (or --cpu N)"),
    }
}

#[cfg(feature = "cuda")]
fn peer(bytes: &[usize], iters: usize) -> anyhow::Result<()> {
    let n = cuda_count()?;
    let (pairs, rounds) = comm_bench::peer_bench(n, bytes, iters).map_err(anyhow::Error::msg)?;
    println!("src\tdst\tbytes\tdirect\tasync_us\tsync_p50_us");
    for p in &pairs {
        println!(
            "{}\t{}\t{}\t{}\t{:.2}\t{:.2}",
            p.src, p.dst, p.bytes, p.direct, p.async_us, p.sync_us
        );
    }
    println!("round_gpus\tbytes\tp10_us\tp50_us\tp90_us");
    for r in &rounds {
        println!(
            "{n}\t{}\t{:.2}\t{:.2}\t{:.2}",
            r.bytes,
            comm_bench::quantile(&r.runs_us, 0.1),
            comm_bench::quantile(&r.runs_us, 0.5),
            comm_bench::quantile(&r.runs_us, 0.9),
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn host_read(nodes: &[u32], reps: usize) -> anyhow::Result<()> {
    let n = cuda_count()?;
    println!("gpu\tnode\trep\tin_place_gbs\tcopy_gbs");
    for (d, node, r, p) in comm_bench::host_read_bench(n, nodes, reps) {
        match p {
            Some((in_place, copy)) => println!("{d}\t{node}\t{r}\t{in_place:.1}\t{copy:.1}"),
            None => println!("{d}\t{node}\t{r}\tfailed\tfailed"),
        }
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn peer(_bytes: &[usize], _iters: usize) -> anyhow::Result<()> {
    anyhow::bail!("comm-bench peer needs a CUDA build")
}

#[cfg(not(feature = "cuda"))]
fn host_read(_nodes: &[u32], _reps: usize) -> anyhow::Result<()> {
    anyhow::bail!("comm-bench host-read needs a CUDA build")
}
