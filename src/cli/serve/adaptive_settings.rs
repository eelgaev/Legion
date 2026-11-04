//! CLI controls for load-adaptive speculative decoding.

#[derive(clap::Args)]
pub(super) struct AdaptiveSettings {
    /// Use fixed speculative depth, disabling acceptance adaptation and the batch gate.
    #[arg(long)]
    pub spec_fixed: bool,
    /// Disable speculative drafts at this many decoding sequences or more.
    #[arg(long, default_value_t = 8)]
    pub spec_max_batch: usize,
    /// Whether --spec-max-batch also stops the drafters whose depth follows measured step costs
    /// (`--spec mtp` and block drafters such as DFlash 2; src/spec/depth.rs). `off`: they keep
    /// drafting at any load and their cost model picks the depth (0 when drafts do not pay);
    /// n-gram drafters (also --spec-mtp-ngram) and other model drafters keep the gate. `on`
    /// gates everyone. Not given: `on`, except for `--spec mtp` (without --spec-fixed) on a
    /// V100 layer split of 2 GPUs with the expert cache and --spec-max-batch at most 8, where
    /// the gate stops MTP drafts only up to 8 sequences (steps of up to 8 rows run as
    /// matrix-vector products, which drafts would leave) and the cost model drafts past that:
    /// Qwen3.8-Flash-Next decoded 12 users 71 -> 83 tok/s, 16 users 80 -> 89, 8 users as gated
    /// (ungated: 98 -> 93). Try `off` with MTP at 8 or more parallel users on an integrated GPU: one GB10
    /// decoded Qwen3.8-Flash-Next +27% at 8 and +37% at 16 users (4x V100: no gain). Also with
    /// DFlash 2 at 8 or more users on an integrated GPU: one GB10 decoded 16 users of
    /// Qwen3.8-27B NVFP4 at 229 tok/s instead of 125. Greedy text where drafting changes
    /// moves, as with any depth change.
    #[arg(long, value_name = "on|off", action = clap::ArgAction::Set,
        value_parser = clap::builder::BoolishValueParser::new())]
    pub spec_gate_cost_model: Option<bool>,
    /// `--spec mtp` without --spec-fixed: charge each further draft of the MTP head, next to
    /// its draft pass, what its row in the verify step costs more than a padding row, measured
    /// by fitting each verify shape's step time to the rows that carried drafts. On a MoE a
    /// drafted row reads experts a padding row does not, so late, rarely accepted drafts stop.
    /// Default `on`; `off` charges the pass alone (the policy before #836 part 27). Greedy text
    /// changes with it, as with any depth change; --deterministic and --spec-fixed ignore it.
    #[arg(long, value_name = "on|off", default_value = "on", action = clap::ArgAction::Set,
        value_parser = clap::builder::BoolishValueParser::new())]
    pub spec_row_cost: bool,
    /// Speculative decoding: pad a verify step's sequences only to the step's longest draft
    /// instead of 1 + --spec-k (with --spec-fixed or --deterministic) or 1 + the depth policy's
    /// depth. When every draft stopped early (--spec-p-min), the step runs fewer rows of
    /// attention, DeltaNet, dense matmuls and the output head; each row count is a verify graph
    /// of its own (kept, see --verify-kept-graphs). Default `off`. Try `on` with MTP on an
    /// integrated GPU such as the GB10.
    #[arg(long, value_name = "on|off", default_value = "off", action = clap::ArgAction::Set,
        value_parser = clap::builder::BoolishValueParser::new())]
    pub spec_pad_longest: bool,
    /// `--spec mtp` and block drafters without --spec-fixed: the most rows a verify step takes
    /// (each drafting sequence's token and its drafts). The depth policy drafts less at large
    /// batches to stay within it, and nothing once a row per sequence fills it. 0 (default): no
    /// cap. Try 63 on V100s with --spec-gate-cost-model off: from 64 rows ggml runs quantized
    /// matmuls and the output head as cuBLAS F16 GEMMs over F16 copies of the weights (2.6 GiB
    /// per GPU for Gemma 4's head, taken from the cache). Greedy text moves as with any depth
    /// change; --deterministic and --spec-fixed ignore it.
    #[arg(long, default_value_t = 0, value_name = "ROWS")]
    pub spec_max_rows: usize,
    /// Minimum expected acceptance probability of the last draft, in (0, 1) (drafters other
    /// than `--spec mtp`, whose depth follows measured costs).
    #[arg(long, default_value_t = 0.5, value_parser = probability)]
    pub spec_p_floor: f32,
}

fn probability(value: &str) -> Result<f32, String> {
    let value = value.parse::<f32>().map_err(|e| e.to_string())?;
    if value > 0.0 && value < 1.0 {
        Ok(value)
    } else {
        Err("speculative probability floor must be in (0, 1)".into())
    }
}

#[cfg(test)]
mod tests;
