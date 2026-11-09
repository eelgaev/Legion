//! Weight matmuls run the way llama.cpp's `build_lora_mm` / `build_lora_mm_id` run them
//! (`src/llama-graph.cpp`, MIT): per-tensor scales and precision hints for NVFP4 weights.
//!
//! llama.cpp's converter (`conversion/base.py`, `_repack_nvfp4`) turns a ModelOpt or
//! compressed-tensors NVFP4 layer into three GGUF tensors:
//! - `<name>.weight`: `GGML_TYPE_NVFP4` blocks (the E2M1 codes and E4M3 per-16 scales),
//! - `<name>.scale`: the F32 per-tensor global scale (`weight_scale_2`), `[1]`, or `[n_expert]`
//!   for merged experts; left out when it is 1,
//! - `<name>.input_scale`: the activation scale, which llama.cpp loads but never applies.
//!
//! The matmul result is multiplied by `.scale` (per selected expert for `mul_mat_id`), right
//! after the matmul and before any bias, activation or residual. NVFP4 matmuls may accumulate in
//! BF16, and checkpoints quantized as `W4A16_NVFP4` are listed in `general.tensor_extra.prec_a4`
//! as `false`: their activations must not be quantized to 4 bits, which CUDA's Blackwell kernels
//! otherwise do by default. The CPU backend ignores both hints (it quantizes activations to
//! Q8_0 and accumulates in F32), so CPU results do not depend on them.

use std::collections::HashSet;

use super::ModelError;
use crate::ggml::{Context, Tensor, ffi, types};
use crate::gguf::GgufFile;

const KEY_NAMES: &str = "general.tensor_extra.name";
use crate::hf::nvfp4::KEY_FP8_E4M3;
const KEY_PREC_A4: &str = "general.tensor_extra.prec_a4";

/// The activation precision of NVFP4 matmuls (`--nvfp4-act`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Nvfp4Act {
    /// Activations stay at 8 bits or wider for every NVFP4 weight: closest to the W4A16
    /// checkpoints NVIDIA ships (measured on Nemotron 3.5 30B-A3B against BF16 activations:
    /// KL 2.1e-3, about the CPU's Q8_0 noise).
    #[default]
    A8,
    /// What llama.cpp does: weights not listed in `prec_a4` (the converter leaves out stacked
    /// experts) take 4-bit activations in CUDA's batched matmuls on Blackwell. Faster prefill,
    /// ~9x the KL on Nemotron 3.5 (2.0e-2, top-1 95% vs 98%).
    A4,
}

/// Which kernels run NVFP4 matmuls on CUDA (`--nvfp4-mm`), for weights that keep 8-bit or
/// wider activations (see [`Nvfp4Act`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Nvfp4Mm {
    /// Legion's W4A16 kernels (BF16 activations, exact weights, F32 sums) for batches of more
    /// than [`NVFP4_GGML_MAX_N`] tokens: prefill, decode steps of 9+ sequences, and routed
    /// experts grouped by expert. ggml's MMVQ (Q8_1 activations) below that.
    #[default]
    Auto,
    /// Legion's W4A16 kernels at every batch size: one numeric mode for a whole request, at
    /// some cost in decode steps of up to 8 sequences.
    Legion,
    /// ggml's kernels everywhere (MMVQ / MMQ with Q8_1 activations).
    Ggml,
}

/// Which NVFP4 matmuls take 4-bit activations on Legion's W4A4 kernels (`--nvfp4-a4`,
/// [`crate::model::nvfp4_a4`]): among those [`Nvfp4Mm`] gives to Legion's kernels, on GPUs
/// whose build has the block-scaled FP4 tensor-core kernels (sm_120a / sm_121a: GB10, RTX 50).
/// Everywhere else the W4A16 kernels run. W4A16 checkpoints are not calibrated for 4-bit
/// activations: every value but `off` moves the logits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Nvfp4A4 {
    /// W4A16 everywhere.
    #[default]
    Off,
    /// Routed experts' grouped matmuls of batches of more than [`NVFP4_A4_MIN_N`] tokens.
    Experts,
    /// Every NVFP4 matmul of a batch of more than [`NVFP4_A4_MIN_N`] tokens (prefill): dense,
    /// fused SwiGLU and routed experts.
    Prefill,
    /// Every NVFP4 matmul on Legion's kernels, decode steps included.
    All,
    /// Routed experts' grouped matmuls of batches of more than [`NVFP4_GGML_MAX_N`] tokens on
    /// Legion's kernels: prefill as `experts`, and decode steps of 9+ tokens (9+ sequences, or
    /// verify steps) on the W4A4 decode kernel. Dense NVFP4 matmuls stay W4A16.
    ExpertsDecode,
}

/// The activations of FP8 matmuls (`--fp8-act`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Fp8Act {
    /// W8A8 as the checkpoint declares it: activations quantized to E4M3 with its static
    /// per-tensor `input_scale` (vLLM's mode), on E4M3 tensor cores for batches.
    #[default]
    A8,
    /// W8A16: activations stay F32 (GEMV) or BF16 (tensor-core GEMM).
    A16,
    /// W8A8 in batched matmuls (prefill, more than 8 tokens), W8A16 below (decode).
    #[value(skip)]
    PrefillA8,
    /// As `a8`, and weights without a static `input_scale` (unsloth's FP8 layers) take W8A8
    /// with dynamic per-token activation scales (`amax / 448`, NInfer's scheme) in batches of
    /// more than [`NVFP4_A4_MIN_N`] tokens (prefill chunks) on E4M3 tensor cores (sm_89+);
    /// decode steps stay W8A16.
    A8Dynamic,
}

/// How an FP8 matmul takes its activations ([`Linear::fp8_route`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fp8Route {
    /// W8A8 with the checkpoint's static `input_scale`.
    Static,
    /// W8A8 with per-token scales ([`Fp8Act::A8Dynamic`]).
    Dynamic,
    /// W8A16.
    A16,
}

/// What a dense FFN may fuse into its kernels ([`Linear::swiglu_ffn_fused`]); the default
/// fuses nothing.
#[derive(Clone, Copy, Default)]
pub struct FfnFusion<'a> {
    /// The residual added to the FFN's output (`add(ffn, res)`).
    pub res: Option<Tensor<'a>>,
    /// `(x, w, eps)` when the FFN's input is `mul(rms_norm(x, eps), w)`: the norm then rides
    /// in the W4A4 activations' quantization.
    pub norm: Option<(Tensor<'a>, Tensor<'a>, f32)>,
}

/// The model's per-tensor matmul policy (see the module docs).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Linear {
    /// Weights whose activations stay at 8 bits or more (`prec_a4 = false`).
    no_a4: HashSet<String>,
    /// FP8 E4M3 weights kept as shipped (`I8` tensors, `legion.tensor_extra.fp8_e4m3`).
    fp8: HashSet<String>,
    /// Every weight keeps 8-bit or wider activations ([`Self::set_all_a8`]).
    all_a8: bool,
    /// Activations of FP8 matmuls ([`Self::set_fp8_act`]).
    fp8_act: Fp8Act,
    /// Kernels of NVFP4 matmuls on CUDA ([`Self::set_nvfp4_mm`]).
    nvfp4_mm: Nvfp4Mm,
    /// NVFP4 matmuls with 4-bit activations ([`Self::set_nvfp4_a4`]).
    nvfp4_a4: Nvfp4A4,
}

impl Linear {
    /// Reads `general.tensor_extra.{name,prec_a4}` (llama.cpp's `llama_prec_policy::load`).
    /// Both are optional; when present they must be arrays of the same length.
    pub fn from_gguf(g: &GgufFile) -> Result<Self, ModelError> {
        let bad = |k: &str| ModelError::Metadata(k.into());
        let fp8: HashSet<String> = match g.get(KEY_FP8_E4M3) {
            None => HashSet::new(),
            Some(v) => v
                .as_array()
                .ok_or_else(|| bad(KEY_FP8_E4M3))?
                .iter()
                .map(|n| {
                    n.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| bad(KEY_FP8_E4M3))
                })
                .collect::<Result<_, _>>()?,
        };
        let Some(names) = g.get(KEY_NAMES) else {
            return Ok(Self {
                fp8,
                ..Self::default()
            });
        };
        let names = names.as_array().ok_or_else(|| bad(KEY_NAMES))?;
        let Some(values) = g.get(KEY_PREC_A4) else {
            return Ok(Self {
                fp8,
                ..Self::default()
            });
        };
        let values = values.as_array().ok_or_else(|| bad(KEY_PREC_A4))?;
        if values.len() != names.len() {
            return Err(bad(KEY_PREC_A4));
        }
        let mut no_a4 = HashSet::new();
        for (n, v) in names.iter().zip(values) {
            let n = n.as_str().ok_or_else(|| bad(KEY_NAMES))?;
            if !v.as_bool().ok_or_else(|| bad(KEY_PREC_A4))? {
                no_a4.insert(n.to_string());
            }
        }
        Ok(Self {
            no_a4,
            fp8,
            all_a8: false,
            fp8_act: Fp8Act::default(),
            nvfp4_mm: Nvfp4Mm::default(),
            nvfp4_a4: Nvfp4A4::default(),
        })
    }

    /// Whether the weight `name` must keep 8-bit or wider activations.
    pub fn keeps_a8(&self, name: &str) -> bool {
        self.all_a8 || self.no_a4.contains(name)
    }

    /// Keeps every weight's activations at 8 bits or wider, listed in `prec_a4` or not (the
    /// converter leaves out NVFP4 experts it stacks, though W4A16 checkpoints declare them so).
    pub fn set_all_a8(&mut self, on: bool) {
        self.all_a8 = on;
    }

    /// How FP8 matmuls treat their activations (W8A8 needs the weight's `input_scale`).
    pub fn set_fp8_act(&mut self, a: Fp8Act) {
        self.fp8_act = a;
    }

    /// Which kernels run NVFP4 matmuls on CUDA.
    pub fn set_nvfp4_mm(&mut self, m: Nvfp4Mm) {
        self.nvfp4_mm = m;
    }

    /// Which NVFP4 matmuls take 4-bit activations.
    pub fn set_nvfp4_a4(&mut self, a: Nvfp4A4) {
        self.nvfp4_a4 = a;
    }

    /// Which NVFP4 matmuls take 4-bit activations.
    pub fn nvfp4_a4(&self) -> Nvfp4A4 {
        self.nvfp4_a4
    }

    /// Whether [`Nvfp4A4`] gives a batch of `n` tokens 4-bit activations (`grouped`: routed
    /// experts).
    fn a4_batch(&self, n: i64, grouped: bool) -> bool {
        match self.nvfp4_a4 {
            Nvfp4A4::Off => false,
            Nvfp4A4::Experts => grouped && n > NVFP4_A4_MIN_N,
            Nvfp4A4::Prefill => n > NVFP4_A4_MIN_N,
            Nvfp4A4::All => true,
            Nvfp4A4::ExpertsDecode => grouped && n > NVFP4_GGML_MAX_N,
        }
    }

    /// How an FP8 matmul of `k` inputs for a batch of `n` tokens takes its activations under
    /// [`Fp8Act`]: the static input scale when the weight has one (`has_input_scale`), else
    /// per-token scales for `a8-dynamic` prefill batches where `dynamic_ok` (an F32 `x`, a
    /// device weight, E4M3 tensor cores), else W8A16. W8A8 needs `k` a multiple of 64.
    pub(crate) fn fp8_route(
        &self,
        n: i64,
        k: i64,
        has_input_scale: bool,
        dynamic_ok: impl FnOnce() -> bool,
    ) -> Fp8Route {
        let a8 = match self.fp8_act {
            Fp8Act::A8 | Fp8Act::A8Dynamic => true,
            Fp8Act::A16 => false,
            Fp8Act::PrefillA8 => n > 8,
        };
        if !a8 || k % 64 != 0 {
            Fp8Route::A16
        } else if has_input_scale {
            Fp8Route::Static
        } else if self.fp8_act == Fp8Act::A8Dynamic && n > NVFP4_A4_MIN_N && dynamic_ok() {
            Fp8Route::Dynamic
        } else {
            Fp8Route::A16
        }
    }

    /// Whether FP8 matmuls without a static input scale may take [`Fp8Route::Dynamic`] in a
    /// batch of `n` tokens ([`Self::fp8_route`]): `a8-dynamic`, more than [`NVFP4_A4_MIN_N`]
    /// tokens, and a model with FP8 weights. The ops that write an activation's per-token FP8 codes directly
    /// ([`crate::model::fp8_act`]) are built only then.
    pub(crate) fn dynamic_batch(&self, n: i64) -> bool {
        self.fp8_act == Fp8Act::A8Dynamic && n > NVFP4_A4_MIN_N && !self.fp8.is_empty()
    }

    /// Whether a matmul of `w` that runs on Legion's NVFP4 kernels for a batch of `n` tokens
    /// takes the W4A4 kernels: [`Self::a4_batch`] on a GPU that has them.
    fn w4a4(&self, w: Tensor<'_>, n: i64, grouped: bool) -> bool {
        self.a4_batch(n, grouped) && crate::model::nvfp4_a4::supports(w)
    }

    /// [`Self::w4a4`] for models that build their NVFP4 matmuls themselves (GLM-5.3-Flash):
    /// whether `w`'s matmul for a batch of `n` tokens takes 4-bit activations under
    /// [`Nvfp4A4`] (`grouped`: routed experts).
    pub(crate) fn nvfp4_a4_takes(&self, w: Tensor<'_>, n: i64, grouped: bool) -> bool {
        self.w4a4(w, n, grouped)
    }

    /// Whether [`Nvfp4A4`] gives a batch of `n` tokens 4-bit activations where the weight
    /// supports them ([`Self::nvfp4_a4_takes`] before the weight is at hand).
    pub(crate) fn nvfp4_a4_batch(&self, n: i64, grouped: bool) -> bool {
        self.a4_batch(n, grouped)
    }

    /// Marks `res = w @ x` (`w` is the weight `name`) with the activation and accumulator
    /// precision its weight takes.
    pub(crate) fn hint(&self, ctx: &Context, name: &str, w: Tensor<'_>, res: Tensor<'_>) {
        if self.keeps_a8(name) {
            ctx.prec_set_src(res, ffi::ggml_prec_GGML_PREC_Q8, 1);
        }
        if w.ty() == types::NVFP4 {
            ctx.prec_set_acc(res, ffi::ggml_prec_GGML_PREC_BF16);
        }
    }

    /// Whether `w @ x` for `n` tokens runs on Legion's W4A16 NVFP4 kernels
    /// ([`crate::model::nvfp4_mm`]): an NVFP4 weight on a GPU that keeps 8-bit or wider
    /// activations, a batch [`Nvfp4Mm`] gives to them and a per-tensor (or per-expert) scale.
    fn w4a16(&self, name: &str, w: Tensor<'_>, scale: Option<Tensor<'_>>, n: i64) -> bool {
        let batch = match self.nvfp4_mm {
            Nvfp4Mm::Auto => n > NVFP4_GGML_MAX_N,
            Nvfp4Mm::Legion => true,
            Nvfp4Mm::Ggml => false,
        };
        batch
            && w.ty() == types::NVFP4
            && !w.is_host()
            && self.keeps_a8(name)
            && scale.is_none_or(|s| s.ne()[1..] == [1, 1, 1] && s.ne()[0] == w.ne()[2])
            && crate::model::nvfp4_mm::supports(w)
    }

    /// `w @ x` (`w` is the weight `name`), times `scale` when the checkpoint has one.
    /// `input_scale`: an FP8 weight's static activation scale, for W8A8.
    pub fn mul_mat<'a>(
        &self,
        ctx: &'a Context,
        name: &str,
        w: Tensor<'a>,
        scale: Option<Tensor<'a>>,
        input_scale: Option<Tensor<'a>>,
        x: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        // FP8 E4M3 bytes: Legion's FP8 matmul (W8A8 or W8A16), with the weight scale (per
        // tensor, or per row on W8A16) folded into its epilogue.
        if self.fp8.contains(name) {
            let n: i64 = x.ne()[1..].iter().product();
            let dynamic_ok =
                || x.ty() == types::F32 && !w.is_host() && crate::model::fp8::fp8_mma();
            let route = self.fp8_route(n, w.ne()[0], input_scale.is_some(), dynamic_ok);
            return Ok(match (route, input_scale) {
                (Fp8Route::Static, Some(s)) => {
                    let xq = crate::model::fp8::quantize(ctx, x, s);
                    match scale.filter(|s| s.elements() > 1) {
                        Some(rows) => {
                            ctx.mul(crate::model::fp8::mul_mat_a8(ctx, w, xq, s, None), rows)
                        }
                        None => crate::model::fp8::mul_mat_a8(ctx, w, xq, s, scale),
                    }
                }
                (Fp8Route::Dynamic, _) => crate::model::fp8::mul_mat_dynamic(ctx, w, x, scale),
                _ => crate::model::fp8::mul_mat(ctx, w, x, scale),
            });
        }
        let n: i64 = x.ne()[1..].iter().product();
        if self.w4a16(name, w, scale, n) && w.ne()[2] == 1 {
            // The scale folded into the kernels' epilogue too.
            if self.w4a4(w, n, false) {
                return Ok(crate::model::nvfp4_a4::mul_mat(ctx, w, x, scale));
            }
            return Ok(crate::model::nvfp4_mm::mul_mat(ctx, w, x, scale));
        }
        // BF16 weights in steps of 2-16 tokens: ggml-cuda's bits from faster kernels.
        let res = if crate::model::dense_mv::takes(w, x) {
            crate::model::dense_mv::mul_mat(ctx, &[w], x)[0]
        } else if crate::model::dense_mv::takes_wide_chain(w, x) {
            // BF16 weights in steps of 17-48 tokens: cuBLAS's wmma order, by default only where
            // that is cuBLAS's bits (#836).
            crate::model::dense_mv::mul_mat_wide_chain(ctx, w, x)
        } else if let Some((w16, x16)) = f16_joined(ctx, w, x) {
            // An F16 `x` an op wrote: ggml's cuBLAS GEMM without its conversions (#774).
            let res = ctx.mul_mat(w16, x16);
            self.hint(ctx, name, w, res);
            res
        } else if crate::model::core::q8_head::takes_out(w, x) {
            // One-column Q8_0 projections of 6,144 values on a V100 (attention out, DeltaNet
            // ssm_out), opt-in: one warp a row, the input quantized once per CTA, ggml's bits
            // (#814; slower than ggml's mmvq there, so off by default).
            crate::model::core::q8_head::mul_mat_out(ctx, w, x)
        } else if crate::model::q8_mmq::takes(w, x) {
            // Q8_0 weights in verify steps of 9-255 tokens: ggml's MMQ bits without its
            // stream-k fixup launch (#836).
            crate::model::q8_mmq::mul_mat(ctx, w, x)
        } else if crate::model::core::q_small_m::takes(w, x) {
            // GGUF-quantized weights in steps of 2-8 tokens on a V100: ggml's quantize + MMVQ
            // bits, opt-in (--q-small-m, #1210).
            crate::model::core::q_small_m::mul_mat(ctx, w, x)
        } else {
            let x = bf16_cast_joined(ctx, w, x).unwrap_or(x);
            let res = ctx.mul_mat(w, x);
            self.hint(ctx, name, w, res);
            res
        };
        Ok(match scale {
            Some(s) => ctx.mul(res, s),
            None => res,
        })
    }

    /// Whether a batch of `n` tokens runs matmuls that can add a residual in their epilogue
    /// ([`Self::mul_mat_add`]): `--fp8-act a8-dynamic` prefill batches, or batches
    /// [`Nvfp4A4`] gives 4-bit activations (dense weights). Models build the fused adds only
    /// then; elsewhere they keep ggml's add.
    pub(crate) fn residual_fusion(&self, n: i64) -> bool {
        self.dynamic_batch(n) || self.a4_batch(n, false)
    }

    /// `w @ x + res` ([`Self::mul_mat`], then ggml's `add(.., res)`) with the add in the
    /// matmul's epilogue where its kernel has one: an FP8 matmul on [`Fp8Route::Dynamic`]
    /// (the large-batch GEMM) or a dense W4A4 NVFP4 matmul, `res` contiguous F32 of the output's
    /// shape. The fused add rounds once, as ggml's does: the same bits either way.
    #[allow(clippy::too_many_arguments)]
    pub fn mul_mat_add<'a>(
        &self,
        ctx: &'a Context,
        name: &str,
        w: Tensor<'a>,
        scale: Option<Tensor<'a>>,
        input_scale: Option<Tensor<'a>>,
        x: Tensor<'a>,
        res: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        let [_, n1, n2, n3] = x.ne();
        let n = n1 * n2 * n3;
        let res_ok = res.ty() == types::F32
            && res.is_contiguous()
            && res.ne() == [w.ne()[1], n1, n2, n3]
            && w.ne()[2] == 1;
        if res_ok && self.fp8.contains(name) {
            use crate::model::fp8;
            let dynamic_ok = || x.ty() == types::F32 && !w.is_host() && fp8::fp8_mma();
            let route = self.fp8_route(n, w.ne()[0], input_scale.is_some(), dynamic_ok);
            if route == Fp8Route::Dynamic && fp8::dynamic_add_fits(n) {
                return Ok(fp8::mul_mat_dynamic_add(ctx, w, x, scale, res));
            }
        } else if res_ok && self.w4a16(name, w, scale, n) && self.w4a4(w, n, false) {
            return Ok(crate::model::nvfp4_a4::mul_mat_add(ctx, w, x, scale, res));
        }
        Ok(ctx.add(self.mul_mat(ctx, name, w, scale, input_scale, x)?, res))
    }

    /// The SwiGLU FFN `down @ (silu(gate @ x) * (up @ x))` with the gate and up matmuls as one
    /// fused W4A16 op ([`crate::model::nvfp4_mm::swiglu`], BF16 out) when all three weights take
    /// Legion's NVFP4 kernels at this batch size; `None`: run it unfused. Each weight is
    /// `(name, weight, scale)`.
    pub fn swiglu_ffn<'a>(
        &self,
        ctx: &'a Context,
        gate: (&str, Tensor<'a>, Option<Tensor<'a>>),
        up: (&str, Tensor<'a>, Option<Tensor<'a>>),
        down: (&str, Tensor<'a>, Option<Tensor<'a>>),
        x: Tensor<'a>,
    ) -> Option<Tensor<'a>> {
        self.swiglu_ffn_fused(ctx, gate, up, down, x, FfnFusion::default())
    }

    /// [`Self::swiglu_ffn`] with what [`FfnFusion`] asks fused into the W4A4 kernels (other
    /// kernels: ggml's ops for it, the same bits).
    pub fn swiglu_ffn_fused<'a>(
        &self,
        ctx: &'a Context,
        gate: (&str, Tensor<'a>, Option<Tensor<'a>>),
        up: (&str, Tensor<'a>, Option<Tensor<'a>>),
        down: (&str, Tensor<'a>, Option<Tensor<'a>>),
        x: Tensor<'a>,
        fusion: FfnFusion<'a>,
    ) -> Option<Tensor<'a>> {
        let n: i64 = x.ne()[1..].iter().product();
        let fits = |(name, w, s): (&str, Tensor<'_>, Option<Tensor<'_>>)| {
            self.w4a16(name, w, s, n) && w.ne()[2] == 1
        };
        let with_res = |out: Tensor<'a>| match fusion.res {
            Some(r) => ctx.add(out, r),
            None => out,
        };
        if !(fits(gate)
            && fits(up)
            && fits(down)
            && gate.1.ne() == up.1.ne()
            && gate.1.nb() == up.1.nb()
            && down.1.ne()[0] == gate.1.ne()[1])
        {
            return None;
        }
        if self.w4a4(gate.1, n, false) {
            use crate::model::nvfp4_a4;
            let act = match fusion.norm {
                Some((h, nw, eps)) if nvfp4_a4::rms_norm_fits(h, nw) => {
                    nvfp4_a4::swiglu_norm(ctx, gate.1, up.1, h, nw, eps, gate.2, up.2)
                }
                _ => nvfp4_a4::swiglu(ctx, gate.1, up.1, x, gate.2, up.2),
            };
            return Some(match fusion.res {
                Some(r) if down_res_fits(down.1, act, r) => {
                    nvfp4_a4::mul_mat_add(ctx, down.1, act, down.2, r)
                }
                _ => with_res(nvfp4_a4::mul_mat(ctx, down.1, act, down.2)),
            });
        }
        let act = crate::model::nvfp4_mm::swiglu(ctx, gate.1, up.1, x, gate.2, up.2);
        Some(with_res(crate::model::nvfp4_mm::mul_mat(
            ctx, down.1, act, down.2,
        )))
    }

    /// The GeGLU FFN `down @ (gelu(gate @ x) * (up @ x))` (Gemma 4) with the gate and up
    /// matmuls and the tanh GELU as one W4A16 op ([`crate::model::nvfp4_mm::geglu`], BF16 out:
    /// the bits of the two matmuls, ggml's `geglu_split` and the down matmul's cast to BF16),
    /// then the down matmul, in batches the GEMMs run (more than
    /// [`crate::model::nvfp4_mm::dense_mv_max_n`] tokens: prompt steps). The hidden activations
    /// never take the two F32 matmul outputs, the F32 GeGLU and the cast through memory. `None`
    /// (run it unfused) unless all three weights take Legion's W4A16 kernels at this batch.
    /// Batches [`Nvfp4A4`] gives 4-bit activations take the W4A4 GeGLU
    /// ([`crate::model::nvfp4_a4::geglu`], with `norm` (`(h, w, eps)`, `x` being
    /// `mul(rms_norm(h, eps), w)`) in its activations' quantization when it fits) and the W4A4
    /// down matmul instead: that opt-in moves the outputs anyway.
    pub fn geglu_ffn_fused<'a>(
        &self,
        ctx: &'a Context,
        gate: (&str, Tensor<'a>, Option<Tensor<'a>>),
        up: (&str, Tensor<'a>, Option<Tensor<'a>>),
        down: (&str, Tensor<'a>, Option<Tensor<'a>>),
        x: Tensor<'a>,
        norm: Option<(Tensor<'a>, Tensor<'a>, f32)>,
    ) -> Option<Tensor<'a>> {
        let n: i64 = x.ne()[1..].iter().product();
        let fits = |(name, w, s): (&str, Tensor<'_>, Option<Tensor<'_>>)| {
            self.w4a16(name, w, s, n) && w.ne()[2] == 1
        };
        if !(fits(gate)
            && fits(up)
            && fits(down)
            && gate.1.ne() == up.1.ne()
            && gate.1.nb() == up.1.nb()
            && down.1.ne()[0] == gate.1.ne()[1])
        {
            return None;
        }
        let a4 = self.w4a4(gate.1, n, false);
        if a4 != self.w4a4(down.1, n, false) {
            return None;
        }
        if a4 {
            use crate::model::nvfp4_a4;
            let act = match norm {
                Some((h, nw, eps)) if nvfp4_a4::rms_norm_fits(h, nw) => {
                    nvfp4_a4::geglu_norm(ctx, gate.1, up.1, h, nw, eps, gate.2, up.2)
                }
                _ => nvfp4_a4::geglu(ctx, gate.1, up.1, x, gate.2, up.2),
            };
            return Some(nvfp4_a4::mul_mat(ctx, down.1, act, down.2));
        }
        if n <= crate::model::nvfp4_mm::dense_mv_max_n() {
            return None;
        }
        let act = crate::model::nvfp4_mm::geglu(ctx, gate.1, up.1, x, gate.2, up.2);
        Some(crate::model::nvfp4_mm::mul_mat(ctx, down.1, act, down.2))
    }

    /// The matmuls of several weights `ws` (`(name, weight, scale)`) over one input `x` (such
    /// as attention's q, k and v projections) as one W4A4 op
    /// ([`crate::model::nvfp4_a4::mul_mat_multi`]: the activations quantized once, with `norm`
    /// (`(h, w, eps)`, `x` being `mul(rms_norm(h, eps), w)`) in that quantization when it fits),
    /// in batches [`Nvfp4A4`] gives 4-bit activations. `None` (run them one by one) otherwise.
    pub fn mul_mat_multi<'a>(
        &self,
        ctx: &'a Context,
        ws: &[(&str, Tensor<'a>, Option<Tensor<'a>>)],
        x: Tensor<'a>,
        norm: Option<(Tensor<'a>, Tensor<'a>, f32)>,
    ) -> Option<Vec<Tensor<'a>>> {
        use crate::model::nvfp4_a4;
        let n: i64 = x.ne()[1..].iter().product();
        if ws.len() < 2
            || ws.len() > nvfp4_a4::MULTI_MAX
            || !ws.iter().all(|&(name, w, s)| {
                self.w4a16(name, w, s, n)
                    && w.ne()[2] == 1
                    && w.ne()[0] == x.ne()[0]
                    && self.w4a4(w, n, false)
                    && !self.fp8.contains(name)
            })
        {
            return None;
        }
        let pairs: Vec<_> = ws.iter().map(|&(_, w, s)| (w, s)).collect();
        Some(match norm {
            Some((h, nw, eps)) if nvfp4_a4::rms_norm_fits(h, nw) => {
                nvfp4_a4::mul_mat_multi(ctx, &pairs, h, Some((nw, eps)))
            }
            _ => nvfp4_a4::mul_mat_multi(ctx, &pairs, x, None),
        })
    }

    /// [`Self::swiglu_ffn`] for routed experts (`x` `[k, 1 or n_used, n_tok]`, `ids` `[n_used,
    /// n_tok]`, per-expert scales): the gate and up matmuls and the activation as one grouped
    /// W4A16 op ([`crate::model::nvfp4_mm::swiglu_id`], BF16 out, the same bits as the unfused
    /// ops: the GEMM at prefill batches, the grouped SwiGLU mv at decode ones, #836), then the
    /// down matmul on its plan of the pairs; its out `[n_embd, n_used, n_tok]`. `None` (run it
    /// unfused) unless all three weights take Legion's W4A16 kernels at this batch. Batches
    /// [`Nvfp4A4`] gives 4-bit activations take the grouped W4A4 SwiGLU
    /// ([`crate::model::nvfp4_a4::swiglu_id`], F32 out, the same bits as its unfused ops) and the
    /// W4A4 down matmul instead, at any batch size. `rows` (W4A16 only): a verify step's row map
    /// ([`crate::model::core::pad_rows`]), which the plan of the pairs routes by.
    #[allow(clippy::too_many_arguments)]
    pub fn swiglu_ffn_id<'a>(
        &self,
        ctx: &'a Context,
        gate: (&str, Tensor<'a>, Option<Tensor<'a>>),
        up: (&str, Tensor<'a>, Option<Tensor<'a>>),
        down: (&str, Tensor<'a>, Option<Tensor<'a>>),
        x: Tensor<'a>,
        ids: Tensor<'a>,
        rows: Option<Tensor<'a>>,
    ) -> Option<Tensor<'a>> {
        use crate::model::nvfp4_mm::{mul_mat_id, swiglu_id_rows};
        let n = x.ne()[2];
        let m = gate.1.ne()[1];
        let shapes = gate.1.ne() == up.1.ne() && gate.1.nb() == up.1.nb() && down.1.ne()[0] == m;
        let a4 = |(name, w, s): (&str, Tensor<'_>, Option<Tensor<'_>>)| {
            self.w4a16(name, w, s, n) && self.w4a4(w, n, true)
        };
        if shapes && a4(gate) && a4(up) && a4(down) {
            use crate::model::nvfp4_a4;
            let act = nvfp4_a4::swiglu_id(ctx, gate.1, up.1, x, ids, gate.2, up.2);
            return Some(nvfp4_a4::mul_mat_id(ctx, down.1, act, ids, down.2));
        }
        let fits = |(name, w, s): (&str, Tensor<'_>, Option<Tensor<'_>>)| {
            self.w4a16(name, w, s, n) && !self.w4a4(w, n, true)
        };
        (shapes && fits(gate) && fits(up) && fits(down)).then(|| {
            let act = swiglu_id_rows(ctx, gate.1, up.1, x, ids, gate.2, up.2, rows);
            mul_mat_id(ctx, down.1, act, ids, down.2)
        })
    }

    /// Whether [`Self::swiglu_ffn_id_skips`] takes these experts (each `(name, weight,
    /// scale)`) at a batch of `n` tokens: all three on Legion's W4A16 grouped kernels.
    pub(crate) fn takes_skips(
        &self,
        gate: (&str, Tensor<'_>, Option<Tensor<'_>>),
        up: (&str, Tensor<'_>, Option<Tensor<'_>>),
        down: (&str, Tensor<'_>, Option<Tensor<'_>>),
        n: i64,
    ) -> bool {
        let shapes = gate.1.ne() == up.1.ne()
            && gate.1.nb() == up.1.nb()
            && down.1.ne()[0] == gate.1.ne()[1];
        let fits = |(name, w, s): (&str, Tensor<'_>, Option<Tensor<'_>>)| {
            self.w4a16(name, w, s, n) && !self.w4a4(w, n, true)
        };
        shapes && fits(gate) && fits(up) && fits(down)
    }

    /// [`Self::swiglu_ffn_id`]'s W4A16 ops with `ids` that may be -1
    /// ([`crate::model::nvfp4_mm::skips`]): those pairs go to no expert and their rows of the
    /// out are zeros. `--tp-moe-ep` (#1176): a rank's slot table, so it computes only its own
    /// experts' pairs. `None` unless [`Self::takes_skips`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn swiglu_ffn_id_skips<'a>(
        &self,
        ctx: &'a Context,
        gate: (&str, Tensor<'a>, Option<Tensor<'a>>),
        up: (&str, Tensor<'a>, Option<Tensor<'a>>),
        down: (&str, Tensor<'a>, Option<Tensor<'a>>),
        x: Tensor<'a>,
        ids: Tensor<'a>,
        rows: Option<Tensor<'a>>,
    ) -> Option<Tensor<'a>> {
        use crate::model::nvfp4_mm::{mul_mat_id, skips, swiglu_id_rows};
        let n = x.ne()[2];
        self.takes_skips(gate, up, down, n).then(|| {
            let act = skips(swiglu_id_rows(
                ctx, gate.1, up.1, x, ids, gate.2, up.2, rows,
            ));
            skips(mul_mat_id(ctx, down.1, act, ids, down.2))
        })
    }

    /// The squared-ReLU FFN `down @ relu(up @ x)²` with the up matmul's epilogue applying the
    /// activation and storing BF16 ([`crate::model::nvfp4_mm::Act::ReluSqr`]), when both weights
    /// take Legion's NVFP4 kernels at this batch size; `None`: run it unfused. Each weight is
    /// `(name, weight, scale)`.
    pub fn relu_sqr_ffn<'a>(
        &self,
        ctx: &'a Context,
        up: (&str, Tensor<'a>, Option<Tensor<'a>>),
        down: (&str, Tensor<'a>, Option<Tensor<'a>>),
        x: Tensor<'a>,
    ) -> Option<Tensor<'a>> {
        use crate::model::nvfp4_mm::{Act, mul_mat, mul_mat_act};
        let n: i64 = x.ne()[1..].iter().product();
        let fits = |(name, w, s): (&str, Tensor<'_>, Option<Tensor<'_>>)| {
            self.w4a16(name, w, s, n) && w.ne()[2] == 1
        };
        (fits(up) && fits(down) && down.1.ne()[0] == up.1.ne()[1]).then(|| {
            if self.w4a4(up.1, n, false) {
                use crate::model::nvfp4_a4;
                let act = nvfp4_a4::mul_mat_act(ctx, up.1, x, up.2, Act::ReluSqr);
                return nvfp4_a4::mul_mat(ctx, down.1, act, down.2);
            }
            let act = mul_mat_act(ctx, up.1, x, up.2, Act::ReluSqr);
            mul_mat(ctx, down.1, act, down.2)
        })
    }

    /// [`Self::relu_sqr_ffn`] for routed experts: `x` `[k, 1, n_tok]`, `ids` `[n_used,
    /// n_tok]`, per-expert scales; the down matmul's out `[n_embd, n_used, n_tok]`.
    pub fn relu_sqr_ffn_id<'a>(
        &self,
        ctx: &'a Context,
        up: (&str, Tensor<'a>, Option<Tensor<'a>>),
        down: (&str, Tensor<'a>, Option<Tensor<'a>>),
        x: Tensor<'a>,
        ids: Tensor<'a>,
    ) -> Option<Tensor<'a>> {
        use crate::model::nvfp4_mm::{Act, mul_mat_id, mul_mat_id_act};
        let n = x.ne()[2];
        let fits = |(name, w, s): (&str, Tensor<'_>, Option<Tensor<'_>>)| self.w4a16(name, w, s, n);
        (fits(up) && fits(down) && down.1.ne()[0] == up.1.ne()[1]).then(|| {
            if self.w4a4(up.1, n, true) {
                use crate::model::nvfp4_a4;
                let act = nvfp4_a4::mul_mat_id_act(ctx, up.1, x, ids, up.2, Act::ReluSqr);
                return nvfp4_a4::mul_mat_id(ctx, down.1, act, ids, down.2);
            }
            let act = mul_mat_id_act(ctx, up.1, x, ids, up.2, Act::ReluSqr);
            mul_mat_id(ctx, down.1, act, ids, down.2)
        })
    }

    /// [`Self::relu_sqr_ffn_id`] with the routing `weights` (`[1, n_used, n_tok]`) applied and
    /// the slots summed: `[n_embd, n_tok]`, when the batch takes the W4A4 kernels
    /// ([`crate::model::nvfp4_a4::mul_mat_id_weighted`]); `None`: run it the other way.
    pub fn relu_sqr_ffn_id_weighted<'a>(
        &self,
        ctx: &'a Context,
        up: (&str, Tensor<'a>, Option<Tensor<'a>>),
        down: (&str, Tensor<'a>, Option<Tensor<'a>>),
        x: Tensor<'a>,
        ids: Tensor<'a>,
        weights: Tensor<'a>,
    ) -> Option<Tensor<'a>> {
        use crate::model::nvfp4_a4;
        use crate::model::nvfp4_mm::Act;
        let n = x.ne()[2];
        let fits = |(name, w, s): (&str, Tensor<'_>, Option<Tensor<'_>>)| {
            self.w4a16(name, w, s, n) && self.w4a4(w, n, true)
        };
        (fits(up) && fits(down) && down.1.ne()[0] == up.1.ne()[1]).then(|| {
            let act = nvfp4_a4::mul_mat_id_act(ctx, up.1, x, ids, up.2, Act::ReluSqr);
            nvfp4_a4::mul_mat_id_weighted(ctx, down.1, act, ids, down.2, weights)
        })
    }

    /// The expert matmul `mul_mat_id(w, x, ids)` (`x` is `[k, n_used or 1, n_tokens]`,
    /// `ids` I32 `[n_used, n_tokens]`), each row times its expert's entry of `scale`
    /// (`[n_expert]`) when the checkpoint has one.
    pub fn mul_mat_id<'a>(
        &self,
        ctx: &'a Context,
        name: &str,
        w: Tensor<'a>,
        scale: Option<Tensor<'a>>,
        x: Tensor<'a>,
        ids: Tensor<'a>,
    ) -> Result<Tensor<'a>, ModelError> {
        // Block-scaled FP8 experts kept as shipped (an MTP head's), with their scale grid.
        if let Some(s) = scale.filter(|_| crate::model::fp8_block::takes(w, scale)) {
            return Ok(crate::model::fp8_block::mul_mat_id(ctx, w, x, ids, s));
        }
        if self.fp8.contains(name) {
            return Err(ModelError::Fp8Unsupported(name.into()));
        }
        if self.w4a16(name, w, scale, x.ne()[2]) {
            // Grouped by expert on the device, each row times its expert's scale.
            if self.w4a4(w, x.ne()[2], true) {
                return Ok(crate::model::nvfp4_a4::mul_mat_id(ctx, w, x, ids, scale));
            }
            return Ok(crate::model::nvfp4_mm::mul_mat_id(ctx, w, x, ids, scale));
        }
        if !super::expert_mv::usable(w, x, ids) && super::nvfp4_once::usable(w, scale, x, ids) {
            // ggml's MMVQ bits and its scale chain's, each routed expert read once per step (#836).
            return Ok(super::nvfp4_once::mul_mat_id(ctx, w, x, ids, scale));
        }
        let res = if super::expert_mv::usable(w, x, ids) {
            super::expert_mv::mul_mat_id(ctx, w, x, ids)
        } else if super::mmq_ids::usable(w, x, ids) {
            // ggml's MMQ bits on Legion's expert-grid kernel (`--expert-mmq`, #836).
            super::mmq_ids::mul_mat_id(ctx, w, x, ids)
        } else {
            let res = ctx.mul_mat_id(w, x, ids);
            self.hint(ctx, name, w, res);
            res
        };
        let Some(s) = scale else {
            return Ok(res);
        };
        let n_expert = s.ne()[0];
        let n_tokens = x.ne()[2];
        let s = ctx.reshape_3d(s, 1, n_expert, 1);
        let s = ctx.repeat_4d(s, [1, n_expert, n_tokens, 1]);
        Ok(ctx.mul(res, ctx.get_rows(s, ids)))
    }

    /// Whether [`Self::mul_mat_id`] of the expert stack `w` (the weight `name`, per-expert
    /// `scale`) for a step of `n` tokens takes each (token, slot) pair on its own, so a token
    /// may name one expert in several slots: the FP8 block kernel, Legion's grouped NVFP4
    /// kernels (their plans list pairs) and ggml's NVFP4 MMVQ (the steps of up to
    /// [`NVFP4_GGML_MAX_N`] tokens ggml-cuda keeps on it). Not ggml's MMQ or sorted paths, which
    /// count an expert once per token, nor `--nvfp4-once`. `--tp-moe-ep` with per-expert scales
    /// (an NVFP4 checkpoint's experts, #1150) remaps the other ranks' pairs onto slot 0.
    pub(crate) fn mmid_per_pair(
        &self,
        name: &str,
        w: Tensor<'_>,
        scale: Option<Tensor<'_>>,
        n: i64,
    ) -> bool {
        if crate::model::fp8_block::takes(w, scale) {
            return true;
        }
        if self.fp8.contains(name) || w.ty() != types::NVFP4 || w.is_host() {
            return false;
        }
        self.w4a16(name, w, scale, n) || (n <= NVFP4_GGML_MAX_N && !super::nvfp4_once::on())
    }
}

/// Up to this many tokens, [`Nvfp4Mm::Auto`] keeps NVFP4 matmuls on ggml's MMVQ: with Q8_1
/// activations it streams the weights at 230-250 GB/s on the GB10, Legion's W4A16 kernel at
/// 200-220 (`tests/nvfp4_mm.rs`, `cuda_timings`). Above it ggml switches to MMQ (150 GB/s at
/// 16 tokens), which Legion's kernels beat at every size. [`Nvfp4A4::ExpertsDecode`] keeps the
/// same split: on Qwen3.8-Flash-Next's experts, MMVQ beats the W4A4 decode kernel up to 8 tokens
/// (466 vs 525 us per layer at 4 tokens), the W4A4 kernel beats W4A16 above (2-8%,
/// `tests/nvfp4_a4.rs`, `cuda_decode_timings`).
pub const NVFP4_GGML_MAX_N: i64 = 8;

/// [`Nvfp4A4::Experts`] and [`Nvfp4A4::Prefill`] take batches of more than this many tokens:
/// prefill chunks, not decode steps (up to `--max-seqs` 16 sequences of 1 + 3 drafted tokens).
pub const NVFP4_A4_MIN_N: i64 = 64;

/// BF16 weight matmuls of more than this many tokens take a BF16 cast of `x` that another
/// matmul made ([`bf16_cast_joined`]). ggml-cuda runs BF16 weights at more than 16 tokens on
/// cuBLAS either way.
const BF16_JOIN_MIN_N: i64 = 32;

/// For ggml's `w @ x` with a BF16 weight `w` on a GPU with BF16 tensor cores, at more than
/// [`BF16_JOIN_MIN_N`] tokens: the BF16 cast of `x` that another matmul of `x` already made in
/// this graph ([`Context::cast_shared`]), if any, so ggml skips its own conversion. ggml-cuda
/// rounds F32 activations to BF16 for cuBLAS with the same rounding as the cast, and a BF16 `x`
/// goes to cuBLAS as is, so the result is the same. Without such a cast, `None`: ggml's
/// conversion is the faster one for a single matmul.
/// Whether residual `r` can ride in the epilogue of the down projection `w` (`[k, m]`, ggml's
/// layout: `m` output rows) of activations `act` (`[k, n, ..]`): contiguous F32 of the output's
/// shape `[m, n, ..]`. (#749: this compared `r` with `w.ne()[0]`, the input width, so the W4A4
/// down projection never took its residual and a separate add ran after it.)
fn down_res_fits(w: Tensor<'_>, act: Tensor<'_>, r: Tensor<'_>) -> bool {
    let [_, m, ..] = w.ne();
    let [_, n1, n2, n3] = act.ne();
    r.ty() == types::F32 && r.is_contiguous() && r.ne() == [m, n1, n2, n3]
}

fn bf16_cast_joined<'a>(ctx: &'a Context, w: Tensor<'a>, x: Tensor<'a>) -> Option<Tensor<'a>> {
    if !bf16_join_fits(w, x) {
        return None;
    }
    ctx.shared_cast(x, types::BF16)
}

/// Whether [`Linear::mul_mat`] of the plain weight `w` (no FP8 or NVFP4 route) takes a shared
/// BF16 cast of `x` ([`bf16_cast_joined`]).
pub(crate) fn bf16_join_fits(w: Tensor<'_>, x: Tensor<'_>) -> bool {
    let [_, n, x2, x3] = x.ne();
    w.ty() == types::BF16
        && w.ne()[2..] == [1, 1]
        && x.ty() == types::F32
        && x.is_contiguous()
        && n > BF16_JOIN_MIN_N
        && x2 == 1
        && x3 == 1
        && !w.is_host()
        && !w.is_meta()
        && crate::model::fp8::bf16_mma()
}

/// Q8_0 weight matmuls of at least this many tokens take an F16 copy of `x` an op wrote
/// ([`f16_joined`]): ggml-cuda's `MMQ_DP4A_MAX_BATCH_SIZE`, below which a V100 runs them as MMQ.
pub(crate) const F16_JOIN_MIN_N: i64 = 64;

/// For ggml's `w @ x` with a Q8_0 weight `w` in GPU memory, at [`F16_JOIN_MIN_N`] or more
/// tokens, when the op that made `x` also wrote its F16 rounding ([`Context::set_shared_cast`];
/// only done where ggml-cuda runs this matmul as a cuBLAS F16 GEMM of both operands rounded to
/// F16, a V100, see `qwen4::hc_mix16`): the operands of the same GEMM without ggml's per-matmul
/// conversions, `(w as F16, x as F16)` ([`super::f16_weight`]). The same bits.
pub(crate) fn f16_joined<'a>(
    ctx: &'a Context,
    w: Tensor<'a>,
    x: Tensor<'a>,
) -> Option<(Tensor<'a>, Tensor<'a>)> {
    let [_, n, x2, x3] = x.ne();
    let fits = w.ty() == types::Q8_0
        && w.ne()[2..] == [1, 1]
        && w.is_contiguous()
        && x.ty() == types::F32
        && x.is_contiguous()
        && n >= F16_JOIN_MIN_N
        && x2 == 1
        && x3 == 1
        && !w.is_host()
        && !w.is_meta();
    if !fits {
        return None;
    }
    let x16 = ctx.shared_cast(x, types::F16)?;
    Some((super::f16_weight::dequant(ctx, w), x16))
}

/// The scale tensor's name for the weight `name` (`blk.0.ffn_up.weight` -> `blk.0.ffn_up.scale`).
/// Whether matmuls of weight `name` of `g` take anything besides its bytes, by the name: a
/// `.scale` or `.input_scale` companion, the FP8 list, the per-tensor activation-precision list
/// ([`Linear::from_gguf`]). A copy under another name would lose them.
pub(crate) fn has_precision_hint(g: &GgufFile, name: &str) -> bool {
    let listed = |key: &str| {
        g.get(key)
            .and_then(|v| {
                v.as_array()
                    .map(|a| a.iter().any(|n| n.as_str() == Some(name)))
            })
            .unwrap_or(false)
    };
    let companion = |suffix: &str| {
        name.strip_suffix(".weight")
            .is_some_and(|b| g.tensor(&format!("{b}.{suffix}")).is_some())
    };
    companion("scale") || companion("input_scale") || listed(KEY_FP8_E4M3) || listed(KEY_NAMES)
}

pub fn scale_name(name: &str) -> Option<String> {
    name.strip_suffix(".weight").map(|b| format!("{b}.scale"))
}

/// `scale` for the first `rows` rows of an `m`-row weight: a per-tensor scale (one value) as
/// it is, a per-row one (contiguous `[m]`, e.g. a channel-wise FP8 output head) as a view of its
/// first `rows` values. `None`: any other layout.
pub(crate) fn first_rows_scale<'a>(
    ctx: &'a Context,
    scale: Tensor<'a>,
    m: i64,
    rows: i64,
) -> Option<Tensor<'a>> {
    if scale.elements() == 1 {
        return Some(scale);
    }
    (scale.ne() == [m, 1, 1, 1] && scale.is_contiguous() && rows <= m)
        .then(|| ctx.view_1d(scale, rows, 0))
}

#[cfg(test)]
mod tests;
