# Contributing to Legion

Thanks for your interest in Legion. Bug reports, benchmark results on new hardware, fixes and
features are all welcome. This page explains what a pull request needs in order to be merged.

## AI-assisted contributions

You may use AI tools to write code, tests and docs. The rules are the same as for code you
typed yourself:

- **You must understand every line you submit.** You should be able to explain in review why
  the change is correct, what it does to performance and output, and why you chose it over the
  alternatives. "The model wrote it" is not an answer to a review question.
- **You must have built and run it.** Run the checks below on your own machine, and run the
  changed code paths for real (the tests, plus `legion serve` on a model when you touch the
  engine, kernels or the server).
- **Keep PRs focused.** One change per PR, with a description written for a human reviewer.
  Large generated diffs that nobody has read will be closed.

A PR that does not pass the tests does not land, however it was written.

## Before you open a PR

The checks CI runs, which must all pass:

```sh
git submodule update --init third_party/ggml
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

When you change CUDA code, also:

```sh
cargo clippy --all-targets --features cuda -- -D warnings
cargo test --features cuda                         # on whatever NVIDIA GPU you have
cargo test --features cuda -- --ignored            # the GPU tests touching your change
```

`rust-toolchain.toml` pins the Rust version, so clippy gives the same result on every machine.
Build details are in [docs/building.md](docs/building.md).

Test on the hardware you have and say in the PR what it was. Nobody is expected to own a
particular machine; maintainers may re-run a change on other hardware before merging it.

## Rules for the code

- **Rust only. C++ is not accepted**, and neither is C. Host code is Rust. GPU kernels live in
  `src/cuda/` (CUDA device code or PTX) and contain only the kernels; everything around them
  (memory, launches, streams) is Rust. The C/C++ already in the tree is third-party or
  tooling (ggml in `third_party/`, its bindgen header, a vendored image decoder, the reference
  tools in `tools/`) and is not a pattern to extend.
- **Do not edit `third_party/`.** It is vendored as is.
- **Permissive licenses only.** Legion is Apache-2.0. Dependencies must pass `cargo deny check
  licenses` ([deny.toml](deny.toml)). Do not copy code from GPL, LGPL or AGPL projects, and do
  not port it line by line either.
- **Defaults keep the output byte-identical.** With greedy decoding (temperature 0), the default
  settings must produce exactly the same tokens before and after your change. A change that
  moves the output, even within rounding (a different accumulation order, a lower precision, a
  new kernel), must be an opt-in flag, off by default, and documented as output-moving. Check
  with [scripts/determinism.py](scripts/determinism.py): run it against the server before and
  after your change, restart in between, and compare.
- **Settings are command-line flags, not environment variables.** New features and tuning knobs
  are `clap` arguments of `legion serve` (or the relevant subcommand), documented in its
  `--help`. Environment variables are only for secrets and debugging.
- **Tests go in their own files.** Unit tests live in a `tests.rs` file (or a `tests/` directory)
  next to the module, never in an inline `mod tests { ... }` block. Integration tests go in `tests/`.
  Tests that need a GPU are marked `#[ignore = "needs a GPU"]`.
- **Do not break other hardware.** Legion aims to run on all hardware. A fast path for one GPU
  architecture or machine needs a fallback, so every GPU that worked before still works
  (NVIDIA GPUs from Volta, sm_70, are supported today). Hardware-specific choices are picked at
  run time from what the device reports, not hard-coded.

## Performance changes

A PR that claims a speedup needs numbers:

- the machine (GPU model and count), the model and quantization, and the exact `legion serve`
  command, before and after;
- decode at 1, 4 and 16 users, and prefill (for example 8K and 32K tokens), measured from the
  same server, since a decode gain that costs prefill (or the reverse) needs to be visible;
- whether the greedy output is still identical (see above).

[docs/bench.md](docs/bench.md) describes `legion bench`, and [docs/run](docs/run/README.md) shows
how the README's numbers were measured.

## Commits and pull requests

- Branch from `main`. Keep the history clean: rebase on `main` rather than merging it in, and
  squash fix-up commits.
- Commit titles are short and in the imperative ("Add NVFP4 grouped GEMM for sm_120"). The body
  says why the change is needed and anything a reviewer would not see from the diff.
- The PR description says what changed, why, how you tested it, and (for performance work) the
  numbers.
- Do not put machine names, IP addresses, private paths or credentials in code, docs, commits or
  PR text.

## Reporting bugs

Open an issue with:

- the Legion commit (`git rev-parse HEAD`) and the full `legion serve` command;
- the GPU(s), driver and CUDA versions, and the OS;
- the model (Hugging Face repo and file);
- what you expected, what happened, and the server log around the failure.

Security problems: please report them privately through GitHub's "Report a vulnerability"
(the repository's Security tab) instead of opening a public issue.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), the same as the rest of Legion.
