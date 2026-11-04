//! `legion completions <shell>`: print a tab-completion script for the given shell
//! to stdout, e.g. `legion completions bash > ~/.local/share/bash-completion/completions/legion`.

use clap::Args;

#[derive(Args)]
pub struct CompletionsArgs {
    /// Shell to print a completion script for.
    shell: clap_complete::Shell,
}

pub fn run(args: CompletionsArgs) -> anyhow::Result<()> {
    // The script is the command's output, so only it goes to stdout.
    clap_complete::generate(
        args.shell,
        &mut super::command(),
        "legion",
        &mut std::io::stdout(),
    );
    Ok(())
}
