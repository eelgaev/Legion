//! `legion tokenize`: show how a model tokenizes text, like llama.cpp's `llama-tokenize`.
//! Only the GGUF tokenizer metadata is read; no weights are loaded.

use std::io::Read;
use std::path::PathBuf;

use clap::Args;
use legion::gguf::GgufFile;
use legion::tokenizer::Tokenizer;
use serde_json::json;

#[derive(Args)]
pub struct TokenizeArgs {
    /// GGUF file to load the tokenizer from.
    #[arg(long)]
    model: PathBuf,
    /// Parse special tokens (`<|im_start|>`-style) in the text as single ids.
    #[arg(long)]
    special: bool,
    /// Print only the token ids, comma-separated on one line.
    #[arg(long, conflicts_with = "json")]
    ids_only: bool,
    /// Machine-readable output.
    #[arg(long)]
    json: bool,
    /// Text to tokenize; all of stdin when omitted.
    text: Option<String>,
}

pub fn run(args: TokenizeArgs) -> anyhow::Result<()> {
    let text = match &args.text {
        Some(t) => t.clone(),
        None => {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            buf
        }
    };

    let gguf = GgufFile::open(&args.model)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    // Never add BOS/EOS: this shows the text's own tokens.
    let ids = tok.encode_with(&text, false, args.special);

    if args.ids_only {
        let line = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        println!("{line}");
    } else if args.json {
        let tokens: Vec<_> = ids
            .iter()
            .map(|&id| {
                json!({
                    "id": id,
                    "piece": String::from_utf8_lossy(tok.token_to_piece(id)),
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string(&json!({ "tokens": tokens, "count": ids.len() }))?
        );
    } else {
        for &id in &ids {
            // `{:?}` keeps the leading space and newlines of a piece visible.
            let piece = String::from_utf8_lossy(tok.token_to_piece(id));
            println!("{id}\t{piece:?}");
        }
        println!("total: {} tokens", ids.len());
    }
    Ok(())
}
