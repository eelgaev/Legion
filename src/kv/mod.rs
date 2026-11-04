//! Fixed-size KV block ownership and per-sequence block tables.

pub mod blocks;
pub mod radix;

pub use blocks::{BlockId, BlockPool, BlockTable, CopyOp, OutOfBlocks};
