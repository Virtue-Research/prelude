//! Forward execution paths — all task types (generate, classify, embed)
//! and paged attention operations (prefill, decode).

mod classify;
mod embed;
mod generate;
mod paged_decode;
mod paged_prefill;
mod prefill_output;
mod prefill;

pub(crate) use self::classify::{RawClassifyOutput, classify_postprocess};
pub(crate) use self::embed::{RawEmbedOutput, embed_postprocess};
pub(crate) use self::generate::{RawGenerateOutput, generate_postprocess};
