//! Forward execution paths — all task types (generate, classify, embed)
//! and paged attention operations (prefill, decode).

mod classify;
mod embed;
mod generate;
mod paged_decode;
mod paged_mixed;
mod paged_prefill;
mod prefill;
mod prefill_output;
