//! Zhipu / GLM price sheets (USD per 1M tokens, z.ai platform).
//!
//! Snapshot of <https://docs.z.ai/guides/overview/pricing>, 2026-07.
//! Billing rules: `prompt_tokens_details.cached_tokens` is a subset of
//! `prompt_tokens`. Implicit/automatic caching; a cache-storage billing
//! dimension exists but is currently free ("Limited-time Free"). No context
//! length tiers. GLM-4.7-Flash / 4.5-Flash are free.

use crate::pricing::{PriceSheet, Rates, dec};
use rust_decimal::Decimal;
use rusty_money::iso;

pub fn sheet(model: &str) -> Option<PriceSheet> {
    let m = model.to_ascii_lowercase();
    let rates = if m.contains("flash") {
        Rates::simple(Decimal::ZERO, Decimal::ZERO, Decimal::ZERO)
    } else if m.contains("5.2") || m.contains("5.1") {
        Rates::simple(dec("1.40"), dec("0.26"), dec("4.40"))
    } else if m.contains("glm-5") {
        Rates::simple(dec("1"), dec("0.20"), dec("3.20"))
    } else if m.contains("4.7") || m.contains("4.6") {
        Rates::simple(dec("0.60"), dec("0.11"), dec("2.20"))
    } else {
        return None;
    };
    Some(PriceSheet::flat(iso::USD, rates))
}
