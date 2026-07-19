//! xAI Grok price sheets (USD per 1M tokens).
//!
//! Snapshot from <https://docs.x.ai/developers/pricing>, 2026-07 (gathered
//! via search snippets — docs.x.ai was unreachable directly; treat as
//! approximate and prefer the OpenRouter dynamic source for authority).
//! Billing rules: cached tokens are a subset of `prompt_tokens`; caching is
//! automatic with no documented write fee; reasoning is billed at the output
//! rate. A long-context surcharge tier applies above a model-specific
//! threshold (whole request re-tiers when total prompt incl. cache exceeds
//! it) — threshold values are unverified, so only grok-4-fast's historical
//! 128k doubling is modeled.

use crate::pricing::{PriceSheet, Rates, Tier, dec};
use rusty_money::iso;

pub fn sheet(model: &str) -> Option<PriceSheet> {
    let m = model.to_ascii_lowercase();
    if m.contains("4-fast") || m.contains("fast") {
        Some(PriceSheet {
            currency: iso::USD,
            tiers: vec![
                Tier {
                    up_to: Some(128_000),
                    rates: Rates::simple(dec("0.20"), dec("0.05"), dec("0.50")),
                },
                Tier {
                    up_to: None,
                    rates: Rates::simple(dec("0.40"), dec("0.10"), dec("1")),
                },
            ],
        })
    } else if m.contains("4.5") {
        Some(PriceSheet::flat(
            iso::USD,
            Rates::simple(dec("2"), dec("0.50"), dec("6")),
        ))
    } else if m.contains("grok") {
        // grok-4.3 baseline; cached rate reported inconsistently ($0.20 vs
        // $0.50 across sources) — the conservative (higher) figure is used.
        Some(PriceSheet::flat(
            iso::USD,
            Rates::simple(dec("1.25"), dec("0.50"), dec("2.50")),
        ))
    } else {
        None
    }
}
