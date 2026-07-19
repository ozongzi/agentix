//! Google Gemini price sheets (USD per 1M tokens).
//!
//! Snapshot of <https://ai.google.dev/gemini-api/docs/pricing>, 2026-07.
//! Billing rules: `cachedContentTokenCount` is a subset of the prompt (billed
//! at the cache rate, ~10% of input); thinking tokens are additive to
//! candidates and billed at the output rate; Pro-class models tier both
//! rates by prompt length (≤200k vs >200k, total prompt incl. cache).
//! Explicit caches additionally pay per-token-hour storage — NOT modeled
//! here (implicit caching has no storage charge). Batch = 50%.

use crate::pricing::{PriceSheet, Rates, Tier, dec};
use rusty_money::iso;

pub fn sheet(model: &str) -> Option<PriceSheet> {
    let m = model.to_ascii_lowercase();
    if m.contains("3.1-pro") || m.contains("3-pro") {
        Some(PriceSheet {
            currency: iso::USD,
            tiers: vec![
                Tier {
                    up_to: Some(200_000),
                    rates: Rates::simple(dec("2"), dec("0.20"), dec("12")),
                },
                Tier {
                    up_to: None,
                    rates: Rates::simple(dec("4"), dec("0.40"), dec("18")),
                },
            ],
        })
    } else if m.contains("2.5-pro") {
        Some(PriceSheet {
            currency: iso::USD,
            tiers: vec![
                Tier {
                    up_to: Some(200_000),
                    rates: Rates::simple(dec("1.25"), dec("0.125"), dec("10")),
                },
                Tier {
                    up_to: None,
                    rates: Rates::simple(dec("2.50"), dec("0.25"), dec("15")),
                },
            ],
        })
    } else if m.contains("flash") {
        Some(PriceSheet::flat(
            iso::USD,
            Rates {
                input: dec("0.30"),
                input_audio: Some(dec("1")),
                cache_read: dec("0.03"),
                output: dec("2.50"),
                ..Default::default()
            },
        ))
    } else {
        None
    }
}
