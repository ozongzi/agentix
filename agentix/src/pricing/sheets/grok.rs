//! xAI Grok price sheets (USD per 1M tokens).
//!
//! Snapshot of xAI's published rates, 2026-10. **Source caveat:** docs.x.ai has
//! been unreachable from this environment on every attempt (fetch failures and
//! HTTP 000), so these figures come from xAI's rates as mirrored by the
//! OpenRouter catalog — which this module already prefers for authority. Treat
//! them as approximate, and prefer a provider-reported cost.
//!
//! Billing rules: cached tokens are a subset of `prompt_tokens`; caching is
//! automatic with no documented write fee; reasoning is billed at the output
//! rate. A long-context surcharge tier applies above a model-specific
//! threshold, re-tiering the whole request when the total prompt (including
//! cache) crosses it — grok-4.7 doubles above 200k, and grok-4-fast's
//! historical 128k doubling is retained.

use crate::pricing::{PriceSheet, Rates, Tier, dec};
use rusty_money::iso;

pub fn sheet(model: &str) -> Option<PriceSheet> {
    let m = model.to_ascii_lowercase();
    if m.contains("fast") {
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
    } else if m.contains("4.7") {
        // Base $2 / $0.50 / $6, doubling above a 200k-token prompt.
        Some(PriceSheet {
            currency: iso::USD,
            tiers: vec![
                Tier {
                    up_to: Some(200_000),
                    rates: Rates::simple(dec("2"), dec("0.50"), dec("6")),
                },
                Tier {
                    up_to: None,
                    rates: Rates::simple(dec("4"), dec("1"), dec("12")),
                },
            ],
        })
    } else if m.contains("4.5") {
        Some(PriceSheet::flat(
            iso::USD,
            Rates::simple(dec("2"), dec("0.50"), dec("6")),
        ))
    } else if m.contains("grok") {
        // grok-4.3 and earlier baseline; cached rate reported inconsistently
        // ($0.20 vs $0.50 across sources) — the conservative (higher) figure
        // is used.
        Some(PriceSheet::flat(
            iso::USD,
            Rates::simple(dec("1.25"), dec("0.50"), dec("2.50")),
        ))
    } else {
        None
    }
}
