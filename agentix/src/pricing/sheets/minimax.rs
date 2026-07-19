//! MiniMax price sheets (USD per 1M tokens, international platform).
//!
//! Snapshot of <https://platform.minimax.io/docs/guides/pricing-paygo>,
//! 2026-07. Billing rules: Anthropic-additive usage semantics
//! (`input_tokens` excludes cache tokens); explicit cache read and write
//! rates. MiniMax-M3 tiers by prompt length at 512k.

use crate::pricing::{PriceSheet, Rates, Tier, dec};
use rusty_money::iso;

fn m2_rates(cache_read: &str) -> Rates {
    Rates {
        input: dec("0.30"),
        cache_read: dec(cache_read),
        cache_write_5m: dec("0.375"),
        output: dec("1.20"),
        ..Default::default()
    }
}

pub fn sheet(model: &str) -> Option<PriceSheet> {
    let m = model.to_ascii_lowercase();
    if m.contains("m3") {
        Some(PriceSheet {
            currency: iso::USD,
            tiers: vec![
                Tier {
                    up_to: Some(512_000),
                    rates: m2_rates("0.06"),
                },
                Tier {
                    up_to: None,
                    rates: Rates {
                        input: dec("0.60"),
                        cache_read: dec("0.06"),
                        cache_write_5m: dec("0.375"),
                        output: dec("2.40"),
                        ..Default::default()
                    },
                },
            ],
        })
    } else if m.contains("m2.7") {
        Some(PriceSheet::flat(iso::USD, m2_rates("0.06")))
    } else if m.contains("minimax") {
        // MiniMax-M2 legacy.
        Some(PriceSheet::flat(iso::USD, m2_rates("0.03")))
    } else {
        None
    }
}
