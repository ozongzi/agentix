//! OpenRouter — a *dynamic* price source.
//!
//! OpenRouter's `/api/v1/models` catalog quotes USD **per single token**
//! (strings), covering `prompt`, `completion`, `input_cache_read`,
//! `input_cache_write`, `internal_reasoning`, plus per-unit `request` /
//! `image` / `web_search` fees. Cache pricing is passed through from the
//! upstream with no markup.
//!
//! [`sheet_from_catalog`] converts one catalog entry into a [`PriceSheet`].
//! Note that OpenRouter also returns the **actual charge** on every response
//! (`usage.cost` → [`crate::types::Usage::reported_cost`]), which
//! [`crate::pricing::Cost`] prefers over any estimate — the sheet mainly
//! serves cost *projection* and breakdowns.

use crate::pricing::{PriceSheet, Rates};
use rust_decimal::Decimal;
use rusty_money::iso;

/// Pricing fields of one `/api/v1/models` catalog entry, as raw strings of
/// USD-per-token (missing/blank = free).
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct CatalogPricing {
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub completion: Option<String>,
    #[serde(default)]
    pub input_cache_read: Option<String>,
    #[serde(default)]
    pub input_cache_write: Option<String>,
    #[serde(default)]
    pub internal_reasoning: Option<String>,
    #[serde(default)]
    pub image: Option<String>,
}

fn per_million(s: &Option<String>) -> Decimal {
    s.as_deref()
        .and_then(|v| v.parse::<Decimal>().ok())
        .unwrap_or_default()
        * Decimal::from(1_000_000u64)
}

/// Convert one catalog entry (USD per token) into a per-1M [`PriceSheet`].
pub fn sheet_from_catalog(p: &CatalogPricing) -> PriceSheet {
    let reasoning = per_million(&p.internal_reasoning);
    PriceSheet::flat(
        iso::USD,
        Rates {
            input: per_million(&p.prompt),
            cache_read: per_million(&p.input_cache_read),
            cache_write_5m: per_million(&p.input_cache_write),
            output: per_million(&p.completion),
            // Only override the reasoning rate when the catalog quotes a
            // non-zero one; otherwise reasoning bills at the output rate.
            reasoning: (!reasoning.is_zero()).then_some(reasoning),
            // `image` is USD per image — a per-unit fee, not per-token.
            per_image: p
                .image
                .as_deref()
                .and_then(|v| v.parse::<Decimal>().ok())
                .filter(|d| !d.is_zero()),
            ..Default::default()
        },
    )
}
