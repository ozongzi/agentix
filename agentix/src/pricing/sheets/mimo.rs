//! Xiaomi MiMo price sheets (CNY per 1M tokens — domestic platform prices).
//!
//! Snapshot of <https://mimo.mi.com/docs/en-US/price/pay-as-you-go>,
//! 2026-10. Billing rules: input priced by cache hit vs miss (input bucket =
//! miss price, cache_read = hit price); cache writes currently
//! "Limited-time Free". The same page also quotes an overseas USD table
//! (mimo-v2.6-pro: $0.435 / $0.0036 / $0.87) — swap the sheet if billing
//! against it. The v2.6 generation is flat-priced (no length tiers).

use crate::pricing::{PriceSheet, Rates, dec};
use rusty_money::iso;

pub fn sheet(model: &str) -> Option<PriceSheet> {
    let m = model.to_ascii_lowercase();
    let rates = if m.contains("ultraspeed") {
        Rates::simple(dec("30"), dec("0.25"), dec("60"))
    } else if m.contains("pro") {
        Rates::simple(dec("3"), dec("0.025"), dec("6"))
    } else if m.contains("mimo") {
        // mimo-v2.6-flash (and the legacy v2.5 base tier).
        Rates::simple(dec("1"), dec("0.02"), dec("2"))
    } else {
        return None;
    };
    Some(PriceSheet::flat(iso::CNY, rates))
}
