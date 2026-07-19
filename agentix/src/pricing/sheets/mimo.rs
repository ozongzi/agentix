//! Xiaomi MiMo price sheets (CNY per 1M tokens — domestic platform prices).
//!
//! Snapshot of <https://mimo.mi.com/docs/en-US/price/pay-as-you-go>,
//! 2026-07. Billing rules: input priced by cache hit vs miss (input bucket =
//! miss price, cache_read = hit price); cache writes currently
//! "Limited-time Free". The overseas platform quotes USD instead — swap the
//! sheet if billing against it.

use crate::pricing::{PriceSheet, Rates, dec};
use rusty_money::iso;

pub fn sheet(model: &str) -> Option<PriceSheet> {
    let m = model.to_ascii_lowercase();
    let rates = if m.contains("ultraspeed") {
        Rates::simple(dec("9"), dec("0.075"), dec("18"))
    } else if m.contains("pro") {
        Rates::simple(dec("3"), dec("0.025"), dec("6"))
    } else if m.contains("mimo") {
        Rates::simple(dec("1"), dec("0.02"), dec("2"))
    } else {
        return None;
    };
    Some(PriceSheet::flat(iso::CNY, rates))
}
