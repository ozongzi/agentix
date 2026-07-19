//! Cost calculation over normalized [`Usage`] buckets.
//!
//! Three concepts, mirroring how a human reads a pricing page:
//!
//! - [`Rates`] — per-bucket prices, quoted as **money per 1M tokens** in the
//!   sheet's native currency (same unit every provider's docs use).
//! - [`PriceSheet`] — one model's price card: a currency plus one or more
//!   [`Tier`]s. Multi-tier sheets model "prompts ≤200k: X, >200k: Y" pricing
//!   (Gemini Pro, Grok long-context, MiniMax M3): the tier is selected by the
//!   request's **total input tokens including cache**, and the whole request
//!   is billed at that tier — the semantics all three providers document.
//! - [`Cost`] — the answer to "how much, where did it go, and what did the
//!   cache save me", in [`rusty_money::Money`].
//!
//! Static per-model sheets live next to each provider's adapter
//! (`raw::<provider>::pricing`); dynamic sources (the OpenRouter catalog)
//! produce the same [`PriceSheet`] shape.
//!
//! When the provider itself reports a cost ([`Usage::reported_cost`], e.g.
//! OpenRouter's `usage.cost`), that figure is authoritative and
//! [`Cost::total`] returns it in preference to the estimate.

pub mod sheets;

use rust_decimal::Decimal;
use rusty_money::{Money, iso};

use crate::types::Usage;

/// ISO-4217 currency, re-exported for sheet definitions.
pub type Currency = iso::Currency;

/// A monetary amount in a statically-known ISO currency.
pub type Amount = Money<'static, Currency>;

const MILLION: u64 = 1_000_000;

/// Parse a decimal literal used in a price table. Panics on malformed input —
/// price tables are compile-time constants, so a bad literal is a bug.
pub fn dec(s: &str) -> Decimal {
    s.parse()
        .unwrap_or_else(|_| panic!("invalid price literal: {s}"))
}

/// Per-bucket prices for one tier, quoted **per 1M tokens** (or per unit for
/// [`Rates::per_image`]) in the owning [`PriceSheet`]'s currency.
///
/// `None` fields fall back as documented on each field, so a sheet only
/// spells out the rates its provider actually publishes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rates {
    /// Fresh text input, per 1M tokens.
    pub input: Decimal,
    /// Image input tokens. Defaults to `input`.
    pub input_image: Option<Decimal>,
    /// Audio input tokens. Defaults to `input`.
    pub input_audio: Option<Decimal>,
    /// Video input tokens. Defaults to `input`.
    pub input_video: Option<Decimal>,
    /// Document (PDF) input tokens. Defaults to `input`.
    pub input_document: Option<Decimal>,
    /// Flat price **per input image** (OpenRouter-style). Defaults to zero.
    pub per_image: Option<Decimal>,
    /// Cache-read tokens.
    pub cache_read: Decimal,
    /// Cache-write tokens, 5-minute TTL (or the provider's only write rate).
    pub cache_write_5m: Decimal,
    /// Cache-write tokens, 1-hour TTL. Defaults to `cache_write_5m`.
    pub cache_write_1h: Option<Decimal>,
    /// Visible output tokens (reasoning uses `reasoning` if set, else this).
    pub output: Decimal,
    /// Audio output tokens. Defaults to `output`.
    pub output_audio: Option<Decimal>,
    /// Reasoning tokens. Defaults to `output` — the rule on every surveyed
    /// provider; OpenRouter can quote a distinct `internal_reasoning` rate.
    pub reasoning: Option<Decimal>,
}

impl Rates {
    /// The common case: text input / cache read / output, everything else
    /// derived. Cache writes default to free (the norm outside Anthropic).
    pub fn simple(input: Decimal, cache_read: Decimal, output: Decimal) -> Self {
        Rates {
            input,
            cache_read,
            output,
            ..Default::default()
        }
    }

    fn input_image(&self) -> Decimal {
        self.input_image.unwrap_or(self.input)
    }
    fn input_audio(&self) -> Decimal {
        self.input_audio.unwrap_or(self.input)
    }
    fn input_video(&self) -> Decimal {
        self.input_video.unwrap_or(self.input)
    }
    fn input_document(&self) -> Decimal {
        self.input_document.unwrap_or(self.input)
    }
    fn cache_write_1h(&self) -> Decimal {
        self.cache_write_1h.unwrap_or(self.cache_write_5m)
    }
    fn output_audio(&self) -> Decimal {
        self.output_audio.unwrap_or(self.output)
    }
    fn reasoning(&self) -> Decimal {
        self.reasoning.unwrap_or(self.output)
    }
}

/// One pricing tier: applies when the request's total input tokens
/// (including cache) is `<= up_to`. `None` = unbounded (the last tier).
#[derive(Debug, Clone, PartialEq)]
pub struct Tier {
    pub up_to: Option<u64>,
    pub rates: Rates,
}

/// One model's price card.
#[derive(Debug, Clone, PartialEq)]
pub struct PriceSheet {
    pub currency: &'static Currency,
    /// Ascending by `up_to`; the final tier should have `up_to: None`.
    pub tiers: Vec<Tier>,
}

impl PriceSheet {
    /// A single-tier sheet — the common case.
    pub fn flat(currency: &'static Currency, rates: Rates) -> Self {
        PriceSheet {
            currency,
            tiers: vec![Tier { up_to: None, rates }],
        }
    }

    /// The tier a request with this many total input tokens falls into.
    pub fn rates_for(&self, total_input: u64) -> &Rates {
        self.tiers
            .iter()
            .find(|t| t.up_to.is_none_or(|cap| total_input <= cap))
            .or(self.tiers.last())
            .map(|t| &t.rates)
            .expect("PriceSheet must have at least one tier")
    }

    /// Price a request. Estimation always runs; a provider-reported cost
    /// takes precedence for [`Cost::total`].
    ///
    /// Errors when a reported cost exists but cannot be interpreted (unknown
    /// ISO currency code, non-finite amount) — never silently discarded: the
    /// caller decides whether to surface it or knowingly use
    /// [`Cost::estimated`] from a sheet-only pricing pass.
    pub fn cost(&self, usage: &Usage) -> Result<Cost, CostError> {
        let rates = self.rates_for(usage.total_input());
        let per_tok = |tokens: u64, per_million: Decimal| {
            Decimal::from(tokens) * per_million / Decimal::from(MILLION)
        };

        let input = per_tok(usage.input, rates.input);
        let input_image = per_tok(usage.input_image, rates.input_image());
        let input_audio = per_tok(usage.input_audio, rates.input_audio());
        let input_video = per_tok(usage.input_video, rates.input_video());
        let input_document = per_tok(usage.input_document, rates.input_document());
        let images = Decimal::from(usage.images) * rates.per_image.unwrap_or_default();
        let cache_read = per_tok(usage.cache_read, rates.cache_read);
        let cache_write = per_tok(usage.cache_write_5m, rates.cache_write_5m)
            + per_tok(usage.cache_write_1h, rates.cache_write_1h());
        let output = per_tok(usage.output, rates.output);
        let output_audio = per_tok(usage.output_audio, rates.output_audio());
        let reasoning = per_tok(usage.reasoning, rates.reasoning());

        let estimated = input
            + input_image
            + input_audio
            + input_video
            + input_document
            + images
            + cache_read
            + cache_write
            + output
            + output_audio
            + reasoning;

        // "What would this have cost with no cache at all" — every cached
        // token repriced at the fresh text-input rate, writes not needed.
        let uncached_equivalent = estimated - cache_read - cache_write
            + per_tok(usage.cache_read + usage.cache_write(), rates.input);

        // A provider-reported figure is authoritative. Failing to interpret
        // it is an error, not a fallback — a billing path must not quietly
        // substitute an estimate for the provider's own number.
        let reported = match usage.reported_cost.as_ref() {
            None => None,
            Some(r) => {
                let currency =
                    iso::find(&r.currency).ok_or_else(|| CostError::UnknownCurrency {
                        code: r.currency.clone(),
                        amount: r.amount,
                    })?;
                let amount = Decimal::try_from(r.amount).map_err(|_| {
                    CostError::UnrepresentableAmount {
                        currency: r.currency.clone(),
                        amount: r.amount,
                    }
                })?;
                Some(Money::from_decimal(amount, currency))
            }
        };

        Ok(Cost {
            currency: self.currency,
            input,
            input_image,
            input_audio,
            input_video,
            input_document,
            images,
            cache_read,
            cache_write,
            output,
            output_audio,
            reasoning,
            estimated,
            uncached_equivalent,
            reported,
        })
    }
}

/// A provider-reported cost was present but could not be interpreted.
/// Deliberately fatal for the pricing pass — see [`PriceSheet::cost`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CostError {
    /// The reported currency code is not a known ISO-4217 code.
    #[error("reported cost {amount} in unknown currency {code:?}")]
    UnknownCurrency { code: String, amount: f64 },
    /// The reported amount is not representable as a decimal (NaN/∞).
    #[error("reported cost {amount} in {currency} is not representable")]
    UnrepresentableAmount { currency: String, amount: f64 },
}

/// The cost of one request (or an aggregate), broken out by bucket.
///
/// All bucket fields are **estimates** from the price sheet, in
/// [`Cost::currency`]. [`Cost::total`] prefers the provider-reported figure
/// when one exists.
#[derive(Debug, Clone, PartialEq)]
pub struct Cost {
    pub currency: &'static Currency,
    pub input: Decimal,
    pub input_image: Decimal,
    pub input_audio: Decimal,
    pub input_video: Decimal,
    pub input_document: Decimal,
    /// Per-image flat fees.
    pub images: Decimal,
    pub cache_read: Decimal,
    pub cache_write: Decimal,
    pub output: Decimal,
    pub output_audio: Decimal,
    pub reasoning: Decimal,
    estimated: Decimal,
    uncached_equivalent: Decimal,
    /// Provider-reported cost, possibly in a different currency.
    pub reported: Option<Amount>,
}

impl Cost {
    /// The authoritative total: the provider-reported cost when present
    /// (in its own currency), otherwise the price-sheet estimate.
    pub fn total(&self) -> Amount {
        match &self.reported {
            Some(r) => *r,
            None => self.estimated(),
        }
    }

    /// The price-sheet estimate, even when a reported cost exists.
    pub fn estimated(&self) -> Amount {
        Money::from_decimal(self.estimated, self.currency)
    }

    /// True when [`Cost::total`] is a provider-reported figure.
    pub fn is_reported(&self) -> bool {
        self.reported.is_some()
    }

    /// How much the prompt cache saved versus running fully uncached:
    /// `hypothetical uncached cost − estimated cost`. Negative values mean
    /// cache writes cost more than reads saved (a cold, write-heavy request).
    pub fn cache_savings(&self) -> Amount {
        Money::from_decimal(self.uncached_equivalent - self.estimated, self.currency)
    }

    /// Bucket amount as [`Money`] in the sheet currency.
    pub fn bucket(&self, amount: Decimal) -> Amount {
        Money::from_decimal(amount, self.currency)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usd_sheet() -> PriceSheet {
        // Anthropic-shaped: $3 in / $0.30 cache read / $3.75 write 5m /
        // $6 write 1h / $15 out.
        PriceSheet::flat(
            iso::USD,
            Rates {
                input: dec("3"),
                cache_read: dec("0.30"),
                cache_write_5m: dec("3.75"),
                cache_write_1h: Some(dec("6")),
                output: dec("15"),
                ..Default::default()
            },
        )
    }

    #[test]
    fn flat_sheet_costs_disjoint_buckets() {
        let u = Usage {
            input: 1_000_000,
            cache_read: 1_000_000,
            cache_write_5m: 1_000_000,
            cache_write_1h: 1_000_000,
            output: 1_000_000,
            reasoning: 1_000_000,
            ..Default::default()
        };
        let c = usd_sheet().cost(&u).unwrap();
        assert_eq!(c.input, dec("3"));
        assert_eq!(c.cache_read, dec("0.30"));
        assert_eq!(c.cache_write, dec("9.75")); // 3.75 + 6
        assert_eq!(c.output, dec("15"));
        assert_eq!(c.reasoning, dec("15")); // defaults to output rate
        assert_eq!(c.total(), Money::from_decimal(dec("43.05"), iso::USD));
    }

    #[test]
    fn cache_savings_reprices_at_input_rate() {
        let u = Usage {
            input: 0,
            cache_read: 1_000_000,
            output: 0,
            ..Default::default()
        };
        let c = usd_sheet().cost(&u).unwrap();
        // Uncached: 1M × $3 = $3; actual: $0.30 → saved $2.70.
        assert_eq!(
            c.cache_savings(),
            Money::from_decimal(dec("2.70"), iso::USD)
        );
    }

    #[test]
    fn tiers_select_by_total_input_including_cache() {
        let sheet = PriceSheet {
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
        };
        // 150k fresh + 100k cached = 250k total input → high tier.
        let u = Usage {
            input: 150_000,
            cache_read: 100_000,
            output: 1000,
            ..Default::default()
        };
        let c = sheet.cost(&u).unwrap();
        assert_eq!(c.input, dec("0.6")); // 150k × $4/1M
        // 100k fresh → low tier.
        let u2 = Usage {
            input: 100_000,
            output: 1000,
            ..Default::default()
        };
        assert_eq!(sheet.cost(&u2).unwrap().input, dec("0.2"));
    }

    #[test]
    fn unknown_reported_currency_is_an_error_not_a_fallback() {
        let u = Usage {
            input: 1_000_000,
            reported_cost: Some(crate::types::ReportedCost {
                amount: 1.0,
                currency: "CREDITS".into(),
            }),
            ..Default::default()
        };
        assert_eq!(
            usd_sheet().cost(&u),
            Err(CostError::UnknownCurrency {
                code: "CREDITS".into(),
                amount: 1.0
            })
        );
    }

    #[test]
    fn reported_cost_wins() {
        let u = Usage {
            input: 1_000_000,
            reported_cost: Some(crate::types::ReportedCost {
                amount: 1.23,
                currency: "USD".into(),
            }),
            ..Default::default()
        };
        let c = usd_sheet().cost(&u).unwrap();
        assert!(c.is_reported());
        assert_eq!(c.total(), Money::from_decimal(dec("1.23"), iso::USD));
        assert_eq!(c.estimated(), Money::from_decimal(dec("3"), iso::USD));
    }
}
