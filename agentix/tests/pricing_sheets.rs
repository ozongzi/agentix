//! Rate assertions for the static price sheets.
//!
//! These pin the values from each vendor's official pricing page (URL and
//! snapshot date in each sheet's header). They exist because price sheets fail
//! *silently*: a wrong rate still produces a well-formed `Cost`, so a stale
//! number is only caught by a test like this.
//!
//! Scope: the providers whose defaults or recent revisions moved, plus the
//! billing quirks that are easy to regress — OpenAI's long-context re-tiering,
//! Anthropic's two cache-read exceptions, and Kimi's separate cache-write fee.

use agentix::Provider;
use agentix::pricing::{Rates, dec, sheets};

/// Resolve the rates a request with `total_input` tokens would be billed at.
fn rates(provider: Provider, model: &str, total_input: u64) -> Rates {
    sheets::builtin(provider, model)
        .unwrap_or_else(|| panic!("no price sheet for {provider:?} / {model:?}"))
        .rates_for(total_input)
        .clone()
}

// ── OpenAI: long-context re-tiering at 272K ─────────────────────────────────

#[test]
fn openai_gpt_6_1_sol_tiers_at_272k() {
    let short = rates(Provider::OpenAI, "gpt-6.1-sol", 272_000);
    assert_eq!(short.input, dec("2"));
    assert_eq!(short.cache_read, dec("0.10")); // 0.05× — a documented exception
    assert_eq!(short.cache_write_5m, dec("2.50"));
    assert_eq!(short.output, dec("10"));

    // One token past the boundary re-tiers the whole request.
    let long = rates(Provider::OpenAI, "gpt-6.1-sol", 272_001);
    assert_eq!(long.input, dec("4"));
    assert_eq!(long.cache_read, dec("0.20"));
    assert_eq!(long.cache_write_5m, dec("5"));
    assert_eq!(long.output, dec("15"));
}

#[test]
fn openai_pre_5_6_models_have_no_cache_write_fee() {
    let gpt55 = rates(Provider::OpenAI, "gpt-5.5", 0);
    assert_eq!(gpt55.input, dec("5"));
    assert_eq!(gpt55.cache_read, dec("0.50"));
    assert_eq!(gpt55.cache_write_5m, dec("0"));
    assert_eq!(
        rates(Provider::OpenAI, "gpt-5.5", 272_001).output,
        dec("45")
    );
}

// ── Anthropic: cache-read exceptions and the cheaper 5 generation ───────────

#[test]
fn anthropic_current_lineup_rates() {
    let fable = rates(Provider::Anthropic, "claude-fable-5-1", 0);
    assert_eq!(fable.input, dec("10"));
    assert_eq!(fable.output, dec("50"));
    assert_eq!(fable.cache_read, dec("0.25")); // 0.025× — documented exception
    assert_eq!(fable.cache_write_5m, dec("12.50"));
    assert_eq!(fable.cache_write_1h, Some(dec("20")));

    let opus = rates(Provider::Anthropic, "claude-opus-5-5", 0);
    assert_eq!(opus.input, dec("4"));
    assert_eq!(opus.output, dec("20"));
    assert_eq!(opus.cache_read, dec("0.20")); // 0.05× — documented exception
    assert_eq!(opus.cache_write_5m, dec("5"));
    assert_eq!(opus.cache_write_1h, Some(dec("8")));

    let sonnet = rates(Provider::Anthropic, "claude-sonnet-5-5", 0);
    assert_eq!(sonnet.input, dec("2"));
    assert_eq!(sonnet.output, dec("10"));
    assert_eq!(sonnet.cache_read, dec("0.20")); // standard 0.1×
}

#[test]
fn anthropic_legacy_generations_keep_their_own_prices() {
    // Opus 4.x was $5 / $25 — the 5 generation is cheaper, so a family-wide
    // match would misprice one of them.
    assert_eq!(
        rates(Provider::Anthropic, "claude-opus-4-8", 0).input,
        dec("5")
    );
    assert_eq!(
        rates(Provider::Anthropic, "claude-sonnet-4-6", 0).input,
        dec("3")
    );
}

#[test]
fn claude_code_aliases_price_at_the_current_generation() {
    assert_eq!(rates(Provider::Anthropic, "opus", 0).input, dec("4"));
    assert_eq!(rates(Provider::Anthropic, "sonnet", 0).input, dec("2"));
    assert_eq!(rates(Provider::Anthropic, "fable", 0).input, dec("10"));
}

// ── Kimi: cache writes are billed separately on K3 ─────────────────────────

#[test]
fn kimi_k3_bills_cache_writes_by_ttl() {
    let k3 = rates(Provider::Kimi, "kimi-k3", 0);
    assert_eq!(k3.input, dec("3"));
    assert_eq!(k3.cache_read, dec("0.30"));
    assert_eq!(k3.cache_write_5m, dec("3")); // 5m tier is the default
    assert_eq!(k3.cache_write_1h, Some(dec("6")));
    assert_eq!(k3.output, dec("15"));
}

#[test]
fn kimi_k2_line_is_cheaper_than_k3() {
    assert_eq!(rates(Provider::Kimi, "kimi-k2.6", 0).input, dec("0.95"));
    assert_eq!(
        rates(Provider::Kimi, "kimi-k2.7-code", 0).input,
        dec("0.95")
    );
    assert_eq!(
        rates(Provider::Kimi, "kimi-k2.7-code-highspeed", 0).input,
        dec("1.90")
    );
}

// ── GLM: 5.3 Flash tiers are paid SKUs, not free ───────────────────────────

#[test]
fn glm_5_3_family_rates() {
    assert_eq!(rates(Provider::Glm, "glm-5.3", 0).input, dec("1.40"));
    assert_eq!(rates(Provider::Glm, "glm-5.3", 0).cache_read, dec("0.26"));
    assert_eq!(rates(Provider::Glm, "glm-5.3", 0).output, dec("4.40"));
    assert_eq!(rates(Provider::Glm, "glm-5.3-flash", 0).input, dec("0.15"));
    assert_eq!(rates(Provider::Glm, "glm-5.3-flashx", 0).input, dec("0.37"));
}

#[test]
fn glm_4_x_flash_stays_free() {
    assert_eq!(rates(Provider::Glm, "glm-4.5-flash", 0).input, dec("0"));
}

// ── MiniMax: M3 tiers at 512k and publishes no cache-write rate ────────────

#[test]
fn minimax_m3_tiers_at_512k_without_a_write_rate() {
    let low = rates(Provider::Minimax, "MiniMax-M3", 512_000);
    assert_eq!(low.input, dec("0.30"));
    assert_eq!(low.cache_read, dec("0.06"));
    assert_eq!(low.cache_write_5m, dec("0"));
    assert_eq!(low.output, dec("1.20"));

    let high = rates(Provider::Minimax, "MiniMax-M3", 512_001);
    assert_eq!(high.input, dec("0.60"));
    assert_eq!(high.cache_read, dec("0.12"));
    assert_eq!(high.output, dec("2.40"));
}

// ── DeepSeek: peak rates (off-peak is half) ────────────────────────────────

#[test]
fn deepseek_flash_peak_rates() {
    let flash = rates(Provider::DeepSeek, "deepseek-flash", 0);
    assert_eq!(flash.input, dec("0.30"));
    assert_eq!(flash.cache_read, dec("0.006"));
    assert_eq!(flash.output, dec("1.20"));

    // Retired flash names still route to the Flash price.
    assert_eq!(
        rates(Provider::DeepSeek, "deepseek-v4-flash", 0).input,
        dec("0.30")
    );
    assert_eq!(
        rates(Provider::DeepSeek, "deepseek-v4-pro", 0).input,
        dec("1.32")
    );
}

// ── MiMo: ultraspeed is the expensive SKU ──────────────────────────────────

#[test]
fn mimo_v2_6_sku_ladder() {
    assert_eq!(rates(Provider::Mimo, "mimo-v2.6-flash", 0).input, dec("1"));
    assert_eq!(rates(Provider::Mimo, "mimo-v2.6-pro", 0).input, dec("3"));
    assert_eq!(
        rates(Provider::Mimo, "mimo-v2.6-pro-ultraspeed", 0).input,
        dec("30")
    );
}

// ── Gemini: 3.8 Flash is flat, on promotional pricing ──────────────────────

#[test]
fn gemini_3_8_flash_is_flat_and_promotional() {
    let flash = rates(Provider::Gemini, "gemini-3.8-flash", 0);
    assert_eq!(flash.input, dec("0.75"));
    assert_eq!(flash.cache_read, dec("0.075"));
    assert_eq!(flash.output, dec("3.75"));

    // No length tier: a 2M-token prompt bills at the same rate.
    assert_eq!(
        rates(Provider::Gemini, "gemini-3.8-flash", 2_000_000).input,
        dec("0.75")
    );
}

// ── Grok: 4.7 doubles above a 200k prompt ─────────────────────────────────

#[test]
fn grok_4_7_long_context_tier() {
    let short = rates(Provider::Grok, "grok-4.7", 200_000);
    assert_eq!(short.input, dec("2"));
    assert_eq!(short.cache_read, dec("0.50"));
    assert_eq!(short.output, dec("6"));

    let long = rates(Provider::Grok, "grok-4.7", 200_001);
    assert_eq!(long.input, dec("4"));
    assert_eq!(long.output, dec("12"));
}
