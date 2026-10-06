//! Every provider's default model must resolve to a built-in price sheet.
//!
//! `Request::new(provider, key)` seeds the provider's default model, and cost
//! estimation runs against the static sheet for that (provider, model) pair.
//! A default that resolves to `None` silently loses cost estimation — the
//! request succeeds, but `Cost` can never be computed from a sheet. This test
//! pins that invariant so a future default-model bump cannot drop it.
//!
//! OpenRouter is deliberately exempt: it has no static sheet and relies on the
//! provider-reported cost or the dynamic catalog
//! ([`agentix::pricing::sheets::openrouter::sheet_from_catalog`]).

use agentix::Provider;
use agentix::pricing::sheets;

/// Providers whose default model must be priced by a built-in sheet.
#[test]
fn default_models_resolve_to_a_price_sheet() {
    let providers = [
        Provider::DeepSeek,
        Provider::OpenAI,
        Provider::Anthropic,
        Provider::Gemini,
        Provider::Kimi,
        Provider::Glm,
        Provider::Minimax,
        Provider::Mimo,
        Provider::Grok,
    ];
    for provider in providers {
        let model = provider.default_model();
        assert!(
            sheets::builtin(provider, model).is_some(),
            "{provider:?} default model {model:?} has no built-in price sheet"
        );
    }
}

#[test]
fn openrouter_default_has_no_static_sheet() {
    let model = Provider::OpenRouter.default_model();
    assert!(sheets::builtin(Provider::OpenRouter, model).is_none());
}

#[cfg(feature = "claude-code")]
#[test]
fn claude_code_default_alias_resolves_to_anthropic_sheet() {
    let model = Provider::ClaudeCode.default_model();
    assert!(
        sheets::builtin(Provider::ClaudeCode, model).is_some(),
        "ClaudeCode default alias {model:?} has no price sheet"
    );
}

#[cfg(feature = "codex")]
#[test]
fn codex_default_resolves_to_openai_sheet() {
    let model = Provider::Codex.default_model();
    assert!(
        sheets::builtin(Provider::Codex, model).is_some(),
        "Codex default model {model:?} has no price sheet"
    );
}
