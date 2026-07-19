//! Built-in static price sheets, one file per provider.
//!
//! Each file is a snapshot of that provider's official pricing page (URL and
//! date in the file header) — prices drift, so treat these as defaults and
//! prefer a provider-reported cost ([`crate::types::Usage::reported_cost`])
//! or the OpenRouter dynamic catalog ([`openrouter::sheet_from_catalog`])
//! when authority matters.

pub mod anthropic;
pub mod deepseek;
pub mod gemini;
pub mod glm;
pub mod grok;
pub mod kimi;
pub mod mimo;
pub mod minimax;
pub mod openai;
pub mod openrouter;

use crate::pricing::PriceSheet;
use crate::request::Provider;

/// Look up the built-in sheet for a provider + model. Pure routing — the
/// per-provider logic lives in each provider's own file.
///
/// Claude Code resolves to the Anthropic sheets (informational under a
/// subscription); Codex resolves to the OpenAI sheets (API-key auth rates —
/// ChatGPT-plan auth has no marginal cost). OpenRouter has no static sheet:
/// use [`openrouter::sheet_from_catalog`] or rely on its reported cost.
pub fn builtin(provider: Provider, model: &str) -> Option<PriceSheet> {
    match provider {
        Provider::Anthropic => anthropic::sheet(model),
        Provider::OpenAI => openai::sheet(model),
        Provider::DeepSeek => deepseek::sheet(model),
        Provider::Gemini => gemini::sheet(model),
        Provider::Kimi => kimi::sheet(model),
        Provider::Glm => glm::sheet(model),
        Provider::Grok => grok::sheet(model),
        Provider::Minimax => minimax::sheet(model),
        Provider::Mimo => mimo::sheet(model),
        Provider::OpenRouter => None,
        #[cfg(feature = "claude-code")]
        Provider::ClaudeCode => {
            // CLI model aliases ("sonnet", "opus") hit the substring match.
            anthropic::sheet(model)
        }
        #[cfg(feature = "codex")]
        Provider::Codex => openai::sheet(model),
    }
}
