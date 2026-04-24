//! Cross-provider reasoning control via `ReasoningEffort`.
//!
//! Demonstrates four states:
//!   1. **Unset** — use the provider's own default (DeepSeek defaults to thinking on, Anthropic to thinking off).
//!   2. `None` — explicitly disable thinking; sampling params like `temperature` pass through.
//!   3. `High` — engage thinking at high effort.
//!   4. `Max` — maximum effort (maps to the provider's strongest setting).
//!
//! Same code works against DeepSeek (`DEEPSEEK_API_KEY`) or Anthropic (`ANTHROPIC_API_KEY`);
//! each provider translates `ReasoningEffort` to its own wire format:
//!   - DeepSeek:  `thinking.type` + `reasoning_effort: "high" | "max"`.
//!   - Anthropic: `thinking.type: "adaptive" | "disabled"` + `output_config.effort`.
//!   - OpenAI / Grok / others: currently ignore the field (future work).

use agentix::{LlmEvent, Provider, ReasoningEffort, Request};
use futures::StreamExt;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (provider, api_key, model) = pick_provider()?;
    let http = reqwest::Client::new();

    let prompt = "What is 17 * 23? Think briefly, then give the answer.";

    for scenario in [
        Scenario::Unset,
        Scenario::Disabled,
        Scenario::HighEffort,
        Scenario::MaxEffort,
    ] {
        println!("\n═══ {} ═══", scenario.label());

        let mut req = Request::new(provider, &api_key).model(&model).user(prompt);
        match scenario {
            Scenario::Unset => { /* leave reasoning_effort unset */ }
            Scenario::Disabled => {
                req = req.reasoning_effort(ReasoningEffort::None).temperature(0.3); // allowed: thinking is off
            }
            Scenario::HighEffort => {
                req = req.reasoning_effort(ReasoningEffort::High);
            }
            Scenario::MaxEffort => {
                req = req.reasoning_effort(ReasoningEffort::Max);
            }
        }

        let mut stream = req.stream(&http).await?;
        while let Some(event) = stream.next().await {
            match event {
                LlmEvent::Reasoning(r) => print!("\x1b[36m{r}\x1b[0m"), // cyan
                LlmEvent::Token(t) => print!("{t}"),
                LlmEvent::Error(e) => eprintln!("\nError: {e}"),
                LlmEvent::Done => break,
                _ => {}
            }
        }
        println!();
    }

    Ok(())
}

#[derive(Copy, Clone)]
enum Scenario {
    Unset,
    Disabled,
    HighEffort,
    MaxEffort,
}

impl Scenario {
    fn label(self) -> &'static str {
        match self {
            Scenario::Unset => "default (reasoning_effort unset)",
            Scenario::Disabled => "reasoning_effort(None) — thinking off",
            Scenario::HighEffort => "reasoning_effort(High)",
            Scenario::MaxEffort => "reasoning_effort(Max)",
        }
    }
}

fn pick_provider() -> Result<(Provider, String, String), Box<dyn std::error::Error>> {
    if let Ok(k) = env::var("DEEPSEEK_API_KEY") {
        return Ok((Provider::DeepSeek, k, "deepseek-v4-pro".into()));
    }
    if let Ok(k) = env::var("ANTHROPIC_API_KEY") {
        return Ok((Provider::Anthropic, k, "claude-opus-4-7".into()));
    }
    Err("set DEEPSEEK_API_KEY or ANTHROPIC_API_KEY to run this example".into())
}
