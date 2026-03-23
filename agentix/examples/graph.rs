//! Multi-agent pipeline using Graph, PromptTemplate, OutputParser, and middleware.
//!
//! Pipeline:
//!   prompt_template  →  scorer_agent  →  output_parser
//!
//! 1. `PromptTemplate` wraps raw input in a scoring instruction.
//! 2. `scorer_agent` (DeepSeek) responds with JSON: {"score": N}.
//! 3. `OutputParser` extracts the numeric score.
//! 4. A middleware logs every message crossing each edge.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example graph

use agentix::{Graph, Msg, Node, OutputParser, PromptTemplate};
use futures::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");

    // ── Nodes ─────────────────────────────────────────────────────────────────

    let prompt = PromptTemplate::new(
        "Rate the following product review on a scale of 0 to 10. \
         Respond with ONLY valid JSON in this exact format: {{\"score\": <number>}}\n\
         Review: {input}",
    );

    let scorer = agentix::deepseek(&key)
        .model("deepseek-chat")
        .system_prompt("You are a sentiment scorer. Respond only with JSON: {\"score\": N}.")
        .max_tokens(64);

    let parser = OutputParser::new(|s| {
        serde_json::from_str::<serde_json::Value>(s.trim())
            .ok()
            .and_then(|v| v["score"].as_f64().map(|n| format!("{n:.1}")))
            .unwrap_or_else(|| format!("(could not parse: {s})"))
    });

    // ── Wire up ───────────────────────────────────────────────────────────────

    Graph::new()
        .middleware(|msg| {
            if let Msg::User(ref parts) = msg {
                let text: String = parts.iter()
                    .filter_map(|p| if let agentix::UserContent::Text(t) = p { Some(t.as_str()) } else { None })
                    .collect::<Vec<_>>().join(" ");
                eprintln!("[edge →] {}", &text[..text.len().min(80)]);
            }
            Some(msg)
        })
        .edge(&prompt, &scorer)
        .edge(&scorer, &parser);

    // ── Run ───────────────────────────────────────────────────────────────────

    let reviews = [
        "Absolutely love this product! Best purchase I've made all year.",
        "Terrible quality. Broke after one use. Complete waste of money.",
        "It's okay. Does the job but nothing special.",
    ];

    for review in reviews {
        println!("Review: \"{review}\"");

        // Subscribe before sending so we don't miss any events.
        let mut stream = Box::pin(parser.output().subscribe_assembled());
        prompt.input().send(Msg::User(vec![review.into()])).await?;

        // Drain the parser's assembled output.
        while let Some(msg) = stream.next().await {
            match msg {
                Msg::User(parts) => {
                    let score = parts.into_iter()
                        .filter_map(|p| if let agentix::UserContent::Text(t) = p { Some(t) } else { None })
                        .collect::<String>();
                    println!("Score: {score}\n");
                    break;
                }
                Msg::Done => break,
                _ => {}
            }
        }
    }

    // ── Raw streaming view from scorer ────────────────────────────────────────
    println!("--- Watching scorer raw output for one more review ---\n");

    let mut rx = scorer.subscribe();
    prompt.input().send(Msg::User(vec!["Exceptional! Exceeded all expectations.".into()])).await?;

    while let Ok(msg) = rx.recv().await {
        match msg {
            Msg::Token(t) => { print!("{t}"); std::io::stdout().flush().ok(); }
            Msg::Done     => { println!(); break; }
            _             => {}
        }
    }

    Ok(())
}
