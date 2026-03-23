//! Multi-agent pipeline using stream chaining.
//!
//! Pipeline:
//!   input_stream  →  prompt_node  →  scorer_agent  →  output_parser
//!
//! 1. `PromptNode` wraps raw input in a scoring instruction.
//! 2. `scorer_agent` (DeepSeek) responds with JSON: {"score": N}.
//! 3. `OutputParserNode` extracts the numeric score.
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... cargo run --example graph

use agentix::{AgentEvent, Node, PromptNode};
use futures::StreamExt;
use futures::stream::BoxStream;
use tokio::sync::mpsc;

// ── Custom Output Parser Node ────────────────────────────────────────────────

struct OutputParserNode;

impl Node for OutputParserNode {
    type Input = AgentEvent;
    type Output = String;

    fn run(self, input: BoxStream<'static, Self::Input>) -> BoxStream<'static, Self::Output> {
        let mut full_text = String::new();
        async_stream::stream! {
            for await event in input {
                match event {
                    AgentEvent::Token(t) => full_text.push_str(&t),
                    AgentEvent::Done => {
                        let score = serde_json::from_str::<serde_json::Value>(full_text.trim())
                            .ok()
                            .and_then(|v| v["score"].as_f64().map(|n| format!("{n:.1}")))
                            .unwrap_or_else(|| format!("(could not parse: {full_text})"));
                        yield score;
                        full_text.clear();
                    }
                    _ => {}
                }
            }
        }.boxed()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = std::env::var("DEEPSEEK_API_KEY").expect("DEEPSEEK_API_KEY must be set");

    // ── Nodes ─────────────────────────────────────────────────────────────────

    let prompt_node = PromptNode::new(
        "Rate the following product review on a scale of 0 to 10. \
         Respond with ONLY valid JSON in this exact format: {{\"score\": <number>}}\n\
         Review: {input}",
    );

    let scorer = agentix::deepseek(&key)
        .model("deepseek-chat")
        .system_prompt("You are a sentiment scorer. Respond only with JSON: {\"score\": N}.")
        .max_tokens(64);

    let parser_node = OutputParserNode;

    // ── Wire up ───────────────────────────────────────────────────────────────

    let (tx, rx) = mpsc::channel::<String>(64);
    
    // Chain: rx (String) -> prompt_node -> AgentInput -> scorer -> AgentEvent -> parser_node -> String
    let prompt_stream = prompt_node.run(tokio_stream::wrappers::ReceiverStream::new(rx).boxed());
    let agent_stream = scorer.run(prompt_stream);
    let mut final_output = parser_node.run(agent_stream);

    // ── Run ───────────────────────────────────────────────────────────────────

    let reviews = [
        "Absolutely love this product! Best purchase I've made all year.",
        "Terrible quality. Broke after one use. Complete waste of money.",
        "It's okay. Does the job but nothing special.",
    ];

    for review in reviews {
        println!("Review: \"{review}\"");
        tx.send(review.to_string()).await?;

        if let Some(score) = final_output.next().await {
            println!("Score: {score}\n");
        }
    }

    Ok(())
}
