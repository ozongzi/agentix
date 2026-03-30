//! Example 08: Structured output with JSON Schema
//!
//! `Request::json_schema()` constrains the model to emit JSON that matches a
//! Rust struct derived with `schemars::JsonSchema`. The output can then be
//! deserialized directly with `serde_json`.
//!
//! Run with:
//!   OPENAI_API_KEY=sk-... cargo run --example 08_structured_output

use agentix::Request;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Output schema ─────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct MovieReview {
    /// Title of the movie
    title: String,
    /// Year the movie was released
    year: u16,
    /// Overall rating out of 10
    rating: f32,
    /// One-sentence summary
    summary: String,
    /// List of strengths
    pros: Vec<String>,
    /// List of weaknesses
    cons: Vec<String>,
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("Set OPENAI_API_KEY");

    let http = reqwest::Client::new();

    // Generate the JSON Schema from the Rust type and pass it to the request.
    let schema = serde_json::to_value(schemars::schema_for!(MovieReview))?;

    let response = Request::openai(api_key)
        .model("gpt-4o-mini")
        .system_prompt("You are a film critic. Always respond in the requested JSON format.")
        .user("Review the movie Inception (2010).")
        .json_schema("movie_review", schema, true)
        .complete(&http)
        .await?;

    let json_str = response.content.unwrap_or_default();
    let review: MovieReview = serde_json::from_str(&json_str)?;

    println!("Title:   {} ({})", review.title, review.year);
    println!("Rating:  {}/10", review.rating);
    println!("Summary: {}", review.summary);
    println!("\nPros:");
    for p in &review.pros  { println!("  + {p}"); }
    println!("\nCons:");
    for c in &review.cons  { println!("  - {c}"); }

    Ok(())
}
