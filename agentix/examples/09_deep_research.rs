//! Example 09: Multi-agent Deep Research Pipeline
//!
//! A fixed pipeline of four agents:
//!
//!   QueryAgent   → decompose query into sub-questions (structured output)
//!       ↓ (concurrent)
//!   SearchAgent × N → each sub-question: search + read via Tavily MCP
//!       ↓ (collect)
//!   ReasonAgent  → synthesize all findings
//!       ↓
//!   WriterAgent  → produce a Markdown report saved to disk
//!
//! Requirements:
//!   - Tavily MCP server:  npx -y tavily-mcp
//!     (needs TAVILY_API_KEY in env, the MCP server reads it automatically)
//!   - LLM key: DEEPSEEK_API_KEY or OPENAI_API_KEY
//!
//! Run with:
//!   DEEPSEEK_API_KEY=sk-... TAVILY_API_KEY=tvly-... \
//!     cargo run --example 09_deep_research --features mcp -- "Rust async runtime internals"

#[cfg(feature = "mcp")]
mod deep_research {
    use agentix::{McpTool, Message, Request, Tool, ToolBundle, UserContent, agent_turns, tool};
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use std::sync::Arc;

    // ── Writer tool ───────────────────────────────────────────────────────────────

    struct WriterTools;

    #[tool]
    impl Tool for WriterTools {
        /// Save the final research report to a Markdown file.
        /// filename: output filename (e.g. "report.md")
        /// content: full Markdown content of the report
        async fn save_report(&self, filename: String, content: String) -> String {
            match std::fs::write(&filename, &content) {
                Ok(()) => format!("Report saved to {filename} ({} bytes)", content.len()),
                Err(e) => format!("Failed to save: {e}"),
            }
        }
    }

    // ── Structured output schema for QueryAgent ───────────────────────────────────

    #[derive(Debug, Deserialize, Serialize, JsonSchema)]
    struct SubQuestions {
        /// 3-5 focused sub-questions that together cover the research topic
        questions: Vec<String>,
    }

    // ── Stage 1: QueryAgent — decompose into sub-questions ────────────────────────

    async fn run_query_agent(http: &reqwest::Client, base: &Request, query: &str) -> Vec<String> {
        eprintln!("\n╔══ Stage 1: QueryAgent ══════════════════════════════");
        eprintln!("║  Topic: {query}");

        let schema = serde_json::to_value(schemars::schema_for!(SubQuestions)).unwrap();
        let request = base
            .clone()
            .system_prompt(
                "You are a research planner. Given a research topic, decompose it into \
                 3-5 focused, non-overlapping sub-questions that together fully cover the topic. \
                 Respond ONLY with the JSON structure requested.",
            )
            .json_schema("sub_questions", schema, true);

        let response = request
            .user(format!("Research topic: {query}"))
            .complete(http)
            .await
            .unwrap();
        let parsed: SubQuestions = response.json().unwrap_or(SubQuestions {
            questions: vec![query.to_string()],
        });

        eprintln!("║  Sub-questions:");
        for (i, q) in parsed.questions.iter().enumerate() {
            eprintln!("║    {}. {q}", i + 1);
        }
        eprintln!("╚══════════════════════════════════════════════════════");
        parsed.questions
    }

    // ── Stage 2: SearchAgent — one per sub-question, run concurrently ─────────────

    async fn run_search_agent(
        http: reqwest::Client,
        base: Request,
        tavily: Arc<McpTool>,
        question: String,
        idx: usize,
    ) -> String {
        eprintln!("\n╔══ Stage 2.{idx}: SearchAgent ═══════════════════════════");
        eprintln!("║  Question: {question}");

        let tools = ToolBundle::default() + (*tavily).clone();
        let request = base.system_prompt(
            "You are a research agent. Use the available search tools to find accurate, \
             up-to-date information. Search for the question, read the most relevant pages, \
             then summarize the key findings in 3-5 bullet points. Be factual and concise.",
        );
        let history = vec![Message::User(vec![UserContent::Text(format!(
            "Research question: {question}\n\nSearch for this and summarize the key findings."
        ))])];

        let result = agent_turns(tools, http, request, history, Some(25_000)).last_content().await;
        eprintln!("\n╚══════════════════════════════════════════════════════");
        result
    }

    // ── Stage 3: ReasonAgent — synthesize all findings ────────────────────────────

    async fn run_reason_agent(
        http: &reqwest::Client,
        base: &Request,
        questions: &[String],
        findings: &[String],
    ) -> String {
        eprintln!("\n╔══ Stage 3: ReasonAgent ═════════════════════════════");

        let mut context = String::new();
        for (i, (q, f)) in questions.iter().zip(findings.iter()).enumerate() {
            context.push_str(&format!("\n## Sub-question {}\n{q}\n\n### Findings\n{f}\n", i + 1));
        }

        let request = base.clone().system_prompt(
            "You are a research synthesizer. Given findings from multiple research agents, \
             identify key themes, connections, contradictions, and gaps. Produce a structured \
             analysis that will serve as the foundation for a final report.",
        );
        let history = vec![Message::User(vec![UserContent::Text(format!(
            "Synthesize these research findings into a coherent analysis:\n{context}"
        ))])];

        let result = agent_turns(ToolBundle::default(), http.clone(), request, history, Some(25_000)).last_content().await;
        eprintln!("\n╚══════════════════════════════════════════════════════");
        result
    }

    // ── Stage 4: WriterAgent — produce the final Markdown report ──────────────────

    async fn run_writer_agent(
        http: &reqwest::Client,
        base: &Request,
        topic: &str,
        analysis: &str,
    ) -> String {
        eprintln!("\n╔══ Stage 4: WriterAgent ═════════════════════════════");

        let tools = ToolBundle::default() + WriterTools;
        let request = base.clone().system_prompt(
            "You are a technical writer. Given a research analysis, write a comprehensive, \
             well-structured Markdown report with: an executive summary, detailed sections \
             for each key topic, and a conclusion. Then save it using the save_report tool \
             with filename 'research_report.md'.",
        );
        let history = vec![Message::User(vec![UserContent::Text(format!(
            "Topic: {topic}\n\nAnalysis to turn into a report:\n{analysis}"
        ))])];

        let result = agent_turns(tools, http.clone(), request, history, Some(25_000)).last_content().await;
        eprintln!("\n╚══════════════════════════════════════════════════════");
        result
    }

    // ── Pipeline entry point ──────────────────────────────────────────────────────

    pub async fn run(query: &str) -> Result<(), Box<dyn std::error::Error>> {
        let http = reqwest::Client::new();

        let base_request = if let Ok(k) = std::env::var("DEEPSEEK_API_KEY") {
            Request::deepseek(k)
        } else if let Ok(k) = std::env::var("OPENAI_API_KEY") {
            Request::openai(k)
        } else {
            panic!("Set DEEPSEEK_API_KEY or OPENAI_API_KEY");
        };

        // Connect Tavily MCP once, share via Arc across concurrent SearchAgents
        eprintln!("Connecting to Tavily MCP...");
        let tavily = Arc::new(
            McpTool::stdio("npx", &["-y", "tavily-mcp"])
                .await
                .expect("Failed to start Tavily MCP (is TAVILY_API_KEY set? is npx available?)"),
        );
        eprintln!(
            "Tavily MCP ready ({} tools)\n",
            tavily.raw_tools().len()
        );

        // Stage 1: decompose
        let sub_questions = run_query_agent(&http, &base_request, query).await;

        // Stage 2: concurrent search
        let search_futures: Vec<_> = sub_questions
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, q)| {
                run_search_agent(
                    http.clone(),
                    base_request.clone(),
                    Arc::clone(&tavily),
                    q,
                    i + 1,
                )
            })
            .collect();

        let findings = futures::future::join_all(search_futures).await;

        // Stage 3: reason
        let analysis = run_reason_agent(&http, &base_request, &sub_questions, &findings).await;

        // Stage 4: write report
        run_writer_agent(&http, &base_request, query, &analysis).await;

        eprintln!("\nPipeline complete. Check research_report.md");
        Ok(())
    }
}

#[cfg(feature = "mcp")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let query = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Rust async runtime internals".to_string());

    println!("Deep Research Pipeline");
    println!("Topic: {query}\n");

    deep_research::run(&query).await
}

#[cfg(not(feature = "mcp"))]
fn main() {
    println!("Run with the `mcp` feature:");
    println!(
        "  cargo run --example 09_deep_research --features mcp -- \"your research topic\""
    );
}
