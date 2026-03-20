//! Raw API data structures
//!
//! This module provides the raw data structures that map directly to the DeepSeek API's JSON schema.
//! These types are intended for users who need precise control over request and response payloads.
//!
//! # Module layout
//!
//! - [`deepseek`]
//! - [`anthropic`]
//! - [`gemini`]
//! - [`openai`]
//!
//! # Main types
//!
//! ## Request types
//!
//! - [`ChatCompletionRequest`]
//! - [`Message`]
//! - [`Model`]
//! - [`Tool`]
//! - [`ResponseFormat`]
//! - [`Thinking`]
//! - [`Stop`]
//! - [`StreamOptions`]
//!
//! ## Response types
//!
//! - [`ChatCompletionResponse`]
//! - [`ChatCompletionChunk`]
//! - [`Choice`]
//! - [`Usage`]
//!
//! # Use cases
//!
//! Use these raw structures when you need one of the following:
//!
//! 1. Full control over the API request payload (set every field explicitly).
//! 2. Advanced features such as tool/function calls or reasoning modes.
//! 3. Performance optimizations where avoiding builder overhead matters.
//! 4. Integration with existing code that expects direct Serde serialization/deserialization.
//!
//! # Examples
//!
//! The following examples illustrate common usage patterns with the raw types.
//!
//! ```rust
//! use agentix::raw::{ChatCompletionRequest, Message, Model, Role};
//! use serde_json::json;
//!
//! fn main() {
//!     // Example 1: Basic chat completion request
//!     let basic_request = ChatCompletionRequest {
//!         messages: vec![
//!             Message::new(Role::System, "You are a helpful assistant."),
//!             Message::new(Role::User, "What is the capital of France?"),
//!         ],
//!         model: Model::DeepseekChat,
//!         max_tokens: Some(100),
//!         temperature: Some(0.7),
//!         stream: Some(false),
//!         ..Default::default()
//!     };
//!
//!     println!("Basic request JSON:");
//!     let json = serde_json::to_string_pretty(&basic_request).unwrap();
//!     println!("{}\n", json);
//!
//!     // Example 2: Request with a tool/function definition
//!     use agentix::raw::{ToolDefinition, ToolChoice, ToolChoiceMode, ToolKind, FunctionDefinition};
//!
//!     let tool_request = ChatCompletionRequest {
//!         messages: vec![Message::new(Role::User, "What's the weather like in Tokyo?")],
//!         model: Model::DeepseekChat,
//!         max_tokens: Some(200),
//!         temperature: Some(0.8),
//!         stream: Some(false),
//!         tools: Some(vec![ToolDefinition {
//!             kind: ToolKind::Function,
//!             function: FunctionDefinition {
//!                 name: "get_weather".to_string(),
//!                 description: Some("Get the current weather for a location".to_string()),
//!                 parameters: json!({
//!                     "type": "object",
//!                     "properties": {
//!                         "location": {
//!                             "type": "string",
//!                             "description": "The city and country, e.g. Tokyo, Japan"
//!                         }
//!                     },
//!                     "required": ["location"]
//!                 }),
//!                 strict: Some(true),
//!             },
//!         }]),
//!         tool_choice: Some(ToolChoice::String(ToolChoiceMode::Auto)),
//!         ..Default::default()
//!     };
//!
//!     println!("Tool request JSON:");
//!     let json = serde_json::to_string_pretty(&tool_request).unwrap();
//!     println!("{}\n", json);
//!
//!     // Example 3: Using the Deepseek Reasoner model
//!     use agentix::raw::{Thinking, ThinkingType};
//!
//!     let reasoner_request = ChatCompletionRequest {
//!         messages: vec![
//!             Message::new(Role::System, "You are a helpful assistant that explains your reasoning."),
//!             Message::new(Role::User, "Solve this math problem: What is 15% of 200?"),
//!         ],
//!         model: Model::DeepseekReasoner,
//!         thinking: Some(Thinking {
//!             r#type: ThinkingType::Enabled,
//!         }),
//!         max_tokens: Some(150),
//!         temperature: Some(0.3),
//!         stream: Some(false),
//!         ..Default::default()
//!     };
//!
//!     println!("Reasoner request JSON:");
//!     let json = serde_json::to_string_pretty(&reasoner_request).unwrap();
//!     println!("{}\n", json);
//!
//!     // Example 4: JSON response format
//!     use agentix::raw::{ResponseFormat, ResponseFormatKind};
//!
//!     let json_request = ChatCompletionRequest {
//!         messages: vec![
//!             Message::new(Role::System, "You are a helpful assistant that always responds in valid JSON format."),
//!             Message::new(Role::User, "Give me information about Paris in JSON format with fields: name, country, population, and landmarks."),
//!         ],
//!         model: Model::DeepseekChat,
//!         max_tokens: Some(200),
//!         temperature: Some(0.5),
//!         stream: Some(false),
//!         response_format: Some(ResponseFormat {
//!             kind: ResponseFormatKind::JsonObject,
//!         }),
//!         ..Default::default()
//!     };
//!
//!     println!("JSON mode request:");
//!     let json = serde_json::to_string_pretty(&json_request).unwrap();
//!     println!("{}\n", json);
//! }
//! ```
//!
//! # Serialization / Deserialization
//!
//! All structs implement `Serialize` and `Deserialize` and can be used with `serde_json` directly:
//!
//! ```rust
//! use agentix::raw::{ChatCompletionRequest, Message, Model, Role};
//!
//! let request = ChatCompletionRequest {
//!     messages: vec![Message::new(Role::User, "Hello")],
//!     model: Model::DeepseekChat,
//!     ..Default::default()
//! };
//! let json_string = serde_json::to_string(&request).unwrap();
//! let parsed_request: ChatCompletionRequest = serde_json::from_str(&json_string).unwrap();
//! ```
//!
//! # Relationship to higher-level APIs
//!
//! The higher-level APIs (for example, the builder-style `Request` types) are implemented on top
//! of these raw structures and provide ergonomic builders and validation. If you prefer an easier
//! or more opinionated interface, consider using the higher-level APIs instead.
pub mod anthropic;
pub mod deepseek;
pub mod gemini;
pub mod openai;
pub mod shared;

pub use deepseek::*;
pub use shared::{FunctionDefinition, FunctionName, ResponseFormat, ResponseFormatKind, ToolChoice, ToolChoiceFunction, ToolChoiceMode, ToolDefinition, ToolKind};
