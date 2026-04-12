// OpenRouter uses the same response format as OpenAI
pub use crate::raw::openai::response::{
    StreamChunk, ChunkChoice, Delta, DeltaToolCall, DeltaFunctionCall, Usage,
    CompleteResponse, CompleteChoice, CompleteMessage, CompleteToolCall, CompleteFunctionCall,
};
