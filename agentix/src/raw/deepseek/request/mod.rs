pub mod chat_completion;
pub mod message;
pub mod model;
pub mod response_format;
pub mod stop;
pub mod stream_options;
pub mod thinking;
pub mod tool;
pub mod tool_choice;

pub use chat_completion::ChatCompletionRequest;
pub use message::{FunctionCall, Message, Role, ToolCall, ToolType};
pub use model::Model;
pub use response_format::{ResponseFormat, ResponseFormatType};
pub use stop::Stop;
pub use stream_options::StreamOptions;
pub use thinking::{Thinking, ThinkingType};
pub use tool::{Function, Tool};
pub use tool_choice::{FunctionName, ToolChoice, ToolChoiceObject, ToolChoiceType};

impl From<crate::request::Request> for ChatCompletionRequest {
    fn from(req: crate::request::Request) -> Self {
        use crate::raw::shared::ToolKind;
        use crate::request::{Message as AgentMsg, UserContent};

        let mut raw_messages: Vec<Message> = Vec::new();

        if let Some(sys) = req.system_message
            && !sys.is_empty()
        {
            raw_messages.push(Message::system(&sys));
        }

        for msg in req.messages {
            match msg {
                AgentMsg::User(parts) => {
                    let text = parts
                        .into_iter()
                        .filter_map(|p| match p {
                            UserContent::Text(t) => Some(t),
                            UserContent::Image(_) => None,
                        })
                        .collect::<Vec<_>>()
                        .join("");
                    raw_messages.push(Message::user(&text));
                }
                AgentMsg::Assistant {
                    content,
                    reasoning,
                    tool_calls,
                } => {
                    raw_messages.push(Message {
                        role: Role::Assistant,
                        content,
                        reasoning_content: reasoning,
                        tool_calls: if tool_calls.is_empty() {
                            None
                        } else {
                            Some(
                                tool_calls
                                    .into_iter()
                                    .map(|tc| ToolCall {
                                        id: tc.id,
                                        r#type: ToolKind::Function,
                                        function: FunctionCall {
                                            name: tc.name,
                                            arguments: tc.arguments,
                                        },
                                    })
                                    .collect(),
                            )
                        },
                        ..Default::default()
                    });
                }
                AgentMsg::ToolResult { call_id, content } => {
                    raw_messages.push(Message {
                        role: Role::Tool,
                        content: Some(content),
                        tool_call_id: Some(call_id),
                        ..Default::default()
                    });
                }
            }
        }

        ChatCompletionRequest {
            messages: raw_messages,
            model: Model::Custom(req.model),
            tools: req.tools,
            tool_choice: req.tool_choice.map(Into::into),
            stream: Some(req.stream),
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            response_format: req.response_format.map(Into::into),
            extra_body: req.extra_body,
            ..Default::default()
        }
    }
}
