use crate::request::Message;

pub(crate) fn prepare_history(messages: Vec<Message>) -> Vec<Message> {
    messages.into_iter().map(|m| match m {
        Message::Assistant { content, reasoning, tool_calls } => {
            let has_tools = !tool_calls.is_empty();
            Message::Assistant {
                content,
                reasoning: if has_tools { Some(reasoning.unwrap_or_default()) } else { None },
                tool_calls,
            }
        }
        other => other,
    }).collect()
}
