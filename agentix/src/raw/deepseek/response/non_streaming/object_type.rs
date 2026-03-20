use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub enum ObjectType {
    #[serde(rename = "chat.completion")]
    ChatCompletion,
}
