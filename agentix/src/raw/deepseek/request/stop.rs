use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    String(String),
    Array(Vec<String>),
}
