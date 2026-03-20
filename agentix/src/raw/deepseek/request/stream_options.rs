use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}
