use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The model to use for a chat completion request.
///
/// Use [`Model::DeepseekChat`] or [`Model::DeepseekReasoner`] for the standard
/// DeepSeek models, or [`Model::Custom`] to pass any model string directly —
/// useful for OpenAI-compatible providers or future DeepSeek models that have
/// not yet been added as named variants.
///
/// # Examples
///
/// ```
/// use agentix::raw::Model;
///
/// let m = Model::DeepseekChat;
/// let m = Model::Custom("gpt-4o".to_string());
/// ```
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum Model {
    /// `deepseek-chat` — fast, general-purpose DeepSeek model.
    #[default]
    DeepseekChat,
    /// `deepseek-reasoner` — DeepSeek model optimised for deep reasoning.
    DeepseekReasoner,
    /// Any other model identifier, passed through as-is.
    Custom(String),
}

impl Model {
    /// Return the model identifier string as it will appear in the API request.
    pub fn as_str(&self) -> &str {
        match self {
            Model::DeepseekChat => "deepseek-chat",
            Model::DeepseekReasoner => "deepseek-reasoner",
            Model::Custom(s) => s.as_str(),
        }
    }
}

impl Serialize for Model {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Model {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(match s.as_str() {
            "deepseek-chat" => Model::DeepseekChat,
            "deepseek-reasoner" => Model::DeepseekReasoner,
            other => Model::Custom(other.to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_deepseek_chat() {
        let json = serde_json::to_string(&Model::DeepseekChat).unwrap();
        assert_eq!(json, r#""deepseek-chat""#);
    }

    #[test]
    fn serialize_deepseek_reasoner() {
        let json = serde_json::to_string(&Model::DeepseekReasoner).unwrap();
        assert_eq!(json, r#""deepseek-reasoner""#);
    }

    #[test]
    fn serialize_custom() {
        let json = serde_json::to_string(&Model::Custom("gpt-4o".to_string())).unwrap();
        assert_eq!(json, r#""gpt-4o""#);
    }

    #[test]
    fn deserialize_known_variants() {
        let m: Model = serde_json::from_str(r#""deepseek-chat""#).unwrap();
        assert_eq!(m, Model::DeepseekChat);

        let m: Model = serde_json::from_str(r#""deepseek-reasoner""#).unwrap();
        assert_eq!(m, Model::DeepseekReasoner);
    }

    #[test]
    fn deserialize_unknown_becomes_custom() {
        let m: Model = serde_json::from_str(r#""gpt-4o""#).unwrap();
        assert_eq!(m, Model::Custom("gpt-4o".to_string()));
    }

    #[test]
    fn default_is_deepseek_chat() {
        assert_eq!(Model::default(), Model::DeepseekChat);
    }

    #[test]
    fn as_str_roundtrips() {
        assert_eq!(Model::DeepseekChat.as_str(), "deepseek-chat");
        assert_eq!(Model::DeepseekReasoner.as_str(), "deepseek-reasoner");
        assert_eq!(Model::Custom("o3".to_string()).as_str(), "o3");
    }
}
