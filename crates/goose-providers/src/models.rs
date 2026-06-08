use std::collections::HashMap;

use serde_json::Value;

use crate::thinking::ThinkingEffort;

pub struct ModelConfigParams<'a> {
    pub model_name: &'a str,
    pub thinking_effort: Option<ThinkingEffort>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<i32>,
    pub request_params: Option<&'a HashMap<String, Value>>,
}
