use anyhow::Result;
use async_trait::async_trait;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::api_client::{ApiClient, AuthMethod, AuthProvider};
use super::base::{
    ConfigKey, MessageStream, Provider, ProviderDef, ProviderMetadata,
    DEFAULT_PROVIDER_TIMEOUT_SECS,
};
use super::embedding::EmbeddingCapable;
use super::errors::ProviderError;
use super::formats::databricks::create_request;
use super::formats::openai_responses::create_responses_request;
use super::oauth;
use super::openai_compatible::{
    handle_response_openai_compat, handle_status, map_http_error_to_provider_error,
    stream_openai_compat, stream_responses_compat,
};
use super::retry::ProviderRetry;
use super::utils::{ImageFormat, RequestLog};
use crate::config::ConfigError;
use crate::conversation::message::Message;
use crate::instance_id::get_instance_id;
use crate::model::ModelConfig;
use crate::providers::retry::{
    RetryConfig, DEFAULT_BACKOFF_MULTIPLIER, DEFAULT_INITIAL_RETRY_INTERVAL_MS,
    DEFAULT_MAX_RETRIES, DEFAULT_MAX_RETRY_INTERVAL_MS,
};
use rmcp::model::Tool;
use serde_json::json;

const DEFAULT_CLIENT_ID: &str = "databricks-cli";
const DEFAULT_REDIRECT_URL: &str = "http://localhost";
const DEFAULT_SCOPES: &[&str] = &["all-apis", "offline_access"];

const DATABRICKS_PROVIDER_NAME: &str = "databricks";
pub const DATABRICKS_DEFAULT_MODEL: &str = "databricks-claude-sonnet-4";
const DATABRICKS_DEFAULT_FAST_MODEL: &str = "databricks-claude-haiku-4-5";
pub const DATABRICKS_KNOWN_MODELS: &[&str] = &[
    "databricks-claude-sonnet-4-5",
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-meta-llama-3-1-405b-instruct",
];

pub const DATABRICKS_DOC_URL: &str =
    "https://docs.databricks.com/en/generative-ai/external-models/index.html";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatabricksAuth {
    Token(String),
    OAuth {
        host: String,
        client_id: String,
        redirect_url: String,
        scopes: Vec<String>,
    },
}

impl DatabricksAuth {
    pub fn oauth(host: String) -> Self {
        Self::OAuth {
            host,
            client_id: DEFAULT_CLIENT_ID.to_string(),
            redirect_url: DEFAULT_REDIRECT_URL.to_string(),
            scopes: DEFAULT_SCOPES.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn token(token: String) -> Self {
        Self::Token(token)
    }
}

struct DatabricksAuthProvider {
    auth: DatabricksAuth,
    token_cache: Arc<Mutex<Option<String>>>,
}

#[async_trait]
impl AuthProvider for DatabricksAuthProvider {
    async fn get_auth_header(&self) -> Result<(String, String)> {
        let token = match &self.auth {
            DatabricksAuth::Token(original) => {
                let cached = self.token_cache.lock().unwrap().clone();
                match cached {
                    Some(t) => t,
                    None => {
                        // Cache was cleared by refresh_credentials(); re-read
                        // from config which may have a sidecar-rotated token.
                        // Fall back to the constructor-provided token if config
                        // lookup fails (e.g. from_params usage).
                        let fresh = crate::config::Config::global()
                            .get_secret::<String>("DATABRICKS_TOKEN")
                            .unwrap_or_else(|_| original.clone());
                        *self.token_cache.lock().unwrap() = Some(fresh.clone());
                        fresh
                    }
                }
            }
            DatabricksAuth::OAuth {
                host,
                client_id,
                redirect_url,
                scopes,
            } => oauth::get_oauth_token_async(host, client_id, redirect_url, scopes).await?,
        };
        Ok(("Authorization".to_string(), format!("Bearer {}", token)))
    }
}

#[derive(Debug, serde::Serialize)]
pub struct DatabricksProvider {
    #[serde(skip)]
    api_client: ApiClient,
    auth: DatabricksAuth,
    model: ModelConfig,
    image_format: ImageFormat,
    #[serde(skip)]
    retry_config: RetryConfig,
    #[serde(skip)]
    fast_retry_config: RetryConfig,
    #[serde(skip)]
    name: String,
    #[serde(skip)]
    token_cache: Arc<Mutex<Option<String>>>,
    #[serde(skip)]
    instance_id: Option<String>,
}

impl DatabricksProvider {
    pub async fn cleanup() -> Result<()> {
        super::oauth::cleanup_oauth_cache()
    }

    pub async fn from_env(model: ModelConfig) -> Result<Self> {
        let config = crate::config::Config::global();

        let mut host: Result<String, ConfigError> = config.get_param("DATABRICKS_HOST");
        if host.is_err() {
            host = config.get_secret("DATABRICKS_HOST")
        }

        if host.is_err() {
            return Err(ConfigError::NotFound(
                "Did not find DATABRICKS_HOST in either config file or keyring".to_string(),
            )
            .into());
        }

        let host = host?;
        let retry_config = Self::load_retry_config(config);
        let fast_retry_config = Self::load_fast_retry_config(config);

        let auth = if let Ok(api_key) = config.get_secret("DATABRICKS_TOKEN") {
            DatabricksAuth::token(api_key)
        } else {
            DatabricksAuth::oauth(host.clone())
        };

        let token_cache = Arc::new(Mutex::new(match &auth {
            DatabricksAuth::Token(t) => Some(t.clone()),
            _ => None,
        }));

        let auth_method = AuthMethod::Custom(Box::new(DatabricksAuthProvider {
            auth: auth.clone(),
            token_cache: token_cache.clone(),
        }));

        let api_client = ApiClient::with_timeout(
            host,
            auth_method,
            Duration::from_secs(DEFAULT_PROVIDER_TIMEOUT_SECS),
        )?;

        let mut provider = Self {
            api_client,
            auth,
            model: model.clone(),
            image_format: ImageFormat::OpenAi,
            retry_config,
            fast_retry_config,
            name: DATABRICKS_PROVIDER_NAME.to_string(),
            token_cache,
            instance_id: Self::resolve_instance_id(),
        };
        provider.model =
            model.with_fast(DATABRICKS_DEFAULT_FAST_MODEL, DATABRICKS_PROVIDER_NAME)?;
        Ok(provider)
    }

    fn load_retry_config(config: &crate::config::Config) -> RetryConfig {
        let max_retries = config
            .get_param("DATABRICKS_MAX_RETRIES")
            .ok()
            .and_then(|v: String| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_RETRIES);

        let initial_interval_ms = config
            .get_param("DATABRICKS_INITIAL_RETRY_INTERVAL_MS")
            .ok()
            .and_then(|v: String| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_INITIAL_RETRY_INTERVAL_MS);

        let backoff_multiplier = config
            .get_param("DATABRICKS_BACKOFF_MULTIPLIER")
            .ok()
            .and_then(|v: String| v.parse::<f64>().ok())
            .unwrap_or(DEFAULT_BACKOFF_MULTIPLIER);

        let max_interval_ms = config
            .get_param("DATABRICKS_MAX_RETRY_INTERVAL_MS")
            .ok()
            .and_then(|v: String| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_MAX_RETRY_INTERVAL_MS);

        RetryConfig::new(
            max_retries,
            initial_interval_ms,
            backoff_multiplier,
            max_interval_ms,
        )
    }

    fn load_fast_retry_config(_config: &crate::config::Config) -> RetryConfig {
        // Fast models are hardcoded to 0 retries for quick failure on Databricks
        RetryConfig::new(0, 0, 1.0, 0)
    }

    pub fn from_params(host: String, api_key: String, model: ModelConfig) -> Result<Self> {
        let token_cache = Arc::new(Mutex::new(Some(api_key.clone())));
        let auth = DatabricksAuth::token(api_key);
        let auth_method = AuthMethod::Custom(Box::new(DatabricksAuthProvider {
            auth: auth.clone(),
            token_cache: token_cache.clone(),
        }));

        let api_client = ApiClient::with_timeout(
            host,
            auth_method,
            Duration::from_secs(DEFAULT_PROVIDER_TIMEOUT_SECS),
        )?;

        Ok(Self {
            api_client,
            auth,
            model,
            image_format: ImageFormat::OpenAi,
            retry_config: RetryConfig::default(),
            fast_retry_config: RetryConfig::new(0, 0, 1.0, 0),
            name: DATABRICKS_PROVIDER_NAME.to_string(),
            token_cache,
            instance_id: Self::resolve_instance_id(),
        })
    }

    fn resolve_instance_id() -> Option<String> {
        let enabled = crate::config::Config::global()
            .get_param::<bool>("GOOSE_DATABRICKS_CLIENT_REQUEST_ID")
            .unwrap_or(false);
        if enabled {
            Some(get_instance_id().to_string())
        } else {
            None
        }
    }

    fn is_responses_model(model_name: &str) -> bool {
        super::utils::is_openai_responses_model(model_name)
    }

    fn get_endpoint_path(&self, model_name: &str, is_embedding: bool) -> String {
        if is_embedding {
            "serving-endpoints/text-embedding-3-small/invocations".to_string()
        } else {
            let (clean_name, _) = super::utils::extract_reasoning_effort(model_name);
            if Self::is_responses_model(&clean_name) {
                "serving-endpoints/responses".to_string()
            } else {
                format!("serving-endpoints/{}/invocations", clean_name)
            }
        }
    }

    fn build_client_request_id(&self, session_id: &str) -> Option<String> {
        self.instance_id.as_ref().map(|instance_id| {
            json!({
                "sessionId": format!("{}_{}", instance_id, session_id),
            })
            .to_string()
        })
    }

    async fn post(
        &self,
        session_id: Option<&str>,
        mut payload: Value,
        model_name: Option<&str>,
    ) -> Result<Value, ProviderError> {
        let is_embedding = payload.get("input").is_some() && payload.get("messages").is_none();
        let model_to_use = model_name.unwrap_or(&self.model.model_name);
        let path = self.get_endpoint_path(model_to_use, is_embedding);

        if let Some(session_id) = session_id {
            if let Some(client_request_id) = self.build_client_request_id(session_id) {
                payload["client_request_id"] = Value::String(client_request_id);
            }
        }

        let response = self
            .api_client
            .response_post(session_id, &path, &payload)
            .await?;
        handle_response_openai_compat(response).await
    }
}

impl ProviderDef for DatabricksProvider {
    type Provider = Self;

    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            DATABRICKS_PROVIDER_NAME,
            "Databricks",
            "Models on Databricks AI Gateway",
            DATABRICKS_DEFAULT_MODEL,
            DATABRICKS_KNOWN_MODELS.to_vec(),
            DATABRICKS_DOC_URL,
            vec![
                ConfigKey::new("DATABRICKS_HOST", true, false, None, true),
                ConfigKey::new("DATABRICKS_TOKEN", false, true, None, true),
            ],
        )
    }

    fn from_env(
        model: ModelConfig,
        _extensions: Vec<crate::config::ExtensionConfig>,
    ) -> BoxFuture<'static, Result<Self::Provider>> {
        Box::pin(Self::from_env(model))
    }

    fn supports_inventory_refresh() -> bool {
        true
    }
}

#[async_trait]
impl Provider for DatabricksProvider {
    fn get_name(&self) -> &str {
        &self.name
    }

    fn retry_config(&self) -> RetryConfig {
        self.retry_config.clone()
    }

    async fn refresh_credentials(&self) -> Result<(), ProviderError> {
        crate::config::Config::global().invalidate_secrets_cache();
        *self.token_cache.lock().unwrap() = None;
        tracing::info!("Invalidated secrets cache and token cache for credential refresh");
        Ok(())
    }

    fn get_model_config(&self) -> ModelConfig {
        self.model.clone()
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        session_id: &str,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        let path = self.get_endpoint_path(&model_config.model_name, false);
        let client_request_id = self.build_client_request_id(session_id);

        if Self::is_responses_model(&model_config.model_name) {
            let (mut payload, protocol) =
                create_responses_payload_for_databricks(model_config, system, messages, tools)?;
            payload["stream"] = Value::Bool(true);
            if let Some(ref client_request_id) = client_request_id {
                payload["client_request_id"] = Value::String(client_request_id.clone());
            }

            let mut log = RequestLog::start(model_config, &payload)?;

            let response = self
                .with_retry(|| async {
                    let payload_clone = payload.clone();
                    let resp = self
                        .api_client
                        .response_post(Some(session_id), &path, &payload_clone)
                        .await?;
                    handle_status(resp).await
                })
                .await
                .inspect_err(|e| {
                    let _ = log.error(e);
                })?;

            match protocol {
                DatabricksResponsesProtocol::ChatCompletions => stream_openai_compat(response, log),
                DatabricksResponsesProtocol::ResponsesApi => stream_responses_compat(response, log),
            }
        } else {
            let mut payload =
                create_request(model_config, system, messages, tools, &self.image_format)?;
            payload
                .as_object_mut()
                .expect("payload should have model key")
                .remove("model");
            if let Some(client_request_id) = client_request_id {
                payload["client_request_id"] = Value::String(client_request_id);
            }

            payload
                .as_object_mut()
                .unwrap()
                .insert("stream".to_string(), Value::Bool(true));

            if let Some(opts) = payload
                .get_mut("stream_options")
                .and_then(|v| v.as_object_mut())
            {
                opts.entry("include_usage").or_insert(json!(true));
            } else {
                payload
                    .as_object_mut()
                    .unwrap()
                    .insert("stream_options".to_string(), json!({"include_usage": true}));
            }

            let mut log = RequestLog::start(model_config, &payload)?;
            let response = self
                .with_retry(|| async {
                    let resp = self
                        .api_client
                        .response_post(Some(session_id), &path, &payload)
                        .await?;
                    if !resp.status().is_success() {
                        let status = resp.status();
                        let error_text = resp.text().await.unwrap_or_default();

                        let json_payload = serde_json::from_str::<Value>(&error_text).ok();
                        return Err(map_http_error_to_provider_error(status, json_payload));
                    }
                    Ok(resp)
                })
                .await;

            let response = match response {
                Err(e) if e.to_string().contains("stream_options") => {
                    payload.as_object_mut().unwrap().remove("stream_options");
                    self.with_retry(|| async {
                        let resp = self
                            .api_client
                            .response_post(Some(session_id), &path, &payload)
                            .await?;
                        if !resp.status().is_success() {
                            let status = resp.status();
                            let error_text = resp.text().await.unwrap_or_default();
                            let json_payload = serde_json::from_str::<Value>(&error_text).ok();
                            return Err(map_http_error_to_provider_error(status, json_payload));
                        }
                        Ok(resp)
                    })
                    .await
                    .inspect_err(|e| {
                        let _ = log.error(e);
                    })?
                }
                Err(e) => {
                    let _ = log.error(&e);
                    return Err(e);
                }
                Ok(resp) => resp,
            };

            stream_openai_compat(response, log)
        }
    }

    fn supports_embeddings(&self) -> bool {
        true
    }

    async fn create_embeddings(
        &self,
        session_id: &str,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, ProviderError> {
        EmbeddingCapable::create_embeddings(self, session_id, texts)
            .await
            .map_err(|e| ProviderError::ExecutionError(e.to_string()))
    }

    async fn fetch_supported_models(&self) -> Result<Vec<String>, ProviderError> {
        let response = self
            .api_client
            .request(None, "api/2.0/serving-endpoints")
            .response_get()
            .await
            .map_err(|e| {
                ProviderError::RequestFailed(format!("Failed to fetch Databricks models: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let detail = response.text().await.unwrap_or_default();
            return Err(ProviderError::RequestFailed(format!(
                "Failed to fetch Databricks models: {} {}",
                status, detail
            )));
        }

        let json: Value = response.json().await.map_err(|e| {
            ProviderError::RequestFailed(format!("Failed to parse Databricks API response: {}", e))
        })?;

        let endpoints = json
            .get("endpoints")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                ProviderError::RequestFailed(
                    "Unexpected response format from Databricks API: missing 'endpoints' array"
                        .to_string(),
                )
            })?;

        let models: Vec<String> = endpoints
            .iter()
            .filter_map(|endpoint| {
                endpoint
                    .get("name")
                    .and_then(|v| v.as_str())
                    .map(|name| name.to_string())
            })
            .collect();

        Ok(models)
    }
}

#[async_trait]
impl EmbeddingCapable for DatabricksProvider {
    async fn create_embeddings(
        &self,
        session_id: &str,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let request = json!({
            "input": texts,
        });

        let response = self
            .with_retry_config(
                || self.post(Some(session_id), request.clone(), None),
                self.fast_retry_config.clone(),
            )
            .await?;

        let embeddings = response["data"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid response format: missing data array"))?
            .iter()
            .map(|item| {
                item["embedding"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("Invalid embedding format"))?
                    .iter()
                    .map(|v| v.as_f64().map(|f| f as f32))
                    .collect::<Option<Vec<f32>>>()
                    .ok_or_else(|| anyhow::anyhow!("Invalid embedding values"))
            })
            .collect::<Result<Vec<Vec<f32>>>>()?;

        Ok(embeddings)
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum DatabricksResponsesProtocol {
    ChatCompletions,
    ResponsesApi,
}

fn responses_protocol_for_databricks(
    model_name: &str,
    tools: &[Tool],
) -> DatabricksResponsesProtocol {
    let (_model_name, reasoning_effort) = super::utils::extract_reasoning_effort(model_name);
    if !tools.is_empty() && reasoning_effort.is_some() {
        DatabricksResponsesProtocol::ResponsesApi
    } else {
        DatabricksResponsesProtocol::ChatCompletions
    }
}

fn create_responses_payload_for_databricks(
    model_config: &ModelConfig,
    system: &str,
    messages: &[Message],
    tools: &[Tool],
) -> Result<(Value, DatabricksResponsesProtocol)> {
    let protocol = responses_protocol_for_databricks(&model_config.model_name, tools);
    let mut payload = create_responses_request(model_config, system, messages, tools)?;

    if protocol == DatabricksResponsesProtocol::ChatCompletions {
        adapt_responses_payload_for_databricks(&mut payload);
    }

    Ok((payload, protocol))
}

/// Adapt a payload produced by `create_responses_request` (OpenAI Responses API format)
/// into the Chat Completions format accepted by Databricks' `serving-endpoints/responses`
/// route for Responses-family models without the `tools + reasoning_effort` combination.
fn adapt_responses_payload_for_databricks(payload: &mut Value) {
    let obj = match payload.as_object_mut() {
        Some(o) => o,
        None => return,
    };

    // Remove Responses-API-only fields
    obj.remove("store");

    // Convert reasoning: {"effort": "high", "summary": "auto"} → "reasoning_effort": "high"
    if let Some(reasoning) = obj.remove("reasoning") {
        if let Some(effort) = reasoning.get("effort").and_then(|e| e.as_str()) {
            obj.insert("reasoning_effort".to_string(), json!(effort));
        }
    }

    // max_output_tokens → max_completion_tokens
    if let Some(max) = obj.remove("max_output_tokens") {
        obj.insert("max_completion_tokens".to_string(), max);
    }

    // Rename "input" → "messages" and transform items
    if let Some(input) = obj.remove("input") {
        if let Some(items) = input.as_array() {
            obj.insert(
                "messages".to_string(),
                json!(convert_responses_input_to_messages(items)),
            );
        }
    }

    // Convert Responses-API tool format to Chat Completions tool format:
    //   {type, name, description, parameters, strict} →
    //   {type: "function", function: {name, description, parameters}}
    if let Some(tools) = obj.get_mut("tools") {
        if let Some(tools_array) = tools.as_array_mut() {
            for tool in tools_array.iter_mut() {
                if let Some(tool_obj) = tool.as_object_mut() {
                    let name = tool_obj.remove("name");
                    let description = tool_obj.remove("description");
                    let parameters = tool_obj.remove("parameters");
                    tool_obj.remove("strict");

                    let mut function = serde_json::Map::new();
                    if let Some(n) = name {
                        function.insert("name".to_string(), n);
                    }
                    if let Some(d) = description {
                        function.insert("description".to_string(), d);
                    }
                    if let Some(p) = parameters {
                        function.insert("parameters".to_string(), p);
                    }
                    tool_obj.insert("function".to_string(), Value::Object(function));
                }
            }
        }
    }
}

fn convert_responses_input_to_messages(items: &[Value]) -> Vec<Value> {
    let mut messages: Vec<Value> = Vec::new();

    for item in items {
        // Top-level typed items (function_call, function_call_output)
        if let Some(item_type) = item.get("type").and_then(|t| t.as_str()) {
            match item_type {
                "function_call" => {
                    let tool_call = json!({
                        "id": item.get("call_id").cloned().unwrap_or(json!("")),
                        "type": "function",
                        "function": {
                            "name": item.get("name").cloned().unwrap_or(json!("")),
                            "arguments": item.get("arguments").cloned().unwrap_or(json!("{}"))
                        }
                    });

                    // Merge into the last assistant message if one exists
                    if let Some(last) = messages.last_mut() {
                        if last.get("role").and_then(|r| r.as_str()) == Some("assistant") {
                            let tool_calls = last
                                .as_object_mut()
                                .unwrap()
                                .entry("tool_calls")
                                .or_insert_with(|| json!([]));
                            tool_calls.as_array_mut().unwrap().push(tool_call);
                            continue;
                        }
                    }
                    messages.push(json!({
                        "role": "assistant",
                        "content": Value::Null,
                        "tool_calls": [tool_call]
                    }));
                }
                "function_call_output" => {
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": item.get("call_id").cloned().unwrap_or(json!("")),
                        "content": item.get("output").cloned().unwrap_or(json!(""))
                    }));
                }
                _ => {}
            }
            continue;
        }

        // Role-based message items — convert content types
        if item.get("role").is_some() {
            let mut msg = item.clone();
            if let Some(content) = msg.get_mut("content") {
                if let Some(array) = content.as_array_mut() {
                    for entry in array.iter_mut() {
                        if let Some(t) = entry.get("type").and_then(|t| t.as_str()) {
                            match t {
                                "input_text" | "output_text" => {
                                    entry
                                        .as_object_mut()
                                        .unwrap()
                                        .insert("type".to_string(), json!("text"));
                                }
                                "input_image" => {
                                    let obj = entry.as_object_mut().unwrap();
                                    obj.insert("type".to_string(), json!("image_url"));
                                    if let Some(url) = obj.remove("image_url") {
                                        obj.insert("image_url".to_string(), json!({"url": url}));
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
            messages.push(msg);
        }
    }

    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::object;

    fn test_provider() -> DatabricksProvider {
        DatabricksProvider {
            api_client: super::super::api_client::ApiClient::new(
                "https://example.com".to_string(),
                super::super::api_client::AuthMethod::NoAuth,
            )
            .unwrap(),
            auth: DatabricksAuth::Token("fake".into()),
            model: ModelConfig::new_or_fail("databricks-gpt-5.4"),
            image_format: ImageFormat::OpenAi,
            retry_config: RetryConfig::default(),
            fast_retry_config: RetryConfig::new(0, 0, 1.0, 0),
            name: "databricks".into(),
            token_cache: std::sync::Arc::new(std::sync::Mutex::new(None)),
            instance_id: None,
        }
    }

    #[test]
    fn responses_models_route_to_responses_endpoint() {
        let provider = test_provider();

        for (model_name, expected_path) in [
            ("gpt-5.4", "serving-endpoints/responses"),
            ("gpt-5.5", "serving-endpoints/responses"),
            ("databricks-gpt-5.5-high", "serving-endpoints/responses"),
            ("databricks-gpt-5.4-high", "serving-endpoints/responses"),
            ("databricks-gpt-5-4-xhigh", "serving-endpoints/responses"),
            ("o3-mini", "serving-endpoints/responses"),
            (
                "databricks-claude-sonnet-4",
                "serving-endpoints/databricks-claude-sonnet-4/invocations",
            ),
        ] {
            assert_eq!(
                provider.get_endpoint_path(model_name, false),
                expected_path,
                "unexpected endpoint for {model_name}"
            );
        }
    }

    #[test]
    fn adapt_responses_payload_converts_to_chat_completions() {
        let mut payload = json!({
            "model": "gpt-5.5",
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "You are helpful."}]
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hi there!"}]
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "my_tool",
                    "arguments": "{\"x\":1}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "result"
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Thanks"}]
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "my_tool",
                    "description": "A tool",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": false
                }
            ],
            "store": false,
            "max_output_tokens": 4096,
            "reasoning": {"effort": "high", "summary": "auto"}
        });

        adapt_responses_payload_for_databricks(&mut payload);

        // "store" removed
        assert!(payload.get("store").is_none());

        // reasoning converted
        assert!(payload.get("reasoning").is_none());
        assert_eq!(payload["reasoning_effort"], "high");

        // max_output_tokens → max_completion_tokens
        assert!(payload.get("max_output_tokens").is_none());
        assert_eq!(payload["max_completion_tokens"], 4096);

        // "input" renamed to "messages"
        assert!(payload.get("input").is_none());
        let messages = payload["messages"].as_array().unwrap();

        // system message content types converted
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"][0]["type"], "text");
        assert_eq!(messages[0]["content"][0]["text"], "You are helpful.");

        // user message content types converted
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"][0]["type"], "text");

        // assistant message with merged tool_calls
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[2]["content"][0]["type"], "text");
        assert_eq!(messages[2]["content"][0]["text"], "Hi there!");
        let tool_calls = messages[2]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "call_1");
        assert_eq!(tool_calls[0]["function"]["name"], "my_tool");
        assert_eq!(tool_calls[0]["function"]["arguments"], "{\"x\":1}");

        // function_call_output → tool role message
        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["tool_call_id"], "call_1");
        assert_eq!(messages[3]["content"], "result");

        // trailing user message
        assert_eq!(messages[4]["role"], "user");
        assert_eq!(messages[4]["content"][0]["type"], "text");

        // tools converted to Chat Completions format
        let tools = payload["tools"].as_array().unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "my_tool");
        assert_eq!(tools[0]["function"]["description"], "A tool");
        assert!(tools[0].get("name").is_none());
        assert!(tools[0].get("strict").is_none());
    }

    #[test]
    fn responses_protocol_uses_native_responses_for_tools_with_reasoning_effort() {
        let tool = Tool::new(
            "my_tool",
            "A tool",
            object!({
                "type": "object",
                "properties": {}
            }),
        );

        assert_eq!(
            responses_protocol_for_databricks(
                "databricks-gpt-5.5-high",
                std::slice::from_ref(&tool)
            ),
            DatabricksResponsesProtocol::ResponsesApi
        );
        assert_eq!(
            responses_protocol_for_databricks("databricks-gpt-5.5", std::slice::from_ref(&tool)),
            DatabricksResponsesProtocol::ChatCompletions
        );
        assert_eq!(
            responses_protocol_for_databricks("databricks-gpt-5.5-high", &[]),
            DatabricksResponsesProtocol::ChatCompletions
        );
    }

    #[test]
    fn responses_payload_keeps_native_format_for_tools_with_reasoning_effort() {
        let model_config = ModelConfig {
            model_name: "databricks-gpt-5.5-high".to_string(),
            context_limit: None,
            temperature: None,
            max_tokens: Some(4096),
            toolshim: false,
            toolshim_model: None,
            fast_model_config: None,
            request_params: None,
            reasoning: None,
        };
        let tool = Tool::new(
            "my_tool",
            "A tool",
            object!({
                "type": "object",
                "properties": {}
            }),
        );

        let (payload, protocol) = create_responses_payload_for_databricks(
            &model_config,
            "You are helpful.",
            &[],
            &[tool],
        )
        .unwrap();

        assert_eq!(protocol, DatabricksResponsesProtocol::ResponsesApi);
        assert_eq!(payload["model"], "databricks-gpt-5.5");
        assert!(payload.get("input").is_some());
        assert!(payload.get("messages").is_none());
        assert_eq!(payload["reasoning"]["effort"], "high");
        assert_eq!(payload["reasoning"]["summary"], "auto");
        assert!(payload.get("reasoning_effort").is_none());
        assert_eq!(payload["max_output_tokens"], 4096);
        assert!(payload.get("max_completion_tokens").is_none());

        let tools = payload["tools"].as_array().unwrap();
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "my_tool");
        assert_eq!(tools[0]["strict"], false);
        assert!(tools[0].get("function").is_none());
    }
}
