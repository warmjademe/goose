//! `goose::providers` — public, low-level provider surface (phase one).
//!
//! This is the uniffi-exported provider surface described in the proposed
//! design. For now every method returns **mock data**: the goal is to lock in
//! and exercise the FFI-facing shapes (records, enums, interfaces, errors)
//! without wiring up the real `goose` provider implementations yet.
//!
//! Design notes carried over from the sketch:
//!  * The `Provider` is the driver. Callers resolve a [`ModelConfig`] *from* the
//!    provider instead of constructing limits by hand.
//!  * Construction needs only what's required to reach the backend — no model,
//!    no extensions.
//!  * Extensions are an rMCP / caller concern. Their tools enter per call via
//!    [`CompletionRequest::tools`], never at construction.
//!  * Everything crossing the FFI boundary is a plain `Record` or an `interface`
//!    object so the same surface generates idiomatic Python/Kotlin.

use std::sync::Arc;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Construction — connection details only.
// ---------------------------------------------------------------------------

/// Everything needed to *reach* a backend. No model, no extensions.
#[derive(Clone, Debug, uniffi::Record)]
pub struct ProviderConfig {
    /// API key / token. Resolution (env, keychain) is the caller's business.
    pub api_key: Option<String>,
    /// Override the default endpoint (self-hosted, proxy, Azure, etc.).
    #[uniffi(default = None)]
    pub base_url: Option<String>,
    /// Optional org / project / workspace scoping.
    #[uniffi(default = None)]
    pub organization: Option<String>,
    /// Per-request wall-clock timeout in milliseconds.
    #[uniffi(default = Some(60_000))]
    pub timeout_ms: Option<u64>,
}

/// Construct a provider by name (`"anthropic"`, `"openai"`, `"databricks"`, …).
///
/// Mock: accepts any non-empty name and returns a [`MockProvider`]. Unknown or
/// empty names produce a [`ProviderError::ModelNotAvailable`]-style error so the
/// error path is exercisable from the bindings.
#[uniffi::export]
pub fn create(
    provider_name: &str,
    config: ProviderConfig,
) -> Result<Arc<dyn Provider>, ProviderError> {
    if provider_name.is_empty() {
        return Err(ProviderError::InvalidRequest {
            message: "provider_name must not be empty".to_string(),
        });
    }
    Ok(Arc::new(MockProvider {
        name: provider_name.to_string(),
        _config: config,
    }))
}

// ---------------------------------------------------------------------------
// Model metadata — resolved by the provider, not assembled by the caller.
// ---------------------------------------------------------------------------

/// Where a [`ModelConfig`]'s limits/capabilities came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, uniffi::Enum)]
pub enum MetadataSource {
    /// Pulled from the provider's catalog for a recognized model.
    Canonical,
    /// Model not recognized; documented fallback applied (override as needed).
    Default,
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct ModelCapabilities {
    pub supports_tools: bool,
    pub supports_streaming: bool,
    pub supports_vision: bool,
}

/// Default sampling knobs. Carried by the model, overridable per request.
#[derive(Clone, Debug, Default, uniffi::Record)]
pub struct ModelParams {
    #[uniffi(default = None)]
    pub temperature: Option<f64>,
    #[uniffi(default = None)]
    pub max_output_tokens: Option<u32>,
    #[uniffi(default = None)]
    pub top_p: Option<f64>,
}

/// Resolved model metadata. Obtain via [`Provider::model`] for canonical values,
/// or build directly for full manual control via [`model_config_new`].
#[derive(Clone, Debug, uniffi::Record)]
pub struct ModelConfig {
    pub name: String,
    pub context_limit: u32,
    pub capabilities: ModelCapabilities,
    pub params: ModelParams,
    /// Canonical vs. default — set by the provider on resolution.
    pub source: MetadataSource,
}

/// Manual construction for callers who want to bypass provider resolution
/// entirely (testing, unknown/self-hosted models, full control).
///
/// Exposed as a free function rather than an inherent `impl` so it crosses the
/// uniffi boundary as a plain constructor in Python/Kotlin.
#[uniffi::export]
pub fn model_config_new(name: String, context_limit: u32) -> ModelConfig {
    ModelConfig {
        name,
        context_limit,
        capabilities: ModelCapabilities {
            supports_tools: true,
            supports_streaming: true,
            supports_vision: false,
        },
        params: ModelParams::default(),
        source: MetadataSource::Default,
    }
}

// ---------------------------------------------------------------------------
// Messages & tools — the place extensions actually show up.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, uniffi::Enum)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Message content. Tool requests/responses live here so the agent loop is
/// expressible at this layer without the provider knowing about extensions.
#[derive(Clone, Debug, uniffi::Enum)]
pub enum Content {
    Text {
        text: String,
    },
    ToolRequest {
        id: String,
        name: String,
        arguments_json: String,
    },
    ToolResponse {
        id: String,
        result_json: String,
    },
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Content>,
}

/// A tool offered to the model for this call. Whether it was sourced from an
/// rMCP extension or hand-built is the caller's concern, not the provider's.
/// `input_schema_json` is JSON-as-string to stay an FFI-safe `Record` field.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema_json: String,
}

// ---------------------------------------------------------------------------
// Completion request/response.
// ---------------------------------------------------------------------------

/// A single completion call. Model, messages, and tools are all passed per
/// request — nothing is baked into the provider.
#[derive(Clone, Debug, uniffi::Record)]
pub struct CompletionRequest {
    pub model: ModelConfig,
    pub messages: Vec<Message>,
    /// Tools available for this call (typically extension-provided). Empty = none.
    #[uniffi(default = [])]
    pub tools: Vec<Tool>,
    /// Optional per-request override of the model's default sampling params.
    #[uniffi(default = None)]
    pub params: Option<ModelParams>,
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct ProviderUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Provider-computed cost in USD when available.
    pub cost_usd: Option<f64>,
}

/// Named record instead of a `(Message, ProviderUsage)` tuple — tuples don't
/// cross the uniffi boundary.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Completion {
    pub message: Message,
    pub usage: ProviderUsage,
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct CompletionChunk {
    /// Incremental content for this chunk.
    pub delta: Content,
    /// Populated on the final chunk.
    pub usage: Option<ProviderUsage>,
}

/// Streaming exposed as an object with an async pull, not a raw Rust `Stream`
/// (which has no FFI representation). `None` signals end-of-stream.
#[uniffi::export(async_runtime = "tokio")]
#[async_trait::async_trait]
pub trait CompletionStream: Send + Sync {
    async fn next(&self) -> Result<Option<CompletionChunk>, ProviderError>;
}

// ---------------------------------------------------------------------------
// The provider interface.
// ---------------------------------------------------------------------------

/// Resolves models and runs completions. Retry/backoff and error classification
/// live behind this interface.
#[uniffi::export(async_runtime = "tokio")]
#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    /// Resolve a [`ModelConfig`] from the provider's catalog.
    ///
    /// Always returns a config: canonical metadata for recognized models
    /// (`source = Canonical`), or a documented fallback you can override
    /// (`source = Default`).
    fn model(&self, name: &str) -> ModelConfig;

    /// One-shot completion with usage/cost included.
    async fn complete(&self, request: CompletionRequest) -> Result<Completion, ProviderError>;

    /// Streaming variant. Returns an object you pull [`CompletionChunk`]s from.
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Arc<dyn CompletionStream>, ProviderError>;
}

// ---------------------------------------------------------------------------
// Error classification — surfaced as a uniffi error.
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum ProviderError {
    #[error("authentication failed: {message}")]
    Authentication { message: String },

    #[error("rate limited; retry after {retry_after_ms:?} ms")]
    RateLimited { retry_after_ms: Option<u64> },

    #[error("request exceeded the model context limit")]
    ContextLengthExceeded,

    #[error("model not available: {name}")]
    ModelNotAvailable { name: String },

    #[error("transport/server error: {message}")]
    Server { message: String },

    #[error("request was malformed: {message}")]
    InvalidRequest { message: String },
}

// ---------------------------------------------------------------------------
// Mock implementations.
// ---------------------------------------------------------------------------

/// A stand-in [`Provider`] that returns deterministic mock data. Lets callers
/// (and the generated Python/Kotlin examples) exercise the full surface before
/// the real provider implementations are wired in.
struct MockProvider {
    name: String,
    _config: ProviderConfig,
}

const MOCK_CONTEXT_LIMIT: u32 = 128_000;

#[async_trait::async_trait]
impl Provider for MockProvider {
    fn model(&self, name: &str) -> ModelConfig {
        // Pretend a couple of names are "known" so both metadata sources show up.
        let canonical = matches!(name, "gpt-4o" | "claude-3-5-sonnet" | "claude-sonnet-4");
        ModelConfig {
            name: name.to_string(),
            context_limit: MOCK_CONTEXT_LIMIT,
            capabilities: ModelCapabilities {
                supports_tools: true,
                supports_streaming: true,
                supports_vision: canonical,
            },
            params: ModelParams {
                temperature: Some(0.7),
                max_output_tokens: Some(4096),
                top_p: None,
            },
            source: if canonical {
                MetadataSource::Canonical
            } else {
                MetadataSource::Default
            },
        }
    }

    async fn complete(&self, request: CompletionRequest) -> Result<Completion, ProviderError> {
        let reply = format!(
            "mock completion from '{}' using model '{}' ({} message(s), {} tool(s))",
            self.name,
            request.model.name,
            request.messages.len(),
            request.tools.len(),
        );
        Ok(Completion {
            message: Message {
                role: Role::Assistant,
                content: vec![Content::Text { text: reply }],
            },
            usage: ProviderUsage {
                input_tokens: 42,
                output_tokens: 12,
                cost_usd: Some(0.000_54),
            },
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Arc<dyn CompletionStream>, ProviderError> {
        let words = vec![
            "mock".to_string(),
            "stream".to_string(),
            "from".to_string(),
            request.model.name.clone(),
        ];
        Ok(Arc::new(MockStream {
            words: Mutex::new(words.into_iter()),
        }))
    }
}

/// A mock [`CompletionStream`] that yields a fixed sequence of text deltas and a
/// final usage chunk, then `None`.
struct MockStream {
    words: Mutex<std::vec::IntoIter<String>>,
}

#[async_trait::async_trait]
impl CompletionStream for MockStream {
    async fn next(&self) -> Result<Option<CompletionChunk>, ProviderError> {
        let word = self.words.lock().expect("mock stream lock poisoned").next();
        match word {
            Some(text) => Ok(Some(CompletionChunk {
                delta: Content::Text {
                    text: format!("{text} "),
                },
                usage: None,
            })),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ProviderConfig {
        ProviderConfig {
            api_key: Some("mock-key".to_string()),
            base_url: None,
            organization: None,
            timeout_ms: Some(60_000),
        }
    }

    fn test_request(model: ModelConfig) -> CompletionRequest {
        CompletionRequest {
            model,
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text {
                    text: "hello".to_string(),
                }],
            }],
            tools: vec![],
            params: None,
        }
    }

    #[test]
    fn create_rejects_empty_name() {
        assert!(create("", test_config()).is_err());
    }

    #[test]
    fn model_resolution_marks_source() {
        let provider = create("openai", test_config()).expect("provider");
        assert_eq!(provider.model("gpt-4o").source, MetadataSource::Canonical);
        assert_eq!(
            provider.model("some-unknown-model").source,
            MetadataSource::Default
        );
    }

    #[test]
    fn model_config_new_is_default_source() {
        let cfg = model_config_new("custom".to_string(), 8192);
        assert_eq!(cfg.context_limit, 8192);
        assert_eq!(cfg.source, MetadataSource::Default);
    }

    #[tokio::test]
    async fn complete_returns_mock_assistant_message() {
        let provider = create("anthropic", test_config()).expect("provider");
        let model = provider.model("claude-sonnet-4");
        let completion = provider
            .complete(test_request(model))
            .await
            .expect("completion");
        assert_eq!(completion.message.role, Role::Assistant);
        assert!(completion.usage.output_tokens > 0);
    }

    #[tokio::test]
    async fn stream_yields_chunks_then_none() {
        let provider = create("anthropic", test_config()).expect("provider");
        let model = provider.model("gpt-4o");
        let stream = provider.stream(test_request(model)).await.expect("stream");

        let mut count = 0;
        while stream.next().await.expect("chunk").is_some() {
            count += 1;
        }
        assert!(count > 0);
        assert!(stream.next().await.expect("chunk").is_none());
    }
}
