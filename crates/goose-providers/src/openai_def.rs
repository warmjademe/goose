use anyhow::Result;
use std::collections::HashMap;

use crate::api_client::{ApiClient, AuthMethod};
use crate::base::DEFAULT_PROVIDER_TIMEOUT_SECS;
use crate::config::declarative_providers::DeclarativeProviderConfig;
use crate::model::ModelConfig;
use crate::openai::{
    ensure_url_scheme, parse_custom_headers, OpenAiProvider, OpenAiProviderBuilder, ParsedBaseUrl,
    OPEN_AI_DEFAULT_BASE_PATH, OPEN_AI_DEFAULT_FAST_MODEL, OPEN_AI_PROVIDER_NAME,
    OPEN_AI_VERSIONLESS_BASE_PATH,
};

pub async fn from_env(
    model: ModelConfig,
    tls_config: Option<crate::api_client::TlsConfig>,
) -> Result<OpenAiProvider> {
    let config = crate::config::Config::global();

    // Resolve host and base_path.
    //
    // Priority (highest first):
    //   1. OPENAI_HOST env var — session override (deprecated but still
    //      honoured so that `OPENAI_HOST=… goose` keeps working)
    //   2. OPENAI_BASE_URL (env or config) — ecosystem-standard
    //   3. OPENAI_HOST from config file — persisted by `goose configure`
    //   4. Default "https://api.openai.com"
    //
    // OPENAI_BASE_URL is parsed into host + query params + a flag
    // indicating whether the URL included a /v1 path segment.  When /v1
    // is present the default base_path is "v1/chat/completions";
    // otherwise "chat/completions" to match the OpenAI SDK convention.
    //
    // OPENAI_BASE_PATH always wins when set explicitly.
    let parsed = if let Ok(h) = std::env::var("OPENAI_HOST") {
        // OPENAI_HOST env var takes priority as a session override so
        // that existing scripts like `OPENAI_HOST=… goose` still work
        // even after OPENAI_BASE_URL is persisted in config.
        ParsedBaseUrl {
            host: h,
            query_params: vec![],
            has_v1: true,
            from_base_url: false,
        }
    } else if let Some(raw_url) = config
        .get_param::<String>("OPENAI_BASE_URL")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
    {
        OpenAiProvider::parse_base_url(&raw_url)?
    } else {
        let h: String = config
            .get_param("OPENAI_HOST")
            .unwrap_or_else(|_| "https://api.openai.com".to_string());
        ParsedBaseUrl {
            host: h,
            query_params: vec![],
            has_v1: true,
            from_base_url: false,
        }
    };

    // When the host was derived from OPENAI_BASE_URL, read
    // OPENAI_BASE_PATH from env only so that the desktop UI's persisted
    // default ("v1/chat/completions") doesn't shadow the versionless
    // path.  When the host came from OPENAI_HOST (env or config), read
    // from config too — Docker Model Runner and similar setups persist a
    // custom base_path that must be honoured.
    let default_bp = || {
        if parsed.has_v1 {
            OPEN_AI_DEFAULT_BASE_PATH.to_string()
        } else {
            OPEN_AI_VERSIONLESS_BASE_PATH.to_string()
        }
    };
    let base_path: String = if parsed.from_base_url {
        std::env::var("OPENAI_BASE_PATH").unwrap_or_else(|_| default_bp())
    } else {
        config
            .get_param("OPENAI_BASE_PATH")
            .unwrap_or_else(|_| default_bp())
    };

    // Only apply the default fast model when talking to OpenAI directly.
    // Custom/compatible endpoints likely don't serve gpt-4o-mini, so
    // leave fast_model unset (complete_fast will fall back to the main model).
    // Parse the URL and compare the hostname exactly to avoid false positives
    // (e.g. https://api.openai.com.local:8000 or proxy paths containing api.openai.com).
    let host = parsed.host.clone();

    let is_openai = url::Url::parse(&host)
        .ok()
        .and_then(|u| u.host_str().map(|h| h.to_ascii_lowercase()))
        .map(|h| h == "api.openai.com" || h.ends_with(".api.openai.com"))
        .unwrap_or(false);
    let model = if is_openai {
        crate::model_config::with_configured_fast_model(
            model,
            OPEN_AI_PROVIDER_NAME,
            OPEN_AI_DEFAULT_FAST_MODEL,
        )?
    } else {
        model
    };

    let secrets = config
        .get_secrets("OPENAI_API_KEY", &["OPENAI_CUSTOM_HEADERS"])
        .unwrap_or_default();
    let api_key: Option<String> = secrets.get("OPENAI_API_KEY").cloned();
    let custom_headers: Option<HashMap<String, String>> = secrets
        .get("OPENAI_CUSTOM_HEADERS")
        .cloned()
        .map(parse_custom_headers);

    let organization: Option<String> = config.get_param("OPENAI_ORGANIZATION").ok();
    let project: Option<String> = config.get_param("OPENAI_PROJECT").ok();
    let timeout_secs: u64 = config
        .get_param("OPENAI_TIMEOUT")
        .unwrap_or(DEFAULT_PROVIDER_TIMEOUT_SECS);

    let auth = match api_key {
        Some(key) if !key.is_empty() => AuthMethod::BearerToken(key),
        _ => AuthMethod::NoAuth,
    };
    let mut api_client = ApiClient::with_timeout_and_tls(
        parsed.host,
        auth,
        std::time::Duration::from_secs(timeout_secs),
        tls_config,
    )?;

    if !parsed.query_params.is_empty() {
        api_client = api_client.with_query(parsed.query_params);
    }

    if let Some(org) = &organization {
        api_client = api_client.with_header("OpenAI-Organization", org)?;
    }

    if let Some(project) = &project {
        api_client = api_client.with_header("OpenAI-Project", project)?;
    }

    if let Some(headers) = &custom_headers {
        let mut header_map = reqwest::header::HeaderMap::new();
        for (key, value) in headers {
            let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())?;
            let header_value = reqwest::header::HeaderValue::from_str(value)?;
            header_map.insert(header_name, header_value);
        }
        api_client = api_client.with_headers(header_map)?;
    }

    let mut provider = OpenAiProviderBuilder::new(api_client, model)
        .base_path(base_path)
        .organization(organization)
        .project(project)
        .custom_headers(custom_headers)
        .preserve_thinking_context(!is_openai)
        .build();

    provider.probe_context_limit_if_unset().await;

    Ok(provider)
}

/// Resolve the API key from a declarative provider config.
///
/// Returns `Some(key)` if a key is found, `None` if the key is optional/missing,
/// or an error if the key is required but missing/unreadable.
///
/// The `get_secret` closure is used to look up the secret by key name. This allows
/// testing without depending on `Config::global()`.
pub fn resolve_api_key(
    config: &DeclarativeProviderConfig,
    get_secret: &dyn Fn(&str) -> Result<String, crate::config::ConfigError>,
) -> Result<Option<String>> {
    if config.api_key_env.is_empty() {
        return Ok(None);
    }

    match get_secret(&config.api_key_env) {
        Ok(key) => Ok(Some(key)),
        Err(e) => {
            use crate::config::ConfigError;
            match e {
                ConfigError::NotFound(_) => {
                    if config.requires_auth {
                        anyhow::bail!(
                            "Required API key {} is not set. Configure it via `goose configure` or set the {} environment variable.",
                            config.api_key_env,
                            config.api_key_env
                        );
                    }
                    Ok(None)
                }
                other => {
                    if config.requires_auth {
                        anyhow::bail!("Failed to read {}: {}", config.api_key_env, other);
                    } else {
                        tracing::warn!(
                            "Failed to read optional API key {}: {}. Proceeding without authentication.",
                            config.api_key_env,
                            other
                        );
                        Ok(None)
                    }
                }
            }
        }
    }
}

pub fn from_custom_config(
    model: ModelConfig,
    config: DeclarativeProviderConfig,
    tls_config: Option<crate::api_client::TlsConfig>,
) -> Result<OpenAiProvider> {
    let custom_models = if !config.models.is_empty() {
        Some(
            config
                .models
                .iter()
                .map(|m| m.name.clone())
                .collect::<Vec<String>>(),
        )
    } else {
        None
    };

    if config.dynamic_models == Some(false) && custom_models.is_none() {
        return Err(anyhow::anyhow!(
            "Provider '{}' has dynamic_models: false but no static models listed; \
             at least one entry in `models` is required.",
            config.name
        ));
    }

    let global_config = crate::config::Config::global();
    let api_key = resolve_api_key(&config, &|key| global_config.get_secret(key))?;

    let normalized_base_url = ensure_url_scheme(&config.base_url);
    let url = url::Url::parse(&normalized_base_url)
        .map_err(|e| anyhow::anyhow!("Invalid base URL '{}': {}", config.base_url, e))?;

    let host = if let Some(port) = url.port() {
        format!(
            "{}://{}:{}",
            url.scheme(),
            url.host_str().unwrap_or(""),
            port
        )
    } else {
        format!("{}://{}", url.scheme(), url.host_str().unwrap_or(""))
    };
    let base_path = if let Some(ref explicit_path) = config.base_path {
        explicit_path.trim_start_matches('/').to_string()
    } else {
        OpenAiProvider::derive_base_path(url.path())
    };

    let timeout_secs = config
        .timeout_seconds
        .unwrap_or(DEFAULT_PROVIDER_TIMEOUT_SECS);

    let auth = match api_key {
        Some(key) if !key.is_empty() => AuthMethod::BearerToken(key),
        _ => AuthMethod::NoAuth,
    };
    let mut api_client = ApiClient::with_timeout_and_tls(
        host,
        auth,
        std::time::Duration::from_secs(timeout_secs),
        tls_config,
    )?;

    if let Some(headers) = &config.headers {
        let mut header_map = reqwest::header::HeaderMap::new();
        for (key, value) in headers {
            let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())?;
            let header_value = reqwest::header::HeaderValue::from_str(value)?;
            header_map.insert(header_name, header_value);
        }
        api_client = api_client.with_headers(header_map)?;
    }

    let model = if let Some(ref fast_model_name) = config.fast_model {
        crate::model_config::with_configured_fast_model(model, &config.name, fast_model_name)?
    } else {
        model
    };

    Ok(OpenAiProviderBuilder::new(api_client, model)
        .base_path(base_path)
        .custom_headers(config.headers)
        .supports_streaming(config.supports_streaming.unwrap_or(true))
        .name(config.name.clone())
        .custom_models(custom_models)
        .dynamic_models(config.dynamic_models)
        .skip_canonical_filtering(config.skip_canonical_filtering)
        .preserve_thinking_context(config.preserves_thinking)
        .build())
}
