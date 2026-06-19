pub mod auth;
pub mod connection;
pub mod http;
pub mod websocket;

use std::sync::Arc;

use axum::{
    body::Body,
    extract::{
        ws::{rejection::WebSocketUpgradeRejection, WebSocketUpgrade},
        State,
    },
    http::{header, HeaderName, HeaderValue, Method, Request},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Router,
};
use serde_json::Value;
use tower_http::cors::{AllowOrigin, CorsLayer};

use crate::acp::server_factory::AcpServer;

pub(crate) const HEADER_CONNECTION_ID: &str = "Acp-Connection-Id";
pub(crate) const HEADER_SESSION_ID: &str = "Acp-Session-Id";
pub(crate) const EVENT_STREAM_MIME_TYPE: &str = "text/event-stream";
pub(crate) const JSON_MIME_TYPE: &str = "application/json";

pub(crate) fn accepts_mime_type(request: &Request<Body>, mime_type: &str) -> bool {
    request
        .headers()
        .get(axum::http::header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|accept| accept.contains(mime_type))
}

pub(crate) fn content_type_is_json(request: &Request<Body>) -> bool {
    request
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with(JSON_MIME_TYPE))
}

pub(crate) fn header_value(request: &Request<Body>, name: &str) -> Option<String> {
    request
        .headers()
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

pub(crate) fn is_jsonrpc_request_with_id(value: &Value) -> bool {
    value.get("method").is_some() && value.get("id").is_some()
}

pub(crate) fn is_jsonrpc_notification(value: &Value) -> bool {
    value.get("method").is_some() && value.get("id").is_none()
}

pub(crate) fn is_jsonrpc_response(value: &Value) -> bool {
    value.get("id").is_some()
        && value.get("method").is_none()
        && (value.get("result").is_some() || value.get("error").is_some())
}

pub(crate) fn is_initialize_request(value: &Value) -> bool {
    value.get("method").is_some_and(|m| m == "initialize") && value.get("id").is_some()
}

/// Methods that are scoped to a session and require an Acp-Session-Id header.
pub(crate) fn method_requires_session_header(method: &str) -> bool {
    matches!(
        method,
        "session/prompt"
            | "session/cancel"
            | "session/load"
            | "session/set_mode"
            | "session/set_model"
    )
}

async fn handle_get(
    ws_upgrade: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
    registry: Arc<connection::ConnectionRegistry>,
    request: Request<Body>,
    ws_allowed_origins: Option<Arc<[HeaderValue]>>,
) -> Response {
    match ws_upgrade {
        Ok(ws) => {
            if ws_allowed_origins
                .as_deref()
                .is_some_and(|allowed_origins| !websocket_origin_allowed(&request, allowed_origins))
            {
                return (
                    axum::http::StatusCode::FORBIDDEN,
                    "Forbidden: WebSocket Origin is not allowed",
                )
                    .into_response();
            }
            websocket::handle_ws_upgrade(registry, ws).await
        }
        Err(_) => http::handle_get(registry, request).await,
    }
}

async fn health() -> &'static str {
    "ok"
}

fn acp_origin_allowed(origin: &HeaderValue, additional_allowed_origins: &[HeaderValue]) -> bool {
    if !additional_allowed_origins.is_empty() {
        return additional_allowed_origins
            .iter()
            .any(|allowed_origin| allowed_origin == origin);
    }

    let Ok(origin) = origin.to_str() else {
        return false;
    };

    let Ok(url) = url::Url::parse(origin) else {
        return false;
    };

    if !matches!(url.scheme(), "http" | "https") {
        return false;
    }

    url.host_str()
        .is_some_and(|host| matches!(host, "localhost" | "127.0.0.1" | "::1" | "[::1]"))
}

fn websocket_origin_allowed(
    request: &Request<Body>,
    additional_allowed_origins: &[HeaderValue],
) -> bool {
    request
        .headers()
        .get(header::ORIGIN)
        .is_none_or(|origin| acp_origin_allowed(origin, additional_allowed_origins))
}

fn acp_cors_layer(additional_allowed_origins: Vec<HeaderValue>) -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(move |origin, _request_parts| {
            acp_origin_allowed(origin, &additional_allowed_origins)
        }))
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
        .allow_headers([
            header::CONTENT_TYPE,
            header::ACCEPT,
            HeaderName::from_static("x-secret-key"),
            HeaderName::from_static("acp-connection-id"),
            HeaderName::from_static("acp-session-id"),
            header::SEC_WEBSOCKET_VERSION,
            header::SEC_WEBSOCKET_KEY,
            header::CONNECTION,
            header::UPGRADE,
        ])
        .expose_headers([
            HeaderName::from_static("acp-connection-id"),
            HeaderName::from_static("acp-session-id"),
        ])
}

fn create_acp_routes(
    server: Arc<AcpServer>,
    ws_allowed_origins: Option<Vec<HeaderValue>>,
) -> Router {
    let registry = Arc::new(connection::ConnectionRegistry::new(server));
    let ws_allowed_origins: Option<Arc<[HeaderValue]>> = ws_allowed_origins.map(Into::into);

    Router::new()
        .route("/acp", post(http::handle_post).with_state(registry.clone()))
        .route(
            "/acp",
            get({
                let ws_allowed_origins = ws_allowed_origins.clone();
                move |ws_upgrade: Result<WebSocketUpgrade, WebSocketUpgradeRejection>,
                      State(registry): State<Arc<connection::ConnectionRegistry>>,
                      request: Request<Body>| {
                    handle_get(ws_upgrade, registry, request, ws_allowed_origins.clone())
                }
            })
            .with_state(registry.clone()),
        )
        .route("/acp", delete(http::handle_delete).with_state(registry))
}

pub fn create_acp_router(server: Arc<AcpServer>) -> Router {
    create_acp_routes(server, Some(Vec::new())).layer(acp_cors_layer(Vec::new()))
}

pub fn create_router(
    server: Arc<AcpServer>,
    secret_key: String,
    require_token: bool,
    additional_allowed_origins: Vec<HeaderValue>,
) -> Router {
    let mut acp_routes = create_acp_routes(server, Some(additional_allowed_origins.clone()));
    if require_token {
        acp_routes = acp_routes.layer(axum::middleware::from_fn_with_state(
            secret_key.clone(),
            auth::check_acp_token,
        ));
    }
    acp_routes
        .route("/health", get(health))
        .route("/status", get(health))
        .merge(super::mcp_app_proxy::routes(secret_key))
        .layer(acp_cors_layer(additional_allowed_origins))
}
