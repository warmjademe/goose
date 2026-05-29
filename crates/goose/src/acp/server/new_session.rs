use crate::acp::server::{
    build_mode_state, build_usage_updates, builtin_to_extension_config, meta_string, sid_short,
    validate_absolute_cwd, ResultExt,
};
use crate::config::extensions::get_enabled_extensions_with_config;
use crate::config::{Config, GooseMode};
use crate::model::ModelConfig;
use crate::session::SessionType;
use crate::session::{EnabledExtensionsState, ExtensionState};

use super::{GooseAcpAgent, GooseAcpSession};
use agent_client_protocol::schema::{
    NewSessionRequest, NewSessionResponse, SessionId, SessionNotification, SessionUpdate,
};
use agent_client_protocol::{Client, ConnectionTo};
use std::collections::{HashMap, HashSet};
use tracing::{debug, error};

impl GooseAcpAgent {
    #[allow(dead_code)]
    pub(super) async fn handle_new_session(
        &self,
        cx: &ConnectionTo<Client>,
        args: NewSessionRequest,
    ) -> Result<NewSessionResponse, agent_client_protocol::Error> {
        debug!(?args, "new session request");
        let t_start = std::time::Instant::now();
        validate_absolute_cwd(&args.cwd)?;
        let project_id = meta_string(args.meta.as_ref(), "projectId");
        let session_type = match meta_string(args.meta.as_ref(), "client") {
            Some(_) => SessionType::User,
            None => SessionType::Acp,
        };
        let config = Config::global();
        let resolved_provider = config.get_goose_provider().map_err(|error| {
            agent_client_protocol::Error::internal_error()
                .data(format!("Failed to resolve provider: {}", error))
        })?;
        let resolved_model = config.get_goose_model().map_err(|error| {
            agent_client_protocol::Error::internal_error()
                .data(format!("Failed to resolve model: {}", error))
        })?;
        let resolved_model_config = ModelConfig::new(&resolved_model)
            .map(|model_config| model_config.with_canonical_limits(&resolved_provider))
            .map_err(|error| {
                agent_client_protocol::Error::internal_error()
                    .data(format!("Failed to resolve model: {}", error))
            })?;
        let current_mode: GooseMode = config.get_goose_mode().unwrap_or_default();
        let t0 = std::time::Instant::now();
        let goose_session = self
            .session_manager
            .create_session(
                args.cwd.clone(),
                "New Chat".to_string(),
                session_type,
                current_mode,
            )
            .await
            .internal_err_ctx("Failed to create session")?;
        let mut builder = self.session_manager.update(&goose_session.id);
        let mut extensions = get_enabled_extensions_with_config(&config);
        extensions.extend(self.builtins.iter().map(|b| builtin_to_extension_config(b)));
        let mut extension_data = goose_session.extension_data.clone();
        EnabledExtensionsState::new(extensions)
            .to_extension_data(&mut extension_data)
            .internal_err_ctx("Failed to initialize session extensions")?;
        builder = builder
            .provider_name(resolved_provider)
            .model_config(resolved_model_config)
            .extension_data(extension_data);
        if let Some(pid) = project_id {
            builder = builder.project_id(Some(pid));
        }
        builder
            .apply()
            .await
            .internal_err_ctx("Failed to update session")?;

        let goose_session = self
            .session_manager
            .get_session(&goose_session.id, false)
            .await
            .internal_err_ctx("Failed to reload session")?;
        let session_id_str = goose_session.id.clone();
        let sid = sid_short(&session_id_str);
        debug!(target: "perf", sid = %sid, ms = t0.elapsed().as_millis() as u64, "perf: new_session create_session");

        let agent = self
            .get_or_create_session_agent(cx, session_id_str.clone())
            .await?;
        self.apply_acp_extension_overrides(cx, &agent, &goose_session)
            .await;

        if let Err(error) =
            Self::add_mcp_extensions(&agent, args.mcp_servers, &goose_session.id).await
        {
            error!(
                error = %error,
                "new_session MCP server setup failed; continuing with ready session"
            );
        }

        let acp_session = GooseAcpSession {
            agent: agent.clone(),
            tool_requests: HashMap::new(),
            chain_membership: HashMap::new(),
            responded_tool_ids: HashSet::new(),
            summarized_chains: HashSet::new(),
            cancel_token: None,
        };
        self.sessions
            .lock()
            .await
            .insert(session_id_str.clone(), acp_session);

        self.maybe_refresh_provider_inventory_with_agent(&goose_session, &agent)
            .await;

        let goose_session = self
            .session_manager
            .get_session(&goose_session.id, false)
            .await
            .internal_err_ctx("Failed to reload session")?;

        let acp_session_id = SessionId::new(session_id_str.clone());
        let working_dir = goose_session.working_dir.clone();

        let mode_state = build_mode_state(goose_session.goose_mode)?;
        let usage_updates = build_usage_updates(&goose_session);

        let (model_state, config_options) = self
            .build_eager_session_config(&mode_state, &goose_session)
            .await;

        let mut response = NewSessionResponse::new(acp_session_id.clone()).modes(mode_state);
        if let Some(ms) = model_state {
            response = response.models(ms);
        }
        if let Some(co) = config_options {
            response = response.config_options(co);
        }
        if let Some(updates) = usage_updates {
            cx.send_notification(updates.custom)?;
            cx.send_notification(SessionNotification::new(
                acp_session_id.clone(),
                SessionUpdate::UsageUpdate(updates.legacy),
            ))?;
        }
        Self::send_available_commands_update(cx, &acp_session_id, &working_dir)?;
        debug!(
            target: "perf",
            sid = %sid,
            ms = t_start.elapsed().as_millis() as u64,
            "perf: new_session done"
        );
        Ok(response)
    }
}
