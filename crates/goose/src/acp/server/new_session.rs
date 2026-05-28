use crate::acp::server::{meta_string, sid_short, validate_absolute_cwd, ResultExt};
use crate::config::{Config, GooseMode};
use crate::session::SessionType;

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
        if let Some(provider_name) = goose_session
            .provider_name
            .clone()
            .or_else(|| config.get_goose_provider().ok())
        {
            builder = builder.provider_name(provider_name);
        }
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

        let acp_session_id = SessionId::new(session_id_str.clone());
        let init_state = self.prepare_session_init_state(&goose_session).await?;
        let working_dir = goose_session.working_dir.clone();

        let agent = self
            .agent_manager
            .get_or_create_agent(session_id_str.clone())
            .await
            .internal_err_ctx("Failed to create agent")?;

        if let Err(error) =
            Self::add_mcp_extensions(&agent, args.mcp_servers, &goose_session.id).await
        {
            error!(
                error = %error,
                "new_session MCP server setup failed; continuing with ready session"
            );
        }

        let acp_session = GooseAcpSession {
            agent,
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

        let mut response =
            NewSessionResponse::new(acp_session_id.clone()).modes(init_state.mode_state);
        if let Some(ms) = init_state.model_state {
            response = response.models(ms);
        }
        if let Some(co) = init_state.config_options {
            response = response.config_options(co);
        }
        if let Some(updates) = init_state.usage_updates {
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
