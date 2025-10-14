use shai_core::agent::{Agent, AgentError};
use shai_llm::ChatMessage;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info};
use uuid::Uuid;

use shai_core::agent::AgentBuilder;
use super::{AgentSession, RequestSession};

/// Configuration for the session manager
#[derive(Clone, Debug)]
pub struct SessionManagerConfig {
    /// Maximum number of concurrent sessions (None = unlimited)
    pub max_sessions: Option<usize>,
    /// Whether sessions are ephemeral or background (ephemeral session is destroyed after a single query)
    pub ephemeral: bool,
}

impl Default for SessionManagerConfig {
    fn default() -> Self {
        Self {
            max_sessions: Some(100),
            ephemeral: false,
        }
    }
}

/// Session manager - manages multiple agent sessions by ID
/// Handles creation, deletion, and access control for sessions
pub struct SessionManager {
    sessions: Arc<Mutex<HashMap<String, Arc<AgentSession>>>>,
    max_sessions: Option<usize>,
    allow_creation: bool,
    ephemeral: bool
}

impl SessionManager {
    pub fn new(config: SessionManagerConfig) -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            max_sessions: config.max_sessions,
            allow_creation: true,
            ephemeral: config.ephemeral
        }
    }

    async fn create_session(
        &self,
        http_request_id: &String,
        session_id: &str,
        agent_name: Option<String>,
        ephemeral: bool,
    ) -> Result<Arc<AgentSession>, AgentError> {
        info!("[{}] - [{}] Creating new session", http_request_id, session_id);

        // Build the agent
        let mut agent = AgentBuilder::create(agent_name.clone().filter(|name| name != "default"))
            .await
            .map_err(|e| AgentError::ExecutionError(format!("Failed to create agent: {}", e)))?
            .sudo()
            .build();

        let controller = agent.controller();
        let event_rx = agent.watch();

        // Spawn agent task with cleanup logic
        let sessions_for_cleanup = self.sessions.clone();
        let sid_for_cleanup = session_id.to_string();

        let agent_task = tokio::spawn(async move {
            match agent.run().await {
                Ok(_) => {
                    info!("[] - [{}] Agent completed successfully", sid_for_cleanup);
                }
                Err(e) => {
                    error!("[] - [{}] Agent execution error: {}", sid_for_cleanup, e);
                }
            }
            sessions_for_cleanup.lock().await.remove(&sid_for_cleanup);
            info!("[] - [{}] session removed from manager", sid_for_cleanup);
        });

        let session = Arc::new(AgentSession::new(
            session_id.to_string(),
            controller,
            event_rx,
            agent_task,
            agent_name,
            ephemeral,
        ));

        Ok(session)
    }

    async fn get_or_create_session(
        &self,
        http_request_id: &String,
        session_id: &str,
        agent_name: Option<String>,
        ephemeral: bool,
    ) -> Result<Arc<AgentSession>, AgentError> {
        let mut sessions = self.sessions.lock().await;

        // Check if session exists
        if let Some(session) = sessions.get(session_id) {
            info!("[{}] - [{}] Using existing session", http_request_id, session_id);
            return Ok(session.clone());
        }

        // Check if creation is allowed
        if !self.allow_creation {
            return Err(AgentError::ExecutionError(
                "Session creation disabled".to_string(),
            ));
        }

        // Check max sessions limit
        if let Some(max) = self.max_sessions {
            if sessions.len() >= max {
                return Err(AgentError::ExecutionError(format!(
                    "Maximum number of sessions reached: {}",
                    max
                )));
            }
        }

        let session = self.create_session(&http_request_id, session_id, agent_name, ephemeral).await?;

        sessions.insert(session_id.to_string(), session.clone());
        Ok(session)
    }

    /// Handle an incoming request
    /// - If `session_id` is provided, use or create that session
    /// - If `session_id` is None, generate a new ephemeral session ID
    pub async fn handle_request(
        &self,
        http_request_id: String,
        session_id: Option<String>,
        trace: Vec<ChatMessage>,
        agent_name: Option<String>
    ) -> Result<(RequestSession, String), AgentError> {
        let session_id = session_id.unwrap_or_else(|| {
            Uuid::new_v4().to_string()
        });

        let session = self.get_or_create_session(&http_request_id, &session_id, agent_name, self.ephemeral).await?;
        let request_session = session.handle_request(&http_request_id, trace).await?;

        // Cleanup is handled automatically by the session's own lifecycle
        Ok((request_session, session_id))
    }

    /// Cancel a session (stop the agent)
    pub async fn cancel_session(&self, http_request_id: &String, session_id: &str) -> Result<(), AgentError> {
        if let Some(session) = self.sessions.lock().await.get(session_id) {
            session.cancel(http_request_id).await?;
        }
        Ok(())
    }

    /// Get the number of active sessions
    pub async fn session_count(&self) -> usize {
        self.sessions.lock().await.len()
    }

    /// Set whether new sessions can be created
    pub fn set_allow_creation(&mut self, allow: bool) {
        self.allow_creation = allow;
    }
}
