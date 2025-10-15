use shai_core::agent::{Agent, AgentError};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info};

use shai_core::agent::AgentBuilder;
use crate::session::log_event;

use super::AgentSession;

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
    ephemeral: bool
}

impl SessionManager {
    pub fn new(config: SessionManagerConfig) -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            max_sessions: config.max_sessions,
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

        // Spawn logging task alongside agent
        let mut event_for_logger = event_rx.resubscribe();
        let sid_for_logger = session_id.to_string();
        let logging_task = tokio::spawn(async move {
            while let Ok(event) = event_for_logger.recv().await {
                log_event(&event, &sid_for_logger);
            }
        });

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
            logging_task,
            agent_task,
            agent_name,
            ephemeral,
        ));

        Ok(session)
    }

    /// Get an existing session by ID
    /// Returns error if session doesn't exist
    pub async fn get_session(
        &self,
        http_request_id: &str,
        session_id: &str,
    ) -> Result<Arc<AgentSession>, AgentError> {
        let sessions = self.sessions.lock().await;

        if let Some(session) = sessions.get(session_id) {
            info!("[{}] - [{}] Using existing session", http_request_id, session_id);
            Ok(session.clone())
        } else {
            Err(AgentError::ExecutionError(format!(
                "Session not found: {}",
                session_id
            )))
        }
    }

    /// Create a new session with the given ID
    /// Returns error if session already exists
    pub async fn create_new_session(
        &self,
        http_request_id: &str,
        session_id: &str,
        agent_name: Option<String>,
        ephemeral: bool,
    ) -> Result<Arc<AgentSession>, AgentError> {
        let mut sessions = self.sessions.lock().await;

        // Check if session already exists
        if sessions.contains_key(session_id) {
            return Err(AgentError::ExecutionError(format!(
                "Session already exists: {}",
                session_id
            )));
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

         // Check max sessions limit
        if self.ephemeral && !ephemeral {
            return Err(AgentError::ExecutionError(format!(
                "Only Ephemeral session are authorized on this server"
            )));
        }

        let session = self.create_session(&http_request_id.to_string(), session_id, agent_name, ephemeral).await?;
        sessions.insert(session_id.to_string(), session.clone());

        Ok(session)
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
}
