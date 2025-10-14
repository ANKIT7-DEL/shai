use shai_core::agent::{AgentController, AgentError, AgentEvent};
use shai_llm::ChatMessage;
use std::sync::Arc;
use tokio::sync::{broadcast::Receiver, Mutex};
use tokio::task::JoinHandle;
use tracing::debug;
use openai_dive::v1::resources::chat::ChatMessageContentPart;
use shai_llm::ChatMessageContent;
use super::RequestLifecycle;


/// Represents a single HTTP request session with automatic lifecycle management
pub struct RequestSession {
    pub controller: AgentController,
    pub event_rx: Receiver<AgentEvent>,
    pub lifecycle: RequestLifecycle
}


/// A single agent session - represents one running agent instance
/// Can be ephemeral (destroyed after request) or persistent (kept alive)
pub struct AgentSession {
    controller: Arc<Mutex<AgentController>>,
    event_rx: Receiver<AgentEvent>,
    agent_task: JoinHandle<()>,

    pub session_id: String,
    pub agent_name: String,
    pub ephemeral: bool,
}

impl AgentSession {
    pub fn new(
        session_id: String,
        controller: AgentController,
        event_rx: Receiver<AgentEvent>,
        agent_task: JoinHandle<()>,
        agent_name: Option<String>,
        ephemeral: bool,
    ) -> Self {
        let agent_name_display = agent_name.unwrap_or_else(|| "default".to_string());

        Self {
            controller: Arc::new(Mutex::new(controller)),
            event_rx,
            agent_task,
            session_id,
            agent_name: agent_name_display,
            ephemeral: ephemeral,
        }
    }

    pub async fn cancel(&self, http_request_id: &String)  -> Result<(), AgentError> {
        let ctrl = self.controller.clone().lock_owned().await;
        debug!("[{}] - [{}] cancelling session", http_request_id, self.session_id);
        ctrl.cancel().await
    }

    /// Handle a request for this agent session
    /// Returns a RequestSession that manages the lifecycle
    pub async fn handle_request(&self, http_request_id: &String, trace: Vec<ChatMessage>) -> Result<RequestSession, AgentError> {
        let controller_guard = self.controller.clone().lock_owned().await;
        debug!("[{}] - [{}] handling request", http_request_id, self.session_id);

        // TODO
        // make a new controller API to send a full trace
        for msg in trace {
            match msg {
                ChatMessage::User { content, .. } => {
                    let text = match content {
                        ChatMessageContent::Text(t) => t,
                        ChatMessageContent::ContentPart(parts) => {
                            parts.iter()
                                .filter_map(|p| match p {
                                    ChatMessageContentPart::Text(text_part) => Some(text_part.text.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        }
                        ChatMessageContent::None => String::new(),
                    };
                    if !text.is_empty() {
                        controller_guard.send_user_input(text).await?;
                    }
                }
                _ => {}
            }
        }

        let event_rx = self.event_rx.resubscribe();
        let controller = controller_guard.clone();
        let lifecycle = RequestLifecycle::new(self.ephemeral, controller_guard, self.session_id.clone());

        Ok(RequestSession{controller, event_rx, lifecycle})
    }

    pub fn is_ephemeral(&self) -> bool {
        self.ephemeral
    }
}

impl Drop for AgentSession {
    fn drop(&mut self) {
        debug!("[] - [{}] Dropping agent session", self.session_id);
        self.agent_task.abort();
    }
}
