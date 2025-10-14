use shai_core::agent::AgentController;
use tokio::sync::OwnedMutexGuard;
use tracing::debug;


pub enum RequestLifecycle {
    Background {
        controller_guard: OwnedMutexGuard<AgentController>,
        session_id: String,
    },
    Ephemeral {
        controller_guard: OwnedMutexGuard<AgentController>,
        session_id: String,
    },
}

impl RequestLifecycle {
    pub fn new(ephemeral: bool, controller_guard: OwnedMutexGuard<AgentController>, session_id: String) -> Self {
        match ephemeral {
            true => Self::Ephemeral { controller_guard, session_id },
            false => Self::Background { controller_guard, session_id },
        }
    }
}

impl Drop for RequestLifecycle {
    fn drop(&mut self) {
        match self {
            Self::Background { session_id, .. } => {
                debug!(
                    "[{}] Stream completed, releasing controller lock (background session)",
                    session_id
                );
            }
            Self::Ephemeral { controller_guard, session_id } => {
                debug!(
                    "[{}] Stream completed, destroying agent (ephemeral session)",
                    session_id
                );

                // Clone before moving into async task
                let ctrl = controller_guard.clone();
                tokio::spawn(async move {
                    let _ = ctrl.cancel().await;
                });
            }
        }
    }
}
