use std::fs;
use std::io::{self, ErrorKind};
use std::path::PathBuf;
use chrono::{DateTime, Utc};
use openai_dive::v1::resources::chat::ChatMessage;
use serde::{Deserialize, Serialize};
use tracing::{debug, error};
use uuid::Uuid;

/// Session data stored on disk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    pub session_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub trace: Vec<ChatMessage>,
}

/// Handle session persistence to disk
pub struct SessionPersist;

type PersistError = Box<dyn std::error::Error + Send + Sync>;

impl SessionPersist {
    /// Check if session persistence is enabled via environment variable
    pub fn is_enabled() -> bool {
        std::env::var("SHAI_SESSION_PERSIST_ENABLE")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(true)
    }

    /// Get the folder path for session storage
    pub fn folder() -> PathBuf {
        std::env::var("SHAI_SESSION_PERSIST_FOLDER")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(".shai/sessions"))
    }

    /// Get the file path for a specific session
    fn session_file_path(session_id: &str) -> PathBuf {
        Self::folder().join(format!("{}.json", session_id))
    }

    /// Save a session to disk (atomic write using temp file)
    pub fn save_session(
        session_id: &str,
        trace: Vec<ChatMessage>,
    ) -> Result<(), PersistError> {
        if !Self::is_enabled() {
            return Ok(());
        }

        let folder = Self::folder();

        // Create directory if it doesn't exist
        if let Err(e) = fs::create_dir_all(&folder) {
            error!("Failed to create session directory: {}", e);
            return Err(e.into());
        }

        let file_path = Self::session_file_path(session_id);

        // Load existing data to preserve created_at, or create new
        let (created_at, updated_at) = if file_path.exists() {
            match fs::read_to_string(&file_path) {
                Ok(content) => {
                    match serde_json::from_str::<SessionData>(&content) {
                        Ok(existing) => (existing.created_at, Utc::now()),
                        Err(_) => (Utc::now(), Utc::now()),
                    }
                }
                Err(_) => (Utc::now(), Utc::now()),
            }
        } else {
            (Utc::now(), Utc::now())
        };

        let session_data = SessionData {
            session_id: session_id.to_string(),
            created_at,
            updated_at,
            trace,
        };

        // Serialize to JSON
        let json = serde_json::to_string_pretty(&session_data)?;

        // Atomic write: write to temp file, then rename
        let temp_path = folder.join(format!("{}.tmp", Uuid::new_v4()));
        fs::write(&temp_path, json)?;
        fs::rename(&temp_path, &file_path)?;

        debug!("Session saved to disk: {}", file_path.display());
        Ok(())
    }

    /// Load a single session from disk by session_id
    /// Returns the session data if found, or an error if not found or failed to load
    pub fn load_session(session_id: &str) -> Result<SessionData, PersistError> {
        if !Self::is_enabled() {
            return Err(io::Error::new(
                ErrorKind::Other,
                "Session persistence is not enabled",
            )
            .into());
        }

        let file_path = Self::session_file_path(session_id);

        // If file doesn't exist, return error
        if !file_path.exists() {
            debug!("Session file does not exist: {}", file_path.display());
            return Err(io::Error::new(
                ErrorKind::NotFound,
                format!("Session file not found: {}", session_id),
            )
            .into());
        }

        // Read and parse the session file
        let content = fs::read_to_string(&file_path)?;
        let session_data: SessionData = serde_json::from_str(&content)?;

        debug!("Loaded session from disk: {}", session_id);
        Ok(session_data)
    }

    /// Delete a session file from disk
    pub fn delete_session(session_id: &str) {
        if !Self::is_enabled() {
            return;
        }

        let file_path = Self::session_file_path(session_id);

        if file_path.exists() {
            match fs::remove_file(&file_path) {
                Ok(_) => debug!("Deleted session file: {}", file_path.display()),
                Err(e) => error!("Failed to delete session file {:?}: {}", file_path, e),
            }
        }
    }
}
