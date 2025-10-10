use axum::{
    extract::State,
    http::StatusCode,
    Json,
};
use shai_core::agent::{Agent, AgentEvent, AgentBuilder};
use openai_dive::v1::resources::chat::{
    ChatCompletionParameters, ChatCompletionResponse,
    ChatMessage, ChatMessageContent, ChatCompletionChoice,
};
use openai_dive::v1::resources::shared::FinishReason;
use tracing::{error, info};
use uuid::Uuid;

use crate::ServerState;

/// Handle OpenAI chat completion - non-streaming only
pub async fn handle_chat_completion(
    State(state): State<ServerState>,
    Json(payload): Json<ChatCompletionParameters>,
) -> Result<Json<ChatCompletionResponse>, StatusCode> {
    let session_id = Uuid::new_v4();

    // Log request with path
    info!("[{}] POST /v1/chat/completions", session_id);

    // Create a new agent for this request
    let mut agent = AgentBuilder::create(state.agent_config_name.clone()).await
        .map_err(|e| {
            error!("[{}] Failed to create agent: {}", session_id, e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .with_traces(payload.messages.clone())
        .sudo()
        .build();

    let mut event_rx = agent.watch();

    // Run the agent in the background
    let session_id_clone = session_id;
    tokio::spawn(async move {
        if let Err(e) = agent.run().await {
            error!("[{}] Agent execution error: {}", session_id_clone, e);
        }
    });

    // Wait for agent to complete and collect the final message
    let mut final_message = String::new();
    let mut finish_reason = FinishReason::StopSequenceReached;

    while let Ok(event) = event_rx.recv().await {
        match event {
            // Capture assistant messages from brain results
            AgentEvent::BrainResult { thought, .. } => {
                if let Ok(msg) = thought {
                    if let ChatMessage::Assistant { content: Some(ChatMessageContent::Text(text)), .. } = msg {
                        final_message = text;
                    }
                }
            }
            // Log tool calls
            AgentEvent::ToolCallStarted { call, .. } => {
                info!("[{}] TOOL {}", session_id, call.tool_name);
            }
            AgentEvent::ToolCallCompleted { call, result, .. } => {
                use shai_core::tools::ToolResult;
                let status = match &result {
                    ToolResult::Success { .. } => "✓",
                    ToolResult::Error { .. } => "✗",
                    ToolResult::Denied => "⊘",
                };
                info!("[{}] TOOL {} {}", session_id, call.tool_name, status);
            }
            // Agent completed or paused - return the result
            AgentEvent::Completed { message, success, .. } => {
                if !message.is_empty() {
                    final_message = message;
                }
                if !success {
                    finish_reason = FinishReason::StopSequenceReached;
                }
                info!("[{}] Completed", session_id);
                break;
            }
            AgentEvent::StatusChanged { new_status, .. } => {
                use shai_core::agent::PublicAgentState;
                if matches!(new_status, PublicAgentState::Paused { .. }) {
                    info!("[{}] Paused", session_id);
                    break;
                }
            }
            AgentEvent::Error { error } => {
                error!("[{}] Agent error: {}", session_id, error);
                finish_reason = FinishReason::StopSequenceReached;
                break;
            }
            _ => {}
        }
    }

    let response = ChatCompletionResponse {
        id: Some(session_id.to_string()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp() as u32,
        model: payload.model.clone(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage::Assistant {
                content: Some(ChatMessageContent::Text(final_message)),
                tool_calls: None,
                name: None,
                audio: None,
                reasoning_content: None,
                refusal: None,
            },
            finish_reason: Some(finish_reason),
            logprobs: None,
        }],
        usage: None,
        system_fingerprint: None,
        service_tier: None,
    };

    Ok(Json(response))
}
