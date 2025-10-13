use axum::{
    extract::State,
    response::{IntoResponse, Response},
    Json,
};
use shai_core::agent::{Agent, AgentEvent};
use openai_dive::v1::resources::chat::{
    ChatCompletionParameters, ChatCompletionResponse,
    ChatMessage, ChatMessageContent, ChatCompletionChoice,
};
use openai_dive::v1::resources::shared::FinishReason;
use tracing::{error, info};
use uuid::Uuid;

use crate::{ApiJson, ServerState, create_agent_from_model, ErrorResponse};

/// Handle OpenAI chat completion - non-streaming only
pub async fn handle_chat_completion(
    State(_state): State<ServerState>,
    ApiJson(payload): ApiJson<ChatCompletionParameters>,
) -> Result<Response, ErrorResponse> {
    let session_id = Uuid::new_v4();

    // Log request with path
    info!("[{}] POST /v1/chat/completions model={}", session_id, payload.model);

    // Create a new agent for this request
    let mut agent = create_agent_from_model(&payload.model, &session_id).await?
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
                info!("[{}] ToolCall: {}", session_id, call.tool_name);
            }
            AgentEvent::ToolCallCompleted { call, result, .. } => {
                use shai_core::tools::ToolResult;
                match &result {
                    ToolResult::Success { .. } => {
                        info!("[{}] ToolResult: {} ✓", session_id, call.tool_name);
                    }
                    ToolResult::Error { error, .. } => {
                        let error_oneline = error.lines().next().unwrap_or(error);
                        info!("[{}] ToolResult: {} ✗ {}", session_id, call.tool_name, error_oneline);
                    }
                    ToolResult::Denied => {
                        info!("[{}] ToolResult: {} ⊘ Permission denied", session_id, call.tool_name);
                    }
                }
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

    Ok(Json(response).into_response())
}
