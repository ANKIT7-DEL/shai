use axum::{
    extract::State,
    http::StatusCode,
    Json,
};
use shai_core::agent::{Agent, AgentEvent, AgentBuilder};
use openai_dive::v1::resources::response::{
    items::{FunctionToolCall, InputItemStatus}, request::{ContentInput, ContentItem, ResponseInput, ResponseInputItem, ResponseParameters}, response::{
        MessageStatus, OutputContent, OutputMessage, ReasoningStatus, ResponseObject, ResponseOutput, Role
    }
};
use openai_dive::v1::resources::{
    shared::Usage
};
use shai_llm::{ChatMessage, ChatMessageContent};
use tracing::{error, info};
use uuid::Uuid;

use crate::ServerState;

/// Convert OpenAI Response API input to ChatMessage trace
fn build_message_trace(params: &ResponseParameters) -> Vec<ChatMessage> {
    let mut trace = Vec::new();

    // Add instructions as system message if present
    if let Some(instructions) = &params.instructions {
        trace.push(ChatMessage::System {
            content: ChatMessageContent::Text(instructions.clone()),
            name: None,
        });
    }

    // Convert input messages
    match &params.input {
        ResponseInput::Text(text) => {
            trace.push(ChatMessage::User {
                content: ChatMessageContent::Text(text.clone()),
                name: None,
            });
        }
        ResponseInput::List(items) => {
            for item in items {
                if let ResponseInputItem::Message(msg) = item {
                    match &msg.role {
                        Role::User => {
                            // Convert content to text (simplified for now)
                            let text = match &msg.content {
                                ContentInput::Text(t) => t.clone(),
                                ContentInput::List(items) => {
                                    // For now, just extract text items
                                    items
                                        .iter()
                                        .filter_map(|item| {
                                            if let ContentItem::Text { text } = item {
                                                Some(text.clone())
                                            } else {
                                                None
                                            }
                                        })
                                        .collect::<Vec<_>>()
                                        .join("\n")
                                }
                            };
                            trace.push(ChatMessage::User {
                                content: ChatMessageContent::Text(text),
                                name: None,
                            });
                        }
                        Role::Assistant => {
                            let text = match &msg.content {
                                ContentInput::Text(t) => t.clone(),
                                ContentInput::List(items) => {
                                    items
                                        .iter()
                                        .filter_map(|item| {
                                            if let ContentItem::Text { text } = item {
                                                Some(text.clone())
                                            } else {
                                                None
                                            }
                                        })
                                        .collect::<Vec<_>>()
                                        .join("\n")
                                }
                            };
                            trace.push(ChatMessage::Assistant {
                                content: Some(ChatMessageContent::Text(text)),
                                tool_calls: None,
                                name: None,
                                audio: None,
                                reasoning_content: None,
                                refusal: None,
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    trace
}

/// Handle OpenAI Response API - stateless only (store=false)
pub async fn handle_response(
    State(state): State<ServerState>,
    Json(payload): Json<ResponseParameters>,
) -> Result<Json<ResponseObject>, StatusCode> {
    let session_id = Uuid::new_v4();

    // Log request with path
    info!("[{}] POST /v1/responses", session_id);

    // Verify this is stateless mode
    if payload.store.unwrap_or(false) {
        error!("[{}] Stateful mode (store=true) not yet supported", session_id);
        return Err(StatusCode::NOT_IMPLEMENTED);
    }

    if payload.previous_response_id.is_some() {
        error!("[{}] Stateful mode (previous_response_id) not yet supported", session_id);
        return Err(StatusCode::NOT_IMPLEMENTED);
    }

    // Build the message trace from the request
    let trace = build_message_trace(&payload);

    // Create a new agent for this request
    let mut agent = AgentBuilder::create(state.agent_config_name.clone())
        .await
        .map_err(|e| {
            error!("[{}] Failed to create agent: {}", session_id, e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .with_traces(trace)
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

    // Collect output items (tool calls, reasoning, messages)
    let mut output = Vec::new();
    let mut final_message = String::new();
    let mut status = ReasoningStatus::Completed;

    while let Ok(event) = event_rx.recv().await {
        match event {
            // Capture assistant messages from brain results
            AgentEvent::BrainResult { thought, .. } => {
                if let Ok(msg) = thought {
                    if let ChatMessage::Assistant {
                        content: Some(ChatMessageContent::Text(text)),
                        ..
                    } = msg
                    {
                        final_message = text;
                    }
                }
            }
            // Add tool calls to output
            AgentEvent::ToolCallStarted { call, .. } => {
                info!("[{}] TOOL {}", session_id, call.tool_name);
            }
            AgentEvent::ToolCallCompleted { call, result, .. } => {
                use shai_core::tools::ToolResult;

                let (tool_status, _output_result) = match &result {
                    ToolResult::Success { output, .. } => {
                        info!("[{}] TOOL {} ✓", session_id, call.tool_name);
                        (InputItemStatus::Completed, output.clone())
                    }
                    ToolResult::Error { error, .. } => {
                        info!("[{}] TOOL {} ✗", session_id, call.tool_name);
                        (InputItemStatus::Incomplete, error.clone())
                    }
                    ToolResult::Denied => {
                        info!("[{}] TOOL {} ⊘", session_id, call.tool_name);
                        (InputItemStatus::Incomplete, "Tool call denied".to_string())
                    }
                };

                // Add function tool call to output
                output.push(ResponseOutput::FunctionToolCall(FunctionToolCall {
                    id: call.tool_call_id.clone(),
                    call_id: call.tool_call_id.clone(),
                    name: call.tool_name.clone(),
                    arguments: call.parameters.to_string(),
                    status: tool_status
                }));
            }

            // Agent completed or paused - return the result
            AgentEvent::Completed { message, success, .. } => {
                if !message.is_empty() {
                    final_message = message;
                }
                if !success {
                    status = ReasoningStatus::Failed;
                }
                info!("[{}] Completed", session_id);
                break;
            }
            AgentEvent::StatusChanged { new_status, .. } => {
                use shai_core::agent::PublicAgentState;
                if matches!(new_status, PublicAgentState::Paused { .. }) {
                    info!("[{}] Paused", session_id);
                    status = ReasoningStatus::Incomplete;
                    break;
                }
            }
            AgentEvent::Error { error } => {
                error!("[{}] Agent error: {}", session_id, error);
                status = ReasoningStatus::Failed;
                break;
            }
            _ => {}
        }
    }

    // Add final message to output
    output.push(ResponseOutput::Message(OutputMessage {
        id: Uuid::new_v4().to_string(),
        role: Role::Assistant,
        status: MessageStatus::Completed,
        content: vec![OutputContent::Text {
            text: final_message,
            annotations: vec![],
        }],
    }));

    // Build the response object
    let response = ResponseObject {
        id: session_id.to_string(),
        object: "response".to_string(),
        created_at: chrono::Utc::now().timestamp() as u32,
        model: payload.model.clone(),
        status,
        output,
        instruction: payload.instructions.clone(),
        metadata: payload.metadata.clone(),
        temperature: payload.temperature,
        max_output_tokens: payload.max_output_tokens,
        parallel_tool_calls: payload.parallel_tool_calls,
        previous_response_id: None,
        reasoning: payload.reasoning.clone(),
        text: payload.text.clone(),
        tool_choice: payload.tool_choice.clone(),
        tools: payload.tools.clone().unwrap_or_default(),
        top_p: payload.top_p,
        truncation: payload.truncation.clone(),
        user: payload.user.clone(),
        usage: Usage {
            completion_tokens: Some(0),
            prompt_tokens: Some(0),
            total_tokens: 0,
            completion_tokens_details: None,
            prompt_tokens_details: None,
        },
        incomplete_details: None,
        error: None,
    };

    Ok(Json(response))
}