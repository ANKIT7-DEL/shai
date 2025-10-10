use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, sse::Event, Sse},
    Json,
};
use futures::stream::StreamExt;
use shai_core::agent::{Agent, AgentEvent, AgentBuilder};
use openai_dive::v1::resources::response::{
    items::{FunctionToolCall, InputItemStatus},
    request::{ContentInput, ContentItem, ResponseInput, ResponseInputItem, ResponseParameters},
    response::{
        MessageStatus, OutputContent, OutputMessage, ReasoningStatus, ResponseObject,
        ResponseOutput, Role,
    },
};
use openai_dive::v1::resources::shared::Usage;
use shai_llm::{ChatMessage, ChatMessageContent};
use tokio_stream::wrappers::BroadcastStream;
use tracing::{error, info};
use uuid::Uuid;

use crate::{ServerState, DisconnectionHandler};
use super::types::ResponseStreamEvent;

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

/// Handle OpenAI Response API - with streaming support
pub async fn handle_response(
    State(state): State<ServerState>,
    Json(payload): Json<ResponseParameters>,
) -> Result<
    axum::response::Response,
    StatusCode,
> {
    let session_id = Uuid::new_v4();

    // Log request with path
    info!("[{}] POST /v1/responses stream={}", session_id, payload.stream.unwrap_or(false));

    // Verify this is stateless mode
    if payload.store.unwrap_or(false) {
        error!("[{}] Stateful mode (store=true) not yet supported", session_id);
        return Err(StatusCode::NOT_IMPLEMENTED);
    }

    if payload.previous_response_id.is_some() {
        error!(
            "[{}] Stateful mode (previous_response_id) not yet supported",
            session_id
        );
        return Err(StatusCode::NOT_IMPLEMENTED);
    }

    // Check if streaming is requested
    if payload.stream.unwrap_or(false) {
        handle_response_stream(state, payload, session_id).await
    } else {
        handle_response_non_stream(state, payload, session_id).await
    }
}

/// Handle streaming response
async fn handle_response_stream(
    state: ServerState,
    payload: ResponseParameters,
    session_id: Uuid,
) -> Result<axum::response::Response, StatusCode> {
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

    let controller = agent.controller();
    let event_rx = agent.watch();

    // Spawn the agent to run in the background
    let session_id_clone = session_id;
    tokio::spawn(async move {
        if let Err(e) = agent.run().await {
            error!("[{}] Agent execution error: {}", session_id_clone, e);
        }
    });

    let model = payload.model.clone();
    let created_at = chrono::Utc::now().timestamp() as u32;
    let payload_clone = payload.clone();

    // Stream events back via SSE
    let stream = futures::stream::unfold(
        (BroadcastStream::new(event_rx), false, 0u32, Vec::new(), String::new()),
        move |(mut rx, mut done, mut seq, mut output, mut accumulated_text)| {
            let session_id_str = session_id.to_string();
            let model = model.clone();
            let payload = payload_clone.clone();

            async move {
                if done {
                    return None;
                }

                // Send initial events if sequence is 0
                if seq == 0 {
                    let initial_response = build_response_object(
                        &session_id_str,
                        &model,
                        created_at,
                        ReasoningStatus::InProgress,
                        vec![],
                        &payload,
                    );

                    let event = ResponseStreamEvent::created(seq, initial_response);
                    seq += 1;

                    return Some((Some(event), (rx, done, seq, output, accumulated_text)));
                }

                match rx.next().await {
                    Some(result) => {
                        match result {
                            Ok(event) => {
                                match event {
                                    // Capture assistant messages from brain results
                                    AgentEvent::BrainResult { thought, .. } => {
                                        if let Ok(msg) = thought {
                                            if let ChatMessage::Assistant {
                                                content: Some(ChatMessageContent::Text(text)),
                                                ..
                                            } = msg
                                            {
                                                accumulated_text = text;
                                            }
                                        }
                                        return Some((None, (rx, done, seq, output, accumulated_text)));
                                    }
                                    // Tool calls
                                    AgentEvent::ToolCallStarted { call, .. } => {
                                        info!("[{}] TOOL {}", session_id_str, call.tool_name);

                                        let tool_output = ResponseOutput::FunctionToolCall(FunctionToolCall {
                                            id: call.tool_call_id.clone(),
                                            call_id: call.tool_call_id.clone(),
                                            name: call.tool_name.clone(),
                                            arguments: call.parameters.to_string(),
                                            status: InputItemStatus::InProgress,
                                        });

                                        let output_index = output.len();
                                        output.push(tool_output.clone());

                                        let event = ResponseStreamEvent::output_item_added(seq, output_index, tool_output);
                                        seq += 1;

                                        return Some((Some(event), (rx, done, seq, output, accumulated_text)));
                                    }
                                    AgentEvent::ToolCallCompleted { call, result, .. } => {
                                        use shai_core::tools::ToolResult;

                                        let tool_status = match &result {
                                            ToolResult::Success { .. } => {
                                                info!("[{}] TOOL {} ✓", session_id_str, call.tool_name);
                                                InputItemStatus::Completed
                                            }
                                            ToolResult::Error { .. } => {
                                                info!("[{}] TOOL {} ✗", session_id_str, call.tool_name);
                                                InputItemStatus::Incomplete
                                            }
                                            ToolResult::Denied => {
                                                info!("[{}] TOOL {} ⊘", session_id_str, call.tool_name);
                                                InputItemStatus::Incomplete
                                            }
                                        };

                                        // Update the tool call in output
                                        if let Some(idx) = output.iter().position(|o| {
                                            if let ResponseOutput::FunctionToolCall(tc) = o {
                                                tc.id == call.tool_call_id
                                            } else {
                                                false
                                            }
                                        }) {
                                            output[idx] = ResponseOutput::FunctionToolCall(FunctionToolCall {
                                                id: call.tool_call_id.clone(),
                                                call_id: call.tool_call_id.clone(),
                                                name: call.tool_name.clone(),
                                                arguments: call.parameters.to_string(),
                                                status: tool_status,
                                            });

                                            let event = ResponseStreamEvent::output_item_done(seq, idx, output[idx].clone());
                                            seq += 1;

                                            return Some((Some(event), (rx, done, seq, output, accumulated_text)));
                                        }

                                        return Some((None, (rx, done, seq, output, accumulated_text)));
                                    }
                                    // Agent completed
                                    AgentEvent::Completed { message, success, .. } => {
                                        if !message.is_empty() {
                                            accumulated_text = message;
                                        }
                                        info!("[{}] Completed", session_id_str);

                                        // Add final message to output
                                        let msg_output = ResponseOutput::Message(OutputMessage {
                                            id: Uuid::new_v4().to_string(),
                                            role: Role::Assistant,
                                            status: MessageStatus::Completed,
                                            content: vec![OutputContent::Text {
                                                text: accumulated_text.clone(),
                                                annotations: vec![],
                                            }],
                                        });
                                        output.push(msg_output);

                                        let final_status = if success {
                                            ReasoningStatus::Completed
                                        } else {
                                            ReasoningStatus::Failed
                                        };

                                        let final_response = build_response_object(
                                            &session_id_str,
                                            &model,
                                            created_at,
                                            final_status,
                                            output.clone(),
                                            &payload,
                                        );

                                        let event = ResponseStreamEvent::completed(seq, final_response);

                                        done = true;
                                        return Some((Some(event), (rx, done, seq, output, accumulated_text)));
                                    }
                                    AgentEvent::StatusChanged { new_status, .. } => {
                                        use shai_core::agent::PublicAgentState;
                                        if matches!(new_status, PublicAgentState::Paused { .. }) {
                                            info!("[{}] Paused", session_id_str);

                                            // Add final message to output
                                            let msg_output = ResponseOutput::Message(OutputMessage {
                                                id: Uuid::new_v4().to_string(),
                                                role: Role::Assistant,
                                                status: MessageStatus::Completed,
                                                content: vec![OutputContent::Text {
                                                    text: accumulated_text.clone(),
                                                    annotations: vec![],
                                                }],
                                            });
                                            output.push(msg_output);

                                            let final_response = build_response_object(
                                                &session_id_str,
                                                &model,
                                                created_at,
                                                ReasoningStatus::Incomplete,
                                                output.clone(),
                                                &payload,
                                            );

                                            let event = ResponseStreamEvent::completed(seq, final_response);

                                            done = true;
                                            return Some((Some(event), (rx, done, seq, output, accumulated_text)));
                                        }
                                        return Some((None, (rx, done, seq, output, accumulated_text)));
                                    }
                                    AgentEvent::Error { error } => {
                                        error!("[{}] Agent error: {}", session_id_str, error);
                                        return None;
                                    }
                                    _ => return Some((None, (rx, done, seq, output, accumulated_text))),
                                }
                            }
                            Err(e) => {
                                error!("[{}] Error in event stream: {}", session_id_str, e);
                                return None;
                            }
                        }
                    }
                    None => None,
                }
            }
        },
    )
    .filter_map(|event_opt| async move {
        event_opt.and_then(|event| {
            let event_name = event.event_name();
            match serde_json::to_string(&event) {
                Ok(json) => Some(Ok(Event::default().event(event_name).data(json))),
                Err(e) => {
                    error!("Failed to serialize event: {}", e);
                    None
                }
            }
        })
    });

    // Wrap the stream to detect client disconnection
    let disconnection_handler = DisconnectionHandler {
        stream: Box::pin(stream),
        controller: Some(controller),
        session_id,
        completed: false,
    };

    Ok(Sse::new(disconnection_handler).into_response())
}

/// Handle non-streaming response
async fn handle_response_non_stream(
    state: ServerState,
    payload: ResponseParameters,
    session_id: Uuid,
) -> Result<axum::response::Response, StatusCode> {
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

                let tool_status = match &result {
                    ToolResult::Success { .. } => {
                        info!("[{}] TOOL {} ✓", session_id, call.tool_name);
                        InputItemStatus::Completed
                    }
                    ToolResult::Error { .. } => {
                        info!("[{}] TOOL {} ✗", session_id, call.tool_name);
                        InputItemStatus::Incomplete
                    }
                    ToolResult::Denied => {
                        info!("[{}] TOOL {} ⊘", session_id, call.tool_name);
                        InputItemStatus::Incomplete
                    }
                };

                // Add function tool call to output
                output.push(ResponseOutput::FunctionToolCall(FunctionToolCall {
                    id: call.tool_call_id.clone(),
                    call_id: call.tool_call_id.clone(),
                    name: call.tool_name.clone(),
                    arguments: call.parameters.to_string(),
                    status: tool_status,
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
    let response = build_response_object(
        &session_id.to_string(),
        &payload.model,
        chrono::Utc::now().timestamp() as u32,
        status,
        output,
        &payload,
    );

    Ok(Json(response).into_response())
}

/// Helper to build a ResponseObject
fn build_response_object(
    session_id: &str,
    model: &str,
    created_at: u32,
    status: ReasoningStatus,
    output: Vec<ResponseOutput>,
    payload: &ResponseParameters,
) -> ResponseObject {
    ResponseObject {
        id: session_id.to_string(),
        object: "response".to_string(),
        created_at,
        model: model.to_string(),
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
    }
}