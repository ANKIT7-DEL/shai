use axum::{
    extract::State,
    response::{IntoResponse, Response, sse::Event, Sse},
    Json,
};
use futures::stream::StreamExt;
use shai_core::agent::{Agent, AgentEvent};
use openai_dive::v1::resources::response::{
    items::{FunctionToolCall, InputItemStatus},
    request::ResponseParameters,
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

use crate::{ApiJson, ServerState, DisconnectionHandler, create_agent_from_model, ErrorResponse};
use super::types::{ResponseStreamEvent, build_message_trace};

/// Handle OpenAI Response API - with streaming support
pub async fn handle_response(
    State(state): State<ServerState>,
    ApiJson(payload): ApiJson<ResponseParameters>,
) -> Result<Response, ErrorResponse> {
    let session_id = Uuid::new_v4();

    // Log request with path
    info!("[{}] POST /v1/responses stream={}", session_id, payload.stream.unwrap_or(false));

    // Verify this is stateless mode
    if payload.store.unwrap_or(false) {
        error!("[{}] Stateful mode (store=true) not yet supported", session_id);
        return Err(ErrorResponse::invalid_request("Stateful mode (store=true) not yet supported".to_string()));
    }

    if payload.previous_response_id.is_some() {
        error!(
            "[{}] Stateful mode (previous_response_id) not yet supported",
            session_id
        );
        return Err(ErrorResponse::invalid_request("Stateful mode (previous_response_id) not yet supported".to_string()));
    }

    // Check if streaming is requested
    if payload.stream.unwrap_or(false) {
        handle_response_stream(state, payload, session_id).await
    } else {
        handle_response_non_stream(state, payload, session_id).await
    }
}

/// Create the response event stream from agent events
fn create_response_event_stream(
    event_rx: tokio::sync::broadcast::Receiver<AgentEvent>,
    session_id: Uuid,
    model: String,
    created_at: u32,
    payload: ResponseParameters,
) -> impl futures::stream::Stream<Item = Option<ResponseStreamEvent>> {
    futures::stream::unfold(
        (BroadcastStream::new(event_rx), false, 0u32, Vec::new(), String::new()),
        move |(mut rx, mut done, mut seq, mut output, mut accumulated_text)| {
            let session_id_str = session_id.to_string();
            let model = model.clone();
            let payload = payload.clone();

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
                                        info!("[{}] ToolCall: {}", session_id_str, call.tool_name);

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
                                                info!("[{}] ToolResult: {} ✓", session_id_str, call.tool_name);
                                                InputItemStatus::Completed
                                            }
                                            ToolResult::Error { error, .. } => {
                                                let error_oneline = error.lines().next().unwrap_or(error);
                                                info!("[{}] ToolResult: {} ✗ {}", session_id_str, call.tool_name, error_oneline);
                                                InputItemStatus::Incomplete
                                            }
                                            ToolResult::Denied => {
                                                info!("[{}] ToolResult: {} ⊘ Permission denied", session_id_str, call.tool_name);
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
}

/// Handle streaming response
async fn handle_response_stream(
    _state: ServerState,
    payload: ResponseParameters,
    session_id: Uuid,
) -> Result<Response, ErrorResponse> {
    // Build the message trace from the request
    let trace = build_message_trace(&payload);

    // Create a new agent for this request
    let mut agent = create_agent_from_model(&payload.model, &session_id).await?
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

    // Create the response event stream
    let event_stream = create_response_event_stream(event_rx, session_id, model, created_at, payload);

    // Convert to SSE stream
    let stream = event_stream.filter_map(|event_opt| async move {
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
    _state: ServerState,
    payload: ResponseParameters,
    session_id: Uuid,
) -> Result<Response, ErrorResponse> {
    use super::types::{ResponseEventData, ResponseEventType};

    // Build the message trace from the request
    let trace = build_message_trace(&payload);

    // Create a new agent for this request
    let mut agent = create_agent_from_model(&payload.model, &session_id).await?
        .with_traces(trace)
        .sudo()
        .build();

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

    // Create the event stream using the same logic as streaming
    let event_stream = create_response_event_stream(event_rx, session_id, model, created_at, payload);

    // Collect all events from the stream
    let mut events = Vec::new();
    tokio::pin!(event_stream);

    while let Some(event_result) = event_stream.next().await {
        if let Some(event) = event_result {
            events.push(event);
        }
    }

    // Find the last completed event which contains the final response
    let final_response = events
        .iter()
        .rev()
        .find_map(|event| {
            if event.event_type == ResponseEventType::ResponseCompleted {
                if let ResponseEventData::Response { response, .. } = &event.data {
                    Some(response.clone())
                } else {
                    None
                }
            } else {
                None
            }
        })
        .ok_or_else(|| ErrorResponse::internal_error("No completion event received".to_string()))?;

    Ok(Json(final_response).into_response())
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