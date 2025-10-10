use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, Sse},
    Json,
};
use futures::stream::{Stream, StreamExt};
use shai_core::agent::{Agent, AgentEvent, AgentBuilder};
use shai_llm::{ChatMessage, ChatMessageContent, ToolCall as LlmToolCall, Function};
use std::convert::Infallible;
use tracing::{error, info, debug};
use tokio_stream::wrappers::BroadcastStream;
use uuid::Uuid;

use super::types::{MultiModalQuery, MultiModalStreamingResponse, MultiModalResponse, ResponseMessage, AssistantMessage, ToolCall, ToolCallResult};
use crate::{ServerState, DisconnectionHandler};

/// Handle multimodal query - streaming response
pub async fn handle_multimodal_query_stream(
    State(state): State<ServerState>,
    Json(payload): Json<MultiModalQuery>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, StatusCode> {
    let session_id = Uuid::new_v4();
    info!("[{}] POST /v1/multimodal model={}", session_id, payload.model);

    // Build the message trace from the query
    let trace = build_message_trace(&payload);

    // Create a new agent for this request
    let mut agent = AgentBuilder::create(state.agent_config_name.clone()).await
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
    tokio::spawn(async move {
        if let Err(e) = agent.run().await {
            error!("[{}] Agent execution error: {}", session_id, e);
        }
    });

    // Stream events back via SSE
    let model = payload.model.clone();

    // We need to track if we've seen a terminal event to close the stream after sending it
    let stream = futures::stream::unfold(
        (BroadcastStream::new(event_rx), false),
        |(mut rx, mut done)| async move {
            if done {
                return None;
            }

            match rx.next().await {
                Some(result) => {
                    // Check if this is a terminal event
                    let is_terminal = match &result {
                        Ok(AgentEvent::StatusChanged { new_status, .. }) => {
                            use shai_core::agent::PublicAgentState;
                            matches!(new_status, PublicAgentState::Paused { .. })
                        }
                        Ok(AgentEvent::Completed { .. }) => true,
                        _ => false,
                    };

                    if is_terminal {
                        done = true;
                    }

                    Some((result, (rx, done)))
                }
                None => None,
            }
        }
    ).filter_map(move |result| {
        let model = model.clone();
        async move {
        match result {
            Ok(event) => {
                // Log events with minimal information
                log_agent_event(&session_id, &event);

                // Convert AgentEvent to MultiModalStreamingResponse
                match event_to_streaming_response(&session_id, &model, event) {
                    Some(response) => {
                        match serde_json::to_string(&response) {
                            Ok(json) => {
                                Some(Ok(Event::default().data(json)))
                            }
                            Err(e) => {
                                error!("Failed to serialize response: {}", e);
                                None
                            }
                        }
                    }
                    None => None,
                }
            }
            Err(e) => {
                error!("Error in event stream: {}", e);
                None
            }
        }
    }
    });

    // Wrap the stream to detect client disconnection
    let disconnection_handler = DisconnectionHandler {
        stream: Box::pin(stream),
        controller: Some(controller),
        session_id,
        completed: false,
    };

    Ok(Sse::new(disconnection_handler))
}

/// Handle multimodal query - non-streaming response
pub async fn handle_multimodal_query(
    State(state): State<ServerState>,
    Json(payload): Json<MultiModalQuery>,
) -> Result<Json<MultiModalResponse>, StatusCode> {
    let session_id = Uuid::new_v4();
    info!("[{}] POST /v1/multimodal model={} stream={}", session_id, payload.model, payload.stream);

    // Build the message trace from the query
    let trace = build_message_trace(&payload);

    // Create a new agent for this request
    let mut agent = AgentBuilder::create(state.agent_config_name.clone()).await
        .map_err(|e| {
            error!("[{}] Failed to create agent: {}", session_id, e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .with_traces(trace)
        .sudo()
        .build();
    let mut event_rx = agent.watch();

    // Run the agent
    tokio::spawn(async move {
        if let Err(e) = agent.run().await {
            error!("[{}] Agent execution error: {}", session_id, e);
        }
    });

    // Collect all events until completion
    let mut result_messages = Vec::new();

    while let Ok(event) = event_rx.recv().await {
        log_agent_event(&session_id, &event);

        // Convert completed events to response messages
        if let AgentEvent::Completed { message, .. } = event {
            result_messages.push(ResponseMessage::Assistant(AssistantMessage {
                assistant: message,
            }));
            break;
        }
    }

    let response = MultiModalResponse {
        id: session_id.to_string(),
        model: payload.model,
        result: result_messages,
    };

    Ok(Json(response))
}

/// Convert serde_json::Value parameters to HashMap<String, String>
/// Flattens the JSON structure to string key-value pairs
fn parameters_to_args(params: &serde_json::Value) -> std::collections::HashMap<String, String> {
    use std::collections::HashMap;

    let mut args = HashMap::new();

    match params {
        serde_json::Value::Object(map) => {
            for (key, value) in map {
                let value_str = match value {
                    serde_json::Value::String(s) => s.clone(),
                    other => serde_json::to_string(other).unwrap_or_default(),
                };
                args.insert(key.clone(), value_str);
            }
        }
        _ => {
            // If it's not an object, store the entire thing as a single "params" entry
            args.insert("params".to_string(), params.to_string());
        }
    }

    args
}

/// Build message trace from query
fn build_message_trace(query: &MultiModalQuery) -> Vec<ChatMessage> {
    let mut trace = Vec::new();

    if let Some(messages) = &query.messages {
        for msg in messages.iter() {
            match msg {
                super::types::Message::User(user_msg) => {
                    trace.push(ChatMessage::User {
                        content: ChatMessageContent::Text(user_msg.message.clone()),
                        name: None,
                    });
                }
                super::types::Message::Assistant(assistant_msg) => {
                    trace.push(ChatMessage::Assistant {
                        content: Some(ChatMessageContent::Text(assistant_msg.assistant.clone())),
                        tool_calls: None,
                        name: None,
                        audio: None,
                        reasoning_content: None,
                        refusal: None,
                    });
                }
                super::types::Message::PreviousCall(prev_call) => {
                    // Convert args HashMap back to JSON for parameters
                    let parameters = serde_json::to_value(&prev_call.call.args).unwrap_or(serde_json::Value::Object(Default::default()));
                    let tool_call_id = format!("call_{}", Uuid::new_v4());

                    // Create the assistant message with tool call
                    trace.push(ChatMessage::Assistant {
                        content: None,
                        tool_calls: Some(vec![LlmToolCall {
                            id: tool_call_id.clone(),
                            r#type: "function".to_string(),
                            function: Function {
                                name: prev_call.call.tool.clone(),
                                arguments: serde_json::to_string(&parameters).unwrap_or_default(),
                            },
                        }]),
                        name: None,
                        audio: None,
                        reasoning_content: None,
                        refusal: None,
                    });

                    // Create the tool response message
                    let tool_result_text = prev_call.result.text.clone()
                        .or(prev_call.result.error.clone())
                        .unwrap_or_else(|| "No result".to_string());

                    trace.push(ChatMessage::Tool {
                        content: tool_result_text,
                        tool_call_id,
                    });
                }
            }
        }
    }

    trace
}

/// Log agent events with minimal information (no private data)
fn log_agent_event(session_id: &Uuid, event: &AgentEvent) {
    match event {
        AgentEvent::BrainResult { .. } => {
            // Event only, no content
            debug!("[{}] BrainResult", session_id);
        }
        AgentEvent::ToolCallStarted { call, .. } => {
            info!("[{}] TOOL {}", session_id, call.tool_name);
        }
        AgentEvent::ToolCallCompleted { call, result, .. } => {
            use shai_core::tools::ToolResult;
            let status = match result {
                ToolResult::Success { .. } => "✓",
                ToolResult::Error { .. } => "✗",
                ToolResult::Denied => "⊘",
            };
            info!("[{}] TOOL {} {}", session_id, call.tool_name, status);
        }
        AgentEvent::Completed { success, .. } => {
            info!("[{}] Completed: {}", session_id, if *success { "success" } else { "failed" });
        }
        AgentEvent::Error { error } => {
            error!("[{}] Error: {}", session_id, error);
        }
        _ => {}
    }
}

/// Convert AgentEvent to MultiModalStreamingResponse
fn event_to_streaming_response(
    session_id: &Uuid,
    model: &str,
    event: AgentEvent,
) -> Option<MultiModalStreamingResponse> {
    match event {
        AgentEvent::StatusChanged { .. } => {
            None
        }
        AgentEvent::BrainResult { thought, .. } => {
            if let Ok(msg) = thought {
                // Extract text content from the ChatMessage
                let text_content = match &msg {
                    ChatMessage::Assistant { content: Some(ChatMessageContent::Text(text)), .. } => {
                        Some(text.clone())
                    }
                    _ => None,
                };

                if let Some(text) = text_content {
                    debug!("[{}] BrainResult text: {}", session_id, text);
                    return Some(MultiModalStreamingResponse {
                        id: session_id.to_string(),
                        model: model.to_string(),
                        assistant: Some(text),
                        call: None,
                        result: None,
                    });
                }
            }
            None
        }
        AgentEvent::ToolCallStarted { call, .. } => {
            Some(MultiModalStreamingResponse {
                id: session_id.to_string(),
                model: model.to_string(),
                assistant: None,
                call: Some(ToolCall {
                    tool: call.tool_name.clone(),
                    args: parameters_to_args(&call.parameters),
                    output: None,
                }),
                result: None,
            })
        }
        AgentEvent::ToolCallCompleted { call, result, .. } => {
            use shai_core::tools::ToolResult;

            let (tool_result, output_str) = match &result {
                ToolResult::Success { output, .. } => (
                    ToolCallResult {
                        text: Some(output.clone()),
                        text_stream: None,
                        image: None,
                        speech: None,
                        other: None,
                        error: None,
                        extra: None,
                    },
                    output.clone(),
                ),
                ToolResult::Error { error, .. } => (
                    ToolCallResult {
                        text: None,
                        text_stream: None,
                        image: None,
                        speech: None,
                        other: None,
                        error: Some(error.clone()),
                        extra: None,
                    },
                    String::new(),
                ),
                ToolResult::Denied => (
                    ToolCallResult {
                        text: None,
                        text_stream: None,
                        image: None,
                        speech: None,
                        other: None,
                        error: Some("Tool call denied".to_string()),
                        extra: None,
                    },
                    String::new(),
                ),
            };

            Some(MultiModalStreamingResponse {
                id: session_id.to_string(),
                model: model.to_string(),
                assistant: None,
                call: Some(ToolCall {
                    tool: call.tool_name.clone(),
                    args: parameters_to_args(&call.parameters),
                    output: Some(output_str),
                }),
                result: Some(tool_result),
            })
        }
        AgentEvent::Completed { message, .. } => {
            Some(MultiModalStreamingResponse {
                id: session_id.to_string(),
                model: model.to_string(),
                assistant: Some(message),
                call: None,
                result: None,
            })
        }
        AgentEvent::Error { error } => {
            Some(MultiModalStreamingResponse {
                id: session_id.to_string(),
                model: model.to_string(),
                assistant: None,
                call: None,
                result: Some(ToolCallResult {
                    text: None,
                    text_stream: None,
                    image: None,
                    speech: None,
                    other: None,
                    error: Some(error),
                    extra: None,
                }),
            })
        }
        _ => None,
    }
}