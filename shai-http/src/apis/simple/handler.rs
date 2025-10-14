use axum::{
    extract::{Path, State},
    response::{IntoResponse, Response, Sse},
};
use shai_llm::{ChatMessage, ChatMessageContent, ToolCall as LlmToolCall, Function};
use tracing::info;
use uuid::Uuid;

use super::types::{MultiModalQuery, Message};
use super::formatter::SimpleFormatter;
use crate::{ApiJson, ServerState, ErrorResponse, create_sse_stream};

/// Handle multimodal query - streaming response
pub async fn handle_multimodal_query_stream(
    State(state): State<ServerState>,
    session_id_param: Option<Path<String>>,
    ApiJson(payload): ApiJson<MultiModalQuery>,
) -> Result<Response, ErrorResponse> {
    let request_id = Uuid::new_v4();

    let session_id = session_id_param.map(|Path(id)| id);
    info!(
        "[{}] POST /v1/multimodal{} model={}",
        request_id,
        session_id.as_ref().map(|id| format!("/{}", id)).unwrap_or_default(),
        payload.model
    );

    // build trace from query 
    let trace = build_message_trace(&payload);

    // get current session agent
    let (request_session, actual_session_id) = state.session_manager
        .handle_request(request_id.to_string(), session_id,  trace, Some(payload.model.clone()))
        .await
        .map_err(|e| ErrorResponse::internal_error(format!("Failed to handle session: {}", e)))?;

    // Create the formatter for Simple Multimodal API
    let formatter = SimpleFormatter::new(payload.model.clone());

    // Create SSE stream - pass actual_session_id so it appears in the response 'id' field
    let stream = create_sse_stream(request_session, formatter, actual_session_id);

    Ok(Sse::new(stream).into_response())
}


/// Build message trace from query
fn build_message_trace(query: &MultiModalQuery) -> Vec<ChatMessage> {
    let mut trace = Vec::new();

    if let Some(messages) = &query.messages {
        for msg in messages.iter() {
            match msg {
                Message::User(user_msg) => {
                    trace.push(ChatMessage::User {
                        content: ChatMessageContent::Text(user_msg.message.clone()),
                        name: None,
                    });
                }
                Message::Assistant(assistant_msg) => {
                    trace.push(ChatMessage::Assistant {
                        content: Some(ChatMessageContent::Text(assistant_msg.assistant.clone())),
                        tool_calls: None,
                        name: None,
                        audio: None,
                        reasoning_content: None,
                        refusal: None,
                    });
                }
                Message::PreviousCall(prev_call) => {
                    // Convert args HashMap back to JSON for parameters
                    let parameters = serde_json::to_value(&prev_call.call.args)
                        .unwrap_or(serde_json::Value::Object(Default::default()));
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
                    let tool_result_text = prev_call
                        .result
                        .text
                        .clone()
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