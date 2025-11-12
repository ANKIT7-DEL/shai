pub mod client;
pub mod providers;
pub mod provider;
pub mod chat;
pub mod tool;
pub mod logging;

// Re-export our client
pub use client::LlmClient;

pub use tool::{
    ToolDescription, 
    ToolCallMethod,
    ToolBox,
    ContainsTool,
    StructuredOutputBuilder, 
    AssistantResponse, 
    IntoChatMessage, 
    FunctionCallingAutoBuilder, 
    FunctionCallingRequiredBuilder};

