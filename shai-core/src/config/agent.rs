use std::collections::HashMap;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};
use shai_llm::ToolCallMethod;
use crate::tools::mcp::McpConfig;
use super::config::ShaiConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentProviderConfig {
    pub provider: String,
    pub env_vars: HashMap<String, String>,
    pub model: String,
    pub tool_method: ToolCallMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolConfig {
    pub config: McpConfig,
    #[serde(default = "default_enabled_tools")]
    pub enabled_tools: Vec<String>,
    #[serde(default)]
    pub excluded_tools: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTools {
    #[serde(default)]
    pub builtin: Vec<String>,
    #[serde(default)]
    pub builtin_excluded: Vec<String>,
    #[serde(default)]
    pub mcp: HashMap<String, McpToolConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub description: String,
    #[serde(default = "default_llm_provider")]
    pub llm_provider: AgentProviderConfig,
    #[serde(default)]
    pub tools: AgentTools,
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

fn default_llm_provider() -> AgentProviderConfig {
    // Load the default provider from ShaiConfig
    let shai_config = ShaiConfig::load()
        .unwrap_or_else(|_| ShaiConfig::default());

    let provider_config = shai_config
        .get_selected_provider()
        .expect("No provider configured in default config");

    AgentProviderConfig {
        provider: provider_config.provider.clone(),
        env_vars: provider_config.env_vars.clone(),
        model: provider_config.model.clone(),
        tool_method: provider_config.tool_method.clone(),
    }
}

fn default_system_prompt() -> String {
    "{{CODER_BASE_PROMPT}}".to_string()
}

fn default_max_tokens() -> u32 {
    4096
}

fn default_temperature() -> f32 {
    0.3
}

fn default_enabled_tools() -> Vec<String> {
    vec!["*".to_string()]
}

impl Default for AgentTools {
    fn default() -> Self {
        Self {
            builtin: vec!["*".to_string()],
            builtin_excluded: Vec::new(),
            mcp: HashMap::new(),
        }
    }
}

impl AgentConfig {
    /// Get the agents directory path
    pub fn agents_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
        let config_dir = std::env::var("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|_| {
                dirs::home_dir()
                    .map(|home| home.join(".config"))
                    .ok_or("Could not find home directory")
            })?;
        
        let agents_dir = config_dir.join("shai").join("agents");
        std::fs::create_dir_all(&agents_dir)?;
        Ok(agents_dir)
    }

    /// Get the path for a specific agent config file
    pub fn agent_config_path(agent_name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let agents_dir = Self::agents_dir()?;
        Ok(agents_dir.join(format!("{}.config", agent_name)))
    }

    /// Load an agent config from file
    pub fn load(agent_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let config_path = Self::agent_config_path(agent_name)?;
        
        if !config_path.exists() {
            return Err(format!("Agent config '{}' does not exist", agent_name).into());
        }

        let content = std::fs::read_to_string(config_path)?;
        let config: AgentConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save the agent config to file
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let config_path = Self::agent_config_path(&self.name)?;
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&config_path, content)?;
        Ok(())
    }

    /// List all available agents
    pub fn list_agents() -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let agents_dir = Self::agents_dir()?;
        let mut agents = Vec::new();

        if agents_dir.exists() {
            for entry in std::fs::read_dir(agents_dir)? {
                let entry = entry?;
                let path = entry.path();
                
                if let Some(extension) = path.extension() {
                    if extension == "config" {
                        if let Some(filename) = path.file_stem() {
                            if let Some(agent_name) = filename.to_str() {
                                agents.push(agent_name.to_string());
                            }
                        }
                    }
                }
            }
        }

        agents.sort();
        Ok(agents)
    }

    /// Check if an agent config exists
    pub fn exists(agent_name: &str) -> bool {
        Self::agent_config_path(agent_name)
            .map(|path| path.exists())
            .unwrap_or(false)
    }

    /// Delete an agent config
    pub fn delete(agent_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let config_path = Self::agent_config_path(agent_name)?;
        
        if !config_path.exists() {
            return Err(format!("Agent config '{}' does not exist", agent_name).into());
        }

        std::fs::remove_file(config_path)?;
        Ok(())
    }

    /// Check if a builtin tool is enabled
    pub fn is_builtin_tool_enabled(&self, tool_name: &str) -> bool {
        self.tools.builtin.contains(&tool_name.to_string())
    }

    /// Check if a specific MCP tool is enabled
    pub fn is_mcp_tool_enabled(&self, mcp_name: &str, tool_name: &str) -> bool {
        self.tools.mcp
            .get(mcp_name)
            .map(|mcp_tool| mcp_tool.enabled_tools.contains(&tool_name.to_string()))
            .unwrap_or(false)
    }

    /// Get all enabled MCP tool names across all MCP configs
    pub fn get_all_enabled_mcp_tools(&self) -> Vec<String> {
        self.tools.mcp
            .values()
            .flat_map(|mcp_tool| &mcp_tool.enabled_tools)
            .cloned()
            .collect()
    }
}