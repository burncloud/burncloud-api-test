use std::env;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use std::time::Duration;

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionRequest {
    pub messages: Vec<ChatMessage>,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    pub index: i32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug)]
pub struct AzureOpenAI {
    client: Client,
    api_key: String,
    endpoint: String,
    deployment_name: String,
    api_version: String,
}

impl AzureOpenAI {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        dotenv::dotenv().ok();

        let api_key = env::var("AZURE_OPENAI_API_KEY")
            .map_err(|_| "AZURE_OPENAI_API_KEY not found in .env file")?;
        let endpoint = env::var("AZURE_OPENAI_ENDPOINT")
            .map_err(|_| "AZURE_OPENAI_ENDPOINT not found in .env file")?;
        let deployment_name = env::var("AZURE_OPENAI_DEPLOYMENT_NAME")
            .map_err(|_| "AZURE_OPENAI_DEPLOYMENT_NAME not found in .env file")?;
        let api_version = env::var("AZURE_OPENAI_API_VERSION")
            .unwrap_or_else(|_| "2024-02-15-preview".to_string());

        let client = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()?;

        Ok(Self {
            client,
            api_key,
            endpoint,
            deployment_name,
            api_version,
        })
    }

    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Box<dyn std::error::Error>> {
        let url = format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.endpoint.trim_end_matches('/'),
            self.deployment_name,
            self.api_version
        );

        println!("Azure OpenAI 请求 URL: {}", url);
        println!("请求内容: {:?}", serde_json::to_string_pretty(&request)?);

        let response = self.client
            .post(&url)
            .header("api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(format!("Azure OpenAI API 错误 {}: {}", status, error_text).into());
        }

        let response_body: ChatCompletionResponse = response.json().await?;

        println!("Azure OpenAI 响应: {}", serde_json::to_string_pretty(&response_body)?);

        Ok(response_body)
    }

    pub async fn simple_chat(
        &self,
        message: &str,
        model: Option<&str>,
        max_tokens: Option<u32>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let request = ChatCompletionRequest {
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "You are a helpful assistant.".to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: message.to_string(),
                },
            ],
            model: model.unwrap_or("gpt-4").to_string(),
            max_tokens,
            temperature: Some(0.7),
            stream: Some(false),
        };

        let response = self.chat_completion(request).await?;

        if let Some(choice) = response.choices.first() {
            Ok(choice.message.content.clone())
        } else {
            Err("没有收到响应内容".into())
        }
    }

    pub async fn test_connection(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("测试 Azure OpenAI 连接...");

        let test_message = "Hello, this is a test message. Please respond with 'Connection successful!'";
        let response = self.simple_chat(test_message, None, Some(50)).await?;

        println!("✅ Azure OpenAI 连接测试成功!");
        println!("响应内容: {}", response);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_azure_openai_creation() {
        // 这个测试需要 .env 文件中包含正确的配置
        if let Ok(azure) = AzureOpenAI::new() {
            println!("✅ Azure OpenAI 客户端创建成功");

            // 测试连接
            match azure.test_connection().await {
                Ok(_) => println!("✅ 连接测试成功"),
                Err(e) => println!("❌ 连接测试失败: {}", e),
            }
        } else {
            println!("❌ Azure OpenAI 客户端创建失败，请检查 .env 文件配置");
        }
    }
}