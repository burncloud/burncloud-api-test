use dotenv::dotenv;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::sleep;

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let token = std::env::var("API_TOKEN")
        .expect("API_TOKEN not found in .env");
    let url = std::env::var("API_URL")
        .expect("API_URL not found in .env");
    let rpm: u32 = std::env::var("RPM")
        .expect("RPM not found in .env")
        .parse()
        .expect("RPM must be a number");

    println!("API测试开始");
    println!("URL: {}", url);
    println!("RPM限制: {}", rpm);

    let client = Client::new();
    let request = ChatRequest {
        model: "gemini-2.5-flash".to_string(),
        messages: vec![
            Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }
        ],
        max_tokens: 2,
        temperature: 0.7,
    };

    let delay = Duration::from_secs(60 / rpm as u64);
    println!("请求间隔: {:?}", delay);

    match send_request(&client, &url, &token, &request).await {
        Ok(response) => {
            println!("✅ 请求成功");
            if let Some(choice) = response.choices.first() {
                println!("回复: {}", choice.message.content);
            }
        }
        Err(e) => {
            println!("❌ 请求失败: {}", e);
        }
    }

    // 测试RPM限制
    println!("\n测试RPM限制...");
    for i in 1..=3 {
        println!("第{}次请求", i);
        match send_request(&client, &url, &token, &request).await {
            Ok(_) => println!("✅ 成功"),
            Err(e) => println!("❌ 失败: {}", e),
        }

        if i < 3 {
            sleep(delay).await;
        }
    }

    Ok(())
}

async fn send_request(
    client: &Client,
    url: &str,
    token: &str,
    request: &ChatRequest,
) -> Result<ChatResponse, Box<dyn std::error::Error>> {
    let response = client
        .post(url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Content-Type", "application/json")
        .json(request)
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(format!("HTTP {}: {}", response.status(), response.text().await?).into());
    }

    let chat_response: ChatResponse = response.json().await?;
    Ok(chat_response)
}