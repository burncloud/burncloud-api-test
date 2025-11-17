use dotenv::dotenv;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Serialize, Clone)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Serialize, Deserialize, Clone)]
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

    let model = std::env::var("MODEL")
        .expect("MODEL not found in .env");

    println!("API高并发测试开始");
    println!("URL: {}", url);
    println!("模型: {}", model);
    println!("RPM限制: {}", rpm);

    let client = Arc::new(Client::new());
    let token = Arc::new(token);
    let url = Arc::new(url);

    let request = ChatRequest {
        model: model,
        messages: vec![
            Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }
        ],
        max_tokens: 2,
        temperature: 0.7,
    };

    let success_count = Arc::new(Mutex::new(0));
    let error_count = Arc::new(Mutex::new(0));
    let start_time = Instant::now();

    println!("启动{}个并发任务...", rpm);
    let mut tasks = Vec::new();

    for i in 1..=rpm {
        let client = client.clone();
        let token = token.clone();
        let url = url.clone();
        let request = request.clone();
        let success_count = success_count.clone();
        let error_count = error_count.clone();

        let task = tokio::spawn(async move {
            match send_request(&client, &url, &token, &request).await {
                Ok(_) => {
                    let mut count = success_count.lock().unwrap();
                    *count += 1;
                    println!("任务{} ✅", i);
                }
                Err(_) => {
                    let mut count = error_count.lock().unwrap();
                    *count += 1;
                    println!("任务{} ❌", i);
                }
            }
        });

        tasks.push(task);
    }

    for task in tasks {
        task.await?;
    }

    let duration = start_time.elapsed();
    let success = *success_count.lock().unwrap();
    let errors = *error_count.lock().unwrap();

    println!("\n测试结果:");
    println!("总请求数: {}", rpm);
    println!("成功: {}", success);
    println!("失败: {}", errors);
    println!("耗时: {:.2}秒", duration.as_secs_f64());
    let rpm_actual = success as f64 / duration.as_secs_f64() * 60.0;
    println!("计算: {} ÷ {:.2} × 60 = {:.2}", success, duration.as_secs_f64(), rpm_actual);
    println!("实际RPM: {:.2}", rpm_actual);

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

    let status = response.status();
    let body = response.text().await?;

    if !status.is_success() {
        return Err(format!("HTTP {}: {}", status, body).into());
    }

    let chat_response: ChatResponse = serde_json::from_str(&body)?;
    Ok(chat_response)
}