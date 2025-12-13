use dotenv::dotenv;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Serialize, Clone)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u32,
    temperature: f32,
    stream: bool,
}

#[derive(Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct StreamDelta {
    content: Option<String>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct StreamResponse {
    choices: Vec<StreamChoice>,
}

// Claude v1/messages API structures
#[derive(Serialize, Clone)]
struct ClaudeMessage {
    role: String,
    content: Vec<ClaudeContent>,
}

#[derive(Serialize, Clone)]
struct ClaudeContent {
    r#type: String,
    text: String,
}

#[derive(Serialize, Clone)]
struct ClaudeRequest {
    model: String,
    messages: Vec<ClaudeMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    max_tokens: u32,
    temperature: f32,
    stream: bool,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ClaudeResponse {
    id: String,
    r#type: String,
    role: String,
    content: Vec<ClaudeContentResponse>,
    model: String,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    usage: ClaudeUsage,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ClaudeContentResponse {
    r#type: String,
    text: String,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ClaudeUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ClaudeStreamResponse {
    r#type: String,
    index: Option<u32>,
    delta: Option<ClaudeDelta>,
    usage: Option<ClaudeUsage>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct ClaudeDelta {
    r#type: String,
    text: String,
}

// 非流式响应结构
#[derive(Deserialize)]
#[allow(dead_code)]
struct NonStreamResponse {
    choices: Vec<NonStreamChoice>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct NonStreamChoice {
    message: Message,
    finish_reason: Option<String>,
}

#[derive(Clone)]
struct TTFTMetrics {
    duration: Duration,
}

#[derive(Clone)]
struct TPOTMetrics {
    duration: Duration,
    token_count: u32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // 检查命令行参数
    let args: Vec<String> = std::env::args().collect();

    // 如果参数是 "anthropic"，则运行Anthropic直接API测试
    if args.len() > 1 && args[1] == "anthropic" {
        return test_anthropic_direct_api().await;
    }

    // 读取API提供商配置
    let provider = std::env::var("PROVIDER")
        .unwrap_or_else(|_| "openai".to_string())
        .to_lowercase();

    // 根据提供商配置读取相应的参数
    let (token, url, model) = match provider.as_str() {
        "azure" => {
            let api_key = std::env::var("AZURE_OPENAI_API_KEY")
                .expect("AZURE_OPENAI_API_KEY not found in .env when PROVIDER=azure");
            let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT")
                .expect("AZURE_OPENAI_ENDPOINT not found in .env when PROVIDER=azure");
            let deployment_name = std::env::var("AZURE_OPENAI_DEPLOYMENT_NAME")
                .expect("AZURE_OPENAI_DEPLOYMENT_NAME not found in .env when PROVIDER=azure");
            let api_version = std::env::var("AZURE_OPENAI_API_VERSION")
                .unwrap_or_else(|_| "2025-01-01-preview".to_string());

            // Azure OpenAI的URL格式: https://your-resource.openai.azure.com/openai/deployments/{deployment-name}/chat/completions?api-version={api-version}
            let azure_url = format!("{}/openai/deployments/{}/chat/completions?api-version={}",
                endpoint.trim_end_matches('/'), deployment_name, api_version);

            (api_key, azure_url, deployment_name)
        }
        "claude" => {
            let api_key = std::env::var("AZURE_CLAUDE_API_KEY")
                .expect("AZURE_CLAUDE_API_KEY not found in .env when PROVIDER=claude");
            let endpoint = std::env::var("AZURE_CLAUDE_ENDPOINT")
                .expect("AZURE_CLAUDE_ENDPOINT not found in .env when PROVIDER=claude");
            let deployment_name = std::env::var("AZURE_CLAUDE_DEPLOYMENT_NAME")
                .expect("AZURE_CLAUDE_DEPLOYMENT_NAME not found in .env when PROVIDER=claude");

            // Anthropic原生API格式: 检查endpoint是否已包含/v1/messages
            let claude_url = if endpoint.ends_with("/v1/messages") {
                endpoint.to_string()
            } else {
                format!("{}/v1/messages", endpoint.trim_end_matches('/'))
            };

            (api_key, claude_url, deployment_name)
        }
        "anthropic" => {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .expect("ANTHROPIC_API_KEY not found in .env when PROVIDER=anthropic");
            let base_url = std::env::var("TEST_ANTHROPIC_BASE_URL")
                .unwrap_or_else(|_| "https://api.anthropic.com/v1/messages".to_string());
            let model_name = std::env::var("MODEL")
                .unwrap_or_else(|_| std::env::var("ANTHROPIC_MODEL").expect("MODEL or ANTHROPIC_MODEL must be set"));
            
            (api_key, base_url, model_name)
        }
        "openai" | _ => {
            let api_token = std::env::var("API_TOKEN")
                .expect("API_TOKEN not found in .env when PROVIDER=openai");
            let api_url = std::env::var("API_URL")
                .expect("API_URL not found in .env when PROVIDER=openai");
            let model_name = std::env::var("MODEL")
                .expect("MODEL not found in .env");

            (api_token, api_url, model_name)
        }
    };

    let rpm: u32 = std::env::var("RPM")
        .expect("RPM not found in .env")
        .parse()
        .expect("RPM must be a number");

    let max_tokens: u32 = std::env::var("MAX_TOKENS")
        .expect("MAX_TOKENS not found in .env")
        .parse()
        .expect("MAX_TOKENS must be a number");
    let stream: bool = std::env::var("STREAM")
        .unwrap_or_else(|_| "true".to_string())
        .parse()
        .expect("STREAM must be true or false");
    let prompt = std::env::var("PROMPT")
        .unwrap_or_else(|_| "Hello, please introduce yourself.".to_string());

    println!("API高并发测试开始 ({})", if stream { "含TTFT+TPOT测量" } else { "响应时间测量" });
    println!("提供商: {}", match provider.as_str() {
        "azure" => "Azure OpenAI (GPT)",
        "claude" => "Azure Claude (Anthropic)",
        _ => "OpenAI"
    });
    println!("URL: {}", url);
    println!("模型: {}", model);
    println!("RPM限制: {}", rpm);
    println!("最大Token数: {}", max_tokens);
    println!("流式响应: {}", stream);
    println!("提示词: {}", prompt);

    // 配置优化的HTTP客户端
    let client = Arc::new(
        Client::builder()
            .timeout(Duration::from_secs(300))           // 总体超时
            .connect_timeout(Duration::from_secs(300))   // 连接超时
            .timeout(Duration::from_secs(300))            // 读取超时
            .pool_max_idle_per_host(300)                  // 连接池最大空闲连接
            .pool_idle_timeout(Duration::from_secs(300))  // 空闲连接超时
            .tcp_keepalive(Duration::from_secs(300))      // TCP Keep-Alive
            .http2_keep_alive_interval(Duration::from_secs(300)) // HTTP/2 Keep-Alive
            .http2_keep_alive_timeout(Duration::from_secs(300))  // HTTP/2 Keep-Alive超时
            .redirect(reqwest::redirect::Policy::limited(3)) // 限制重定向
            .build()
            .expect("Failed to create HTTP client")
    );
    let token = Arc::new(token);
    let url = Arc::new(url);

    // 调试输出
    println!("调试信息:");
    println!("Provider: {}", provider);
    println!("Token: {}...", &token[..std::cmp::min(10, token.len())]);

    // 根据提供商创建不同的请求格式
    let request_data = if provider == "claude" || provider == "anthropic" {
        // Claude v1/messages格式 - 修复system角色问题
        let claude_request = ClaudeRequest {
            model: model.clone(),
            messages: vec![
                ClaudeMessage {
                    role: "user".to_string(),
                    content: vec![
                        ClaudeContent {
                            r#type: "text".to_string(),
                            text: format!("hi\n\n{}", prompt),
                        }
                    ],
                }
            ],
            system: Some(prompt), // 使用顶级的system参数而不是system角色的消息
            max_tokens: max_tokens,
            temperature: 0.7,
            stream: stream,
        };
        serde_json::to_value(claude_request).expect("Failed to serialize Claude request")
    } else {
        // OpenAI/Azure OpenAI格式
        let request_model = if provider == "azure" { "".to_string() } else { model.clone() };
        
        // 兼容性处理: 如果模型名包含 "claude"，且正在使用 OpenAI 协议 (如 New-API 中转)
        // 最好不要发送 "system" role，因为部分中转可能会直接透传给 Claude 导致 "Unexpected role system" 错误。
        // 解决方案是将 system prompt 合并到 user message 中。
        let messages = if model.to_lowercase().contains("claude") {
            vec![
                Message {
                    role: "user".to_string(),
                    content: format!("{}\n\nhi", prompt),
                }
            ]
        } else {
            vec![
                Message {
                    role: "system".to_string(),
                    content: prompt,
                },
                Message {
                    role: "user".to_string(),
                    content: "hi".to_string(),
                }
            ]
        };

        let chat_request = ChatRequest {
            model: request_model,
            messages: messages,
            max_tokens: max_tokens,
            temperature: 0.7,
            stream: stream,
        };
        serde_json::to_value(chat_request).expect("Failed to serialize ChatRequest")
    };

    let success_count = Arc::new(Mutex::new(0));
    let error_count = Arc::new(Mutex::new(0));
    let ttft_metrics = Arc::new(Mutex::new(Vec::new()));
    let tpot_metrics = Arc::new(Mutex::new(Vec::new()));
    let start_time = Instant::now();

    println!("启动{}个并发任务...", rpm);
    let mut tasks = Vec::new();

    for i in 1..=rpm {
        let client = client.clone();
        let token = token.clone();
        let url = url.clone();
        let request_data = request_data.clone();
        let success_count = success_count.clone();
        let error_count = error_count.clone();
        let ttft_metrics = ttft_metrics.clone();
        let tpot_metrics = tpot_metrics.clone();
        let provider = provider.clone();

        let task = tokio::spawn(async move {
            match send_request_with_retry(&client, &url, &token, &request_data, &provider, 2).await {
                Ok((ttft, tpot_opt, content)) => {
                    let mut count = success_count.lock().unwrap();
                    *count += 1;
                    let mut ttft_list = ttft_metrics.lock().unwrap();
                    ttft_list.push(TTFTMetrics { duration: ttft });

                    // 如果是流式响应且有TPOT数据，则收集TPOT指标
                    if let Some(ref tpot) = tpot_opt {
                        let mut tpot_list = tpot_metrics.lock().unwrap();
                        tpot_list.push(tpot.clone());
                    }

                    // 只显示前5个响应的内容，避免输出过多
                    if i <= 5 {
                        if stream {
                            if let Some(ref tpot) = tpot_opt {
                                println!("任务{} ✅ TTFT: {:?} | TPOT: {:?} ({:.1} tok/s) | 回复: {}",
                                    i, ttft, tpot.duration, tpot.token_count as f64 / tpot.duration.as_secs_f64(), content);
                            } else {
                                println!("任务{} ✅ TTFT: {:?} | 回复: {}", i, ttft, content);
                            }
                        } else {
                            println!("任务{} ✅ 响应时间: {:?} | 回复: {}", i, ttft, content);
                        }
                    } else {
                        if stream {
                            if let Some(ref tpot) = tpot_opt {
                                println!("任务{} ✅ TTFT: {:?} | TPOT: {:?} ({:.1} tok/s)",
                                    i, ttft, tpot.duration, tpot.token_count as f64 / tpot.duration.as_secs_f64());
                            } else {
                                println!("任务{} ✅ TTFT: {:?}", i, ttft);
                            }
                        } else {
                            println!("任务{} ✅ 响应时间: {:?}", i, ttft);
                        }
                    }
                }
                Err(e) => {
                    let mut count = error_count.lock().unwrap();
                    *count += 1;
                    let error_msg = format!("{}", e);
                    println!("任务{} ❌ {}", i, error_msg);
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
    let metrics = ttft_metrics.lock().unwrap();
    let tpot_list = tpot_metrics.lock().unwrap();

    println!("\n测试结果:");
    println!("总请求数: {}", rpm);
    println!("成功: {}", success);
    println!("失败: {}", errors);
    println!("耗时: {:.2}秒", duration.as_secs_f64());
    let rpm_actual = success as f64 / duration.as_secs_f64() * 60.0;
    println!("计算: {} ÷ {:.2} × 60 = {:.2}", success, duration.as_secs_f64(), rpm_actual);
    println!("实际RPM: {:.2}", rpm_actual);

    // 响应时间统计
    if !metrics.is_empty() {
        let total_response_time: Duration = metrics.iter().map(|m| m.duration).sum();
        let avg_response_time = total_response_time / metrics.len() as u32;
        let min_response_time = metrics.iter().map(|m| m.duration).min().unwrap();
        let max_response_time = metrics.iter().map(|m| m.duration).max().unwrap();

        // 计算百分位数
        let mut sorted_response_times: Vec<Duration> = metrics.iter().map(|m| m.duration).collect();
        sorted_response_times.sort();
        let p50_idx = sorted_response_times.len() * 50 / 100;
        let p95_idx = sorted_response_times.len() * 95 / 100;
        let p99_idx = sorted_response_times.len() * 99 / 100;

        let metric_name = if stream { "TTFT (Time To First Token)" } else { "响应时间" };
        println!("\n📊 {} 性能统计:", metric_name);
        println!("成功响应数: {}", metrics.len());
        println!("平均{}: {:?} ({:.2}ms)", if stream { "TTFT" } else { "响应时间" }, avg_response_time, avg_response_time.as_millis());
        println!("最小{}: {:?} ({:.2}ms)", if stream { "TTFT" } else { "响应时间" }, min_response_time, min_response_time.as_millis());
        println!("最大{}: {:?} ({:.2}ms)", if stream { "TTFT" } else { "响应时间" }, max_response_time, max_response_time.as_millis());
        println!("中位数(P50): {:?} ({:.2}ms)", sorted_response_times[p50_idx], sorted_response_times[p50_idx].as_millis());
        if sorted_response_times.len() > 20 {
            println!("P95: {:?} ({:.2}ms)", sorted_response_times[p95_idx], sorted_response_times[p95_idx].as_millis());
            println!("P99: {:?} ({:.2}ms)", sorted_response_times[p99_idx], sorted_response_times[p99_idx].as_millis());
        }

        // 性能评级
        let avg_ms = avg_response_time.as_millis() as f64;
        let performance_rating = if avg_ms < 500.0 {
            "🟢 优秀 (<500ms)"
        } else if avg_ms < 1000.0 {
            "🟡 良好 (500ms-1s)"
        } else if avg_ms < 2000.0 {
            "🟠 一般 (1-2s)"
        } else {
            "🔴 需要优化 (>2s)"
        };

        println!("\n🎯 性能评级: {}", performance_rating);
        println!("💡 连接池和超时优化已启用");
        println!("🔄 重试机制已启用 (最多2次重试)");

        // TPOT统计 (仅对流式响应)
        if stream && !tpot_list.is_empty() {
            println!("\n📊 TPOT (Time Per Output Token) 性能统计:");
            let total_tpot: Duration = tpot_list.iter().map(|t| t.duration).sum();
            let avg_tpot = total_tpot / tpot_list.len() as u32;
            let min_tpot = tpot_list.iter().map(|t| t.duration).min().unwrap();
            let max_tpot = tpot_list.iter().map(|t| t.duration).max().unwrap();
            let total_tokens: u32 = tpot_list.iter().map(|t| t.token_count).sum();

            // 计算TPOT百分位数
            let mut sorted_tpot_times: Vec<Duration> = tpot_list.iter().map(|t| t.duration).collect();
            sorted_tpot_times.sort();
            let p50_idx = sorted_tpot_times.len() * 50 / 100;
            let p95_idx = sorted_tpot_times.len() * 95 / 100;
            let p99_idx = sorted_tpot_times.len() * 99 / 100;

            println!("成功TPOT样本数: {}", tpot_list.len());
            println!("平均TPOT: {:?} ({:.2}ms)", avg_tpot, avg_tpot.as_millis());
            println!("最小TPOT: {:?} ({:.2}ms)", min_tpot, min_tpot.as_millis());
            println!("最大TPOT: {:?} ({:.2}ms)", max_tpot, max_tpot.as_millis());
            println!("中位数TPOT(P50): {:?} ({:.2}ms)", sorted_tpot_times[p50_idx], sorted_tpot_times[p50_idx].as_millis());

            if sorted_tpot_times.len() > 20 {
                println!("P95 TPOT: {:?} ({:.2}ms)", sorted_tpot_times[p95_idx], sorted_tpot_times[p95_idx].as_millis());
                println!("P99 TPOT: {:?} ({:.2}ms)", sorted_tpot_times[p99_idx], sorted_tpot_times[p99_idx].as_millis());
            }

            // Token吞吐量统计
            let avg_tokens_per_second = tpot_list.iter()
                .map(|t| t.token_count as f64 / t.duration.as_secs_f64())
                .sum::<f64>() / tpot_list.len() as f64;

            let total_tokens_per_second = total_tokens as f64 / duration.as_secs_f64();

            println!("\n🚀 Token吞吐量统计:");
            println!("总Token数: {}", total_tokens);
            println!("平均每秒Token数: {:.1} tok/s", avg_tokens_per_second);
            println!("整体吞吐量: {:.1} tok/s", total_tokens_per_second);

            // TPOT性能评级
            let avg_tpot_ms = avg_tpot.as_millis() as f64;
            let tpot_rating = if avg_tpot_ms < 50.0 {
                "🟢 优秀 (<50ms/token)"
            } else if avg_tpot_ms < 100.0 {
                "🟡 良好 (50-100ms/token)"
            } else if avg_tpot_ms < 200.0 {
                "🟠 一般 (100-200ms/token)"
            } else {
                "🔴 需要优化 (>200ms/token)"
            };

            println!("\n🎯 TPOT性能评级: {}", tpot_rating);
        }
    }

    Ok(())
}

async fn send_request_with_retry(
    client: &Client,
    url: &str,
    token: &str,
    request: &serde_json::Value,
    provider: &str,
    max_retries: u32,
) -> Result<(Duration, Option<TPOTMetrics>, String), Box<dyn std::error::Error + Send + Sync>> {
    let _start_time = Instant::now();
    let mut last_error: Option<Box<dyn std::error::Error + Send + Sync>> = None;

    for attempt in 0..=max_retries {
        let request_start = Instant::now();

        let result = if provider == "claude" || provider == "anthropic" {
            if request.get("stream").and_then(|v| v.as_bool()).unwrap_or(false) {
                send_claude_request_stream_once(client, url, token, request, provider).await
            } else {
                send_claude_request_non_stream_once(client, url, token, request, provider).await
            }
        } else {
            if request.get("stream").and_then(|v| v.as_bool()).unwrap_or(false) {
                send_request_stream_once(client, url, token, request).await
            } else {
                send_request_non_stream_once(client, url, token, request).await
            }
        };

        match result {
            Ok((response_time, tpot_opt, content)) => {
                // 记录重试信息
                if attempt > 0 {
                    println!("  🔄 重试{}次后成功，总耗时: {:?}", attempt, request_start.elapsed());
                }
                return Ok((response_time, tpot_opt, content));
            }
            Err(e) => {
                last_error = Some(e);
                if attempt < max_retries {
                    let wait_duration = Duration::from_millis(100 * (2_u64.pow(attempt as u32))); // 指数退避
                    println!("  ⚠️  尝试{}/{} 失败，{}ms后重试...", attempt + 1, max_retries + 1, wait_duration.as_millis());
                    tokio::time::sleep(wait_duration).await;
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| "Unknown error".into()))
}

async fn send_request_stream_once(
    client: &Client,
    url: &str,
    token: &str,
    request: &serde_json::Value,
) -> Result<(Duration, Option<TPOTMetrics>, String), Box<dyn std::error::Error + Send + Sync>> {
    let start_time = Instant::now();

    // 根据URL判断认证方式
    let is_anthropic = url.contains("/v1/messages");
    let is_azure = url.contains("openai.azure.com") || url.contains("api-version=");

    let mut req_builder = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Connection", "keep-alive")
        .header("Accept", "text/event-stream");

    if is_anthropic {
        // 中转平台使用标准OpenAI认证方式：Bearer token
        req_builder = req_builder.header("Authorization", format!("Bearer {}", token));
    } else if is_azure {
        req_builder = req_builder.header("api-key", token);
    } else {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", token));
    }

    // 修复: 如果模型是 Claude，即使是 OpenAI 兼容接口，部分中转也需要 anthropic-version 头
    if let Some(model) = request.get("model").and_then(|m| m.as_str()) {
        if model.to_lowercase().contains("claude") {
            req_builder = req_builder.header("anthropic-version", "2023-06-01");
        }
    }

    let response = req_builder
        .json(request)
        .timeout(Duration::from_secs(25)) // 请求级别超时
        .send()
        .await?;

    let status = response.status();

    if !status.is_success() {
        let body = response.text().await?;
        return Err(format!("HTTP {}: {}", status, body).into());
    }

    // 处理流式响应，测量第一个token的时间并收集完整内容
    let mut byte_stream = response.bytes_stream();
    let mut first_chunk_received = false;
    let mut ttft = Duration::ZERO;
    let mut buffer = Vec::new();
    let mut complete_response = String::new();

    // TPOT计算相关变量
    let mut tpot_start_time = Instant::now();
    let mut token_count = 0u32;
    let mut first_token_received = false;

    // 优化的流处理：更大的缓冲区，更快的首字检测
    while let Some(chunk_result) = byte_stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                let chunk_str = String::from_utf8_lossy(&chunk);
                buffer.extend_from_slice(&chunk);

                // 检查是否收到有效的SSE数据
                if !first_chunk_received {
                    let buffer_str = String::from_utf8_lossy(&buffer);

                    if buffer_str.contains("data: ") &&
                       !buffer_str.contains("[DONE]") &&
                       buffer_str.len() > 12 { // 至少包含基本的数据格式

                        ttft = start_time.elapsed();
                        first_chunk_received = true;
                    }
                }

                // 收集响应内容
                complete_response.push_str(&chunk_str);

                // TPOT计算：解析token并计数
                let lines: Vec<&str> = chunk_str.lines().collect();
                for line in lines {
                    if line.starts_with("data: ") && !line.contains("[DONE]") {
                        let data_str = &line[6..]; // 移除"data: "

                        // 尝试解析JSON数据
                        if let Ok(stream_response) = serde_json::from_str::<StreamResponse>(data_str) {
                            for choice in stream_response.choices {
                                if let Some(content) = choice.delta.content {
                                    // 如果是第一个有内容的token，开始TPOT计时
                                    if !first_token_received && !content.trim().is_empty() {
                                        tpot_start_time = Instant::now();
                                        first_token_received = true;
                                    }

                                    // 计算token数量（简单按字符数估算，可以更精确）
                                    token_count += content.chars().count() as u32;
                                }
                            }
                        }
                    }
                }

                // 检查是否收到结束标记
                if complete_response.contains("[DONE]") {
                    break;
                }
            }
            Err(e) => {
                return Err(format!("Stream error: {}", e).into());
            }
        }

        // 防止无限等待
        if start_time.elapsed() > Duration::from_secs(20) {
            break;
        }
    }

    if !first_chunk_received {
        return Err("No data received from stream".into());
    }

    // 解析响应内容，提取实际的消息内容
    let extracted_content = extract_message_content(&complete_response);

    // 计算TPOT指标
    let tpot_metrics = if first_token_received && token_count > 0 {
        let tpot_duration = tpot_start_time.elapsed();
        Some(TPOTMetrics {
            duration: tpot_duration,
            token_count: token_count,
        })
    } else {
        None
    };

    Ok((ttft, tpot_metrics, extracted_content))
}

async fn send_request_non_stream_once(
    client: &Client,
    url: &str,
    token: &str,
    request: &serde_json::Value,
) -> Result<(Duration, Option<TPOTMetrics>, String), Box<dyn std::error::Error + Send + Sync>> {
    let start_time = Instant::now();

    // 根据URL判断认证方式
    let is_anthropic = url.contains("/v1/messages");
    let is_azure = url.contains("openai.azure.com") || url.contains("api-version=");

    let mut req_builder = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Connection", "keep-alive");

    if is_anthropic {
        // 中转平台使用标准OpenAI认证方式：Bearer token
        req_builder = req_builder.header("Authorization", format!("Bearer {}", token));
    } else if is_azure {
        req_builder = req_builder.header("api-key", token);
    } else {
        req_builder = req_builder.header("Authorization", format!("Bearer {}", token));
    }

    // 修复: 如果模型是 Claude，即使是 OpenAI 兼容接口，部分中转也需要 anthropic-version 头
    if let Some(model) = request.get("model").and_then(|m| m.as_str()) {
        if model.to_lowercase().contains("claude") {
            req_builder = req_builder.header("anthropic-version", "2023-06-01");
        }
    }

    let response = req_builder
        .json(request)
        .timeout(Duration::from_secs(25)) // 请求级别超时
        .send()
        .await?;

    let status = response.status();

    if !status.is_success() {
        let body = response.text().await?;
        return Err(format!("HTTP {}: {}", status, body).into());
    }

    // 非流式响应，等待完整响应
    let response_text = response.text().await?;
    let response_time = start_time.elapsed();

    // 解析非流式响应
    if let Ok(chat_response) = serde_json::from_str::<NonStreamResponse>(&response_text) {
        if let Some(choice) = chat_response.choices.first() {
            Ok((response_time, None, choice.message.content.clone()))
        } else {
            Err("No choices in response".into())
        }
    } else {
        Err("Failed to parse non-stream response".into())
    }
}

fn extract_message_content(sse_data: &str) -> String {
    let mut content_parts = Vec::new();

    // 按行分割SSE数据
    for line in sse_data.lines() {
        // 跳过空行和注释
        if line.trim().is_empty() || line.starts_with(':') {
            continue;
        }

        // 查找data:行
        if line.starts_with("data: ") {
            let data_str = &line[6..]; // 移除"data: "

            // 跳过[DONE]标记
            if data_str.trim() == "[DONE]" {
                continue;
            }

            // 解析JSON数据
            if let Ok(stream_response) = serde_json::from_str::<StreamResponse>(data_str) {
                // 提取内容
                for choice in stream_response.choices {
                    if let Some(content) = choice.delta.content {
                        content_parts.push(content);
                    }
                }
            }
        }
    }

    // 合并所有内容部分
    content_parts.join("")
}

// Claude-specific request functions
async fn send_claude_request_stream_once(
    client: &Client,
    url: &str,
    token: &str,
    request: &serde_json::Value,
    provider: &str,
) -> Result<(Duration, Option<TPOTMetrics>, String), Box<dyn std::error::Error + Send + Sync>> {
    let start_time = Instant::now();

    let mut req_builder = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("Accept", "text/event-stream");

    if provider == "anthropic" {
        // New-API 中转通常希望使用 Authorization 头进行鉴权，
        // 但同时也需要 anthropic-version 头来满足 upstream 的要求。
        // 我们不再发送 x-api-key，以免 token 被错误地透传给 upstream。
        req_builder = req_builder
            .header("Authorization", format!("Bearer {}", token))
            .header("anthropic-version", "2023-06-01");
    } else {
        // Azure Claude 或其他兼容接口
        // Use Bearer token for non-Azure endpoints (like New-API or native Anthropic)
        if url.contains("openai.azure.com") {
            req_builder = req_builder.header("api-key", token);
        } else {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", token));
        }
    }

    let response = req_builder
        .json(request)
        .timeout(Duration::from_secs(25))
        .send()
        .await?;

    let status = response.status();
    let content_type = response.headers().get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    if !status.is_success() {
        let body = response.text().await?;
        return Err(format!("HTTP {}: {}", status, body).into());
    }

    // 处理Claude的流式响应
    let mut byte_stream = response.bytes_stream();
    let mut first_chunk_received = false;
    let mut ttft = Duration::ZERO;
    let mut buffer = Vec::new();
    let mut complete_response = String::new();

    // TPOT计算相关变量
    let mut tpot_start_time = Instant::now();
    let mut token_count = 0u32;
    let mut first_token_received = false;

    // Debug: 检查是否是JSON响应 (有些代理即使在流式模式下出错也会返回JSON)
    if content_type.contains("application/json") {
         // 可能是一个非流式的错误响应
         // 继续读取流，但要注意它可能不是SSE
    }

    while let Some(chunk_result) = byte_stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                let chunk_str = String::from_utf8_lossy(&chunk);
                buffer.extend_from_slice(&chunk);

                // 检查是否收到有效的SSE数据
                if !first_chunk_received {
                    let buffer_str = String::from_utf8_lossy(&buffer);

                    if buffer_str.contains("data: ") {
                        ttft = start_time.elapsed();
                        first_chunk_received = true;
                    } else if buffer.len() > 0 {
                        // 收到数据但不是SSE格式，打印出来调试
                        let debug_preview: String = buffer_str.chars().take(200).collect();
                        println!("⚠️ [Debug] 收到非SSE数据 (Content-Type: {}): {}", content_type, debug_preview);
                        
                        // 如果看起来像JSON错误，尝试解析
                        if debug_preview.trim().starts_with('{') {
                             return Err(format!("Received JSON instead of SSE: {}", debug_preview).into());
                        }
                    }
                }

                // 收集响应内容
                complete_response.push_str(&chunk_str);

                // TPOT计算：解析Claude的token并计数
                let lines: Vec<&str> = chunk_str.lines().collect();
                for line in lines {
                    if line.starts_with("data: ") && !line.contains("[DONE]") {
                        let data_str = &line[6..]; // 移除"data: "

                        // 尝试解析Claude流式响应JSON数据
                        if let Ok(claude_stream) = serde_json::from_str::<ClaudeStreamResponse>(data_str) {
                            if let Some(delta) = claude_stream.delta {
                                if !delta.text.trim().is_empty() {
                                    // 如果是第一个有内容的token，开始TPOT计时
                                    if !first_token_received {
                                        tpot_start_time = Instant::now();
                                        first_token_received = true;
                                    }

                                    // 计算token数量（简单按字符数估算）
                                    token_count += delta.text.chars().count() as u32;
                                }
                            }
                        }
                    }
                }

                // 检查是否收到结束标记
                if complete_response.contains("[DONE]") {
                    break;
                }
            }
            Err(e) => {
                return Err(format!("Claude stream error: {}", e).into());
            }
        }

        // 防止无限等待
        if start_time.elapsed() > Duration::from_secs(20) {
            break;
        }
    }

    if !first_chunk_received {
        return Err("No data received from Claude stream".into());
    }

    // 解析Claude响应内容
    let extracted_content = extract_claude_message_content(&complete_response);

    // 计算TPOT指标
    let tpot_metrics = if first_token_received && token_count > 0 {
        let tpot_duration = tpot_start_time.elapsed();
        Some(TPOTMetrics {
            duration: tpot_duration,
            token_count: token_count,
        })
    } else {
        None
    };

    Ok((ttft, tpot_metrics, extracted_content))
}

async fn send_claude_request_non_stream_once(
    client: &Client,
    url: &str,
    token: &str,
    request: &serde_json::Value,
    provider: &str,
) -> Result<(Duration, Option<TPOTMetrics>, String), Box<dyn std::error::Error + Send + Sync>> {
    let start_time = Instant::now();

    let mut req_builder = client
        .post(url)
        .header("Content-Type", "application/json");

    if provider == "anthropic" {
        // New-API 中转通常希望使用 Authorization 头进行鉴权，
        // 但同时也需要 anthropic-version 头来满足 upstream 的要求。
        // 我们不再发送 x-api-key，以免 token 被错误地透传给 upstream。
        req_builder = req_builder
            .header("Authorization", format!("Bearer {}", token))
            .header("anthropic-version", "2023-06-01");
    } else {
        // Use Bearer token for non-Azure endpoints
        if url.contains("openai.azure.com") {
            req_builder = req_builder.header("api-key", token);
        } else {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", token));
        }
    }

    let response = req_builder
        .json(request)
        .timeout(Duration::from_secs(25))
        .send()
        .await?;

    let status = response.status();

    if !status.is_success() {
        let body = response.text().await?;
        return Err(format!("HTTP {}: {}", status, body).into());
    }

    // 非流式响应，等待完整响应
    let response_text = response.text().await?;
    let response_time = start_time.elapsed();

    // 解析Claude非流式响应
    if let Ok(claude_response) = serde_json::from_str::<ClaudeResponse>(&response_text) {
        let content: String = claude_response.content
            .iter()
            .filter_map(|c| c.text.clone().into())
            .collect();
        Ok((response_time, None, content))
    } else {
        Err("Failed to parse Claude non-stream response".into())
    }
}

fn extract_claude_message_content(sse_data: &str) -> String {
    let mut content_parts = Vec::new();

    // 按行分割SSE数据
    for line in sse_data.lines() {
        // 跳过空行和注释
        if line.trim().is_empty() || line.starts_with(':') {
            continue;
        }

        // 查找data:行
        if line.starts_with("data: ") {
            let data_str = &line[6..]; // 移除"data: "

            // 跳过[DONE]标记
            if data_str.trim() == "[DONE]" {
                continue;
            }

            // 解析Claude流式响应JSON数据
            if let Ok(claude_stream) = serde_json::from_str::<ClaudeStreamResponse>(data_str) {
                // 提取内容
                if let Some(delta) = claude_stream.delta {
                    if !delta.text.trim().is_empty() {
                        content_parts.push(delta.text);
                    }
                }
            }
        }
    }

    // 合并所有内容部分
    content_parts.join("")
}

// 新增：Anthropic直接API测试函数
async fn test_anthropic_direct_api() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    println!("🔥 开始Anthropic直接API测试");

    // 读取Anthropic API配置
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("ANTHROPIC_API_KEY not found in .env");
    let model = std::env::var("ANTHROPIC_MODEL")
        .unwrap_or_else(|_| "claude-3-5-sonnet-20241022".to_string());
    let max_tokens: u32 = std::env::var("MAX_TOKENS")
        .unwrap_or_else(|_| "500".to_string())
        .parse()
        .expect("MAX_TOKENS must be a number");
    let stream: bool = std::env::var("STREAM")
        .unwrap_or_else(|_| "true".to_string())
        .parse()
        .expect("STREAM must be true or false");
    let prompt = std::env::var("PROMPT")
        .unwrap_or_else(|_| "写一篇300字小说".to_string());

    let url = "https://api.anthropic.com/v1/messages";

    println!("模型: {}", model);
    println!("最大Token数: {}", max_tokens);
    println!("流式响应: {}", stream);
    println!("提示词: {}", prompt);

    // 配置HTTP客户端
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client");

    // 创建正确的Anthropic API请求
    let claude_request = ClaudeRequest {
        model: model.clone(),
        messages: vec![
            ClaudeMessage {
                role: "user".to_string(),
                content: vec![
                    ClaudeContent {
                        r#type: "text".to_string(),
                        text: format!("hi\n\n{}", prompt),
                    }
                ],
            }
        ],
        system: Some(prompt), // 正确使用system参数
        max_tokens: max_tokens,
        temperature: 0.7,
        stream: stream,
    };

    println!("\n🚀 发送请求到Anthropic API...");

    let result = if stream {
        send_anthropic_stream_request(&client, url, &api_key, &claude_request).await
    } else {
        send_anthropic_non_stream_request(&client, url, &api_key, &claude_request).await
    };

    match result {
        Ok((response_time, content)) => {
            println!("✅ 请求成功!");
            println!("⏱️  响应时间: {:?}", response_time);
            println!("📝 回复内容: {}\n", content);
            println!("🎯 Anthropic API测试完成");
        }
        Err(e) => {
            println!("❌ 请求失败: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

async fn send_anthropic_stream_request(
    client: &Client,
    url: &str,
    api_key: &str,
    request: &ClaudeRequest,
) -> Result<(Duration, String), Box<dyn std::error::Error + Send + Sync>> {
    let start_time = Instant::now();

    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("Accept", "text/event-stream")
        .json(request)
        .timeout(Duration::from_secs(25))
        .send()
        .await?;

    let status = response.status();

    if !status.is_success() {
        let body = response.text().await?;
        return Err(format!("HTTP {}: {}", status, body).into());
    }

    // 处理流式响应
    let mut byte_stream = response.bytes_stream();
    let mut complete_response = String::new();
    let mut content_parts = Vec::new();

    while let Some(chunk_result) = byte_stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                let chunk_str = String::from_utf8_lossy(&chunk);
                complete_response.push_str(&chunk_str);

                // 解析流式数据
                let lines: Vec<&str> = chunk_str.lines().collect();
                for line in lines {
                    if line.starts_with("data: ") && !line.contains("[DONE]") {
                        let data_str = &line[6..];

                        if let Ok(claude_stream) = serde_json::from_str::<ClaudeStreamResponse>(data_str) {
                            if let Some(delta) = claude_stream.delta {
                                if !delta.text.trim().is_empty() {
                                    content_parts.push(delta.text.clone());
                                    print!("{}", delta.text); // 实时输出
                                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                                }
                            }
                        }
                    }
                }

                if complete_response.contains("[DONE]") {
                    break;
                }
            }
            Err(e) => {
                return Err(format!("Stream error: {}", e).into());
            }
        }

        if start_time.elapsed() > Duration::from_secs(20) {
            break;
        }
    }

    println!(); // 换行

    let response_time = start_time.elapsed();
    let content = content_parts.join("");

    Ok((response_time, content))
}

async fn send_anthropic_non_stream_request(
    client: &Client,
    url: &str,
    api_key: &str,
    request: &ClaudeRequest,
) -> Result<(Duration, String), Box<dyn std::error::Error + Send + Sync>> {
    let start_time = Instant::now();

    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(request)
        .timeout(Duration::from_secs(25))
        .send()
        .await?;

    let status = response.status();

    if !status.is_success() {
        let body = response.text().await?;
        return Err(format!("HTTP {}: {}", status, body).into());
    }

    let response_text = response.text().await?;
    let response_time = start_time.elapsed();

    // 解析响应
    if let Ok(claude_response) = serde_json::from_str::<ClaudeResponse>(&response_text) {
        let content: String = claude_response.content
            .iter()
            .filter_map(|c| c.text.clone().into())
            .collect();
        Ok((response_time, content))
    } else {
        Err("Failed to parse Claude response".into())
    }
}
