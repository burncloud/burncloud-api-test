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
    let max_tokens: u32 = std::env::var("MAX_TOKENS")
        .expect("MAX_TOKENS not found in .env")
        .parse()
        .expect("MAX_TOKENS must be a number");
    let stream: bool = std::env::var("STREAM")
        .unwrap_or_else(|_| "true".to_string())
        .parse()
        .expect("STREAM must be true or false");
    let prompt = std::env::var("PROMPT")
        .expect("PROMPT not found in .env");

    println!("API高并发测试开始 ({})", if stream { "含TTFT+TPOT测量" } else { "响应时间测量" });
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

    let request = ChatRequest {
        model: model,
        messages: vec![
            Message {
                role: "system".to_string(),
                content: prompt,
            },
            Message {
                role: "user".to_string(),
                content: "hi".to_string(),
            }
        ],
        max_tokens: max_tokens,
        temperature: 0.7,
        stream: stream,
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
        let request = request.clone();
        let success_count = success_count.clone();
        let error_count = error_count.clone();
        let ttft_metrics = ttft_metrics.clone();
        let tpot_metrics = tpot_metrics.clone();

        let task = tokio::spawn(async move {
            match send_request_with_retry(&client, &url, &token, &request, 2).await {
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
                        if request.stream {
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
                        if request.stream {
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
    request: &ChatRequest,
    max_retries: u32,
) -> Result<(Duration, Option<TPOTMetrics>, String), Box<dyn std::error::Error + Send + Sync>> {
    let _start_time = Instant::now();
    let mut last_error: Option<Box<dyn std::error::Error + Send + Sync>> = None;

    for attempt in 0..=max_retries {
        let request_start = Instant::now();

        let result = if request.stream {
            send_request_stream_once(client, url, token, request).await
        } else {
            send_request_non_stream_once(client, url, token, request).await
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
    request: &ChatRequest,
) -> Result<(Duration, Option<TPOTMetrics>, String), Box<dyn std::error::Error + Send + Sync>> {
    let start_time = Instant::now();
    let response = client
        .post(url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Content-Type", "application/json")
        .header("Connection", "keep-alive")
        .header("Accept", "text/event-stream")
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
    request: &ChatRequest,
) -> Result<(Duration, Option<TPOTMetrics>, String), Box<dyn std::error::Error + Send + Sync>> {
    let start_time = Instant::now();
    let response = client
        .post(url)
        .header("Authorization", format!("Bearer {}", token))
        .header("Content-Type", "application/json")
        .header("Connection", "keep-alive")
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
