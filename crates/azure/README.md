# Azure OpenAI 测试接口

这是一个用于测试 Azure OpenAI API 的 Rust 库和示例程序。

## 配置

在使用之前，需要在项目根目录的 `.env` 文件中配置以下变量：

```env
# Azure OpenAI 配置
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### 获取配置信息

1. **API 密钥**: 在 Azure Portal 中找到你的 Azure OpenAI 资源，在"密钥和端点"页面获取 API 密钥
2. **端点**: 同样在"密钥和端点"页面获取端点 URL
3. **部署名称**: 在 Azure OpenAI Studio 中创建部署时指定的名称
4. **API 版本**: 通常使用最新的 API 版本，如 `2024-02-15-preview`

## 使用方法

### 运行测试程序

```bash
cd crates/azure
cargo run
```

### 作为库使用

```rust
use azure::AzureOpenAI;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let azure = AzureOpenAI::new()?;

    // 测试连接
    azure.test_connection().await?;

    // 发送消息
    let response = azure.simple_chat("你好，请介绍一下你自己", None, Some(200)).await?;
    println!("AI 回复: {}", response);

    Ok(())
}
```

### 高级用法

```rust
use azure::{AzureOpenAI, ChatCompletionRequest, ChatMessage};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let azure = AzureOpenAI::new()?;

    let request = ChatCompletionRequest {
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "你是一个专业的 Rust 编程助手。".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "请解释一下 Rust 的所有权概念。".to_string(),
            },
        ],
        model: "gpt-4".to_string(),
        max_tokens: Some(500),
        temperature: Some(0.7),
        stream: Some(false),
    };

    let response = azure.chat_completion(request).await?;

    if let Some(choice) = response.choices.first() {
        println!("回复: {}", choice.message.content);
    }

    Ok(())
}
```

## 功能特性

- ✅ 支持聊天补全 API
- ✅ 支持自定义系统消息
- ✅ 支持流式和非流式响应
- ✅ 支持温度、最大令牌数等参数配置
- ✅ 完整的错误处理
- ✅ 连接测试功能
- ✅ 详细的调试日志

## 错误处理

库提供了详细的错误信息，包括：
- 配置错误（缺少必需的环境变量）
- API 错误（HTTP 状态码、错误详情）
- 网络错误（连接超时、网络问题）
- JSON 序列化/反序列化错误

## 注意事项

1. 确保在 Azure Portal 中正确配置了 Azure OpenAI 资源
2. 部署名称区分大小写
3. API 版本需要与你的 Azure OpenAI 服务兼容
4. 注意 API 调用的配额和费用