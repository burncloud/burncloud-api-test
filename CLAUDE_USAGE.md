# Azure Claude 支持使用指南

本项目现已支持在 Azure 上测试 Claude 模型，使用 v1/messages API 格式。

## 配置方法

1. 复制 `.env.example` 为 `.env`，或使用 `.env.claude.example` 作为模板
2. 修改配置参数：

```env
# 设置提供商为 claude
PROVIDER=claude

# Azure Claude 配置
AZURE_CLAUDE_API_KEY=your-claude-api-key-here
AZURE_CLAUDE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_CLAUDE_DEPLOYMENT_NAME=claude-3-5-sonnet-20241022
AZURE_CLAUDE_API_VERSION=2025-01-01-preview

# 测试参数
RPM=5
MAX_TOKENS=100
STREAM=true
PROMPT=你是一个有用的助手。请简要介绍一下自己。
```

## 配置参数说明

- `PROVIDER=claude`: 指定使用 Claude API
- `AZURE_CLAUDE_API_KEY`: Azure OpenAI API 密钥（用于 Claude 模型）
- `AZURE_CLAUDE_ENDPOINT`: Azure 资源终端点 URL
- `AZURE_CLAUDE_DEPLOYMENT_NAME`: Claude 模型部署名称
- `AZURE_CLAUDE_API_VERSION`: API 版本（默认：2025-01-01-preview）

## 支持的 Claude 模型

- `claude-3-5-sonnet-20241022`
- `claude-3-5-haiku-20241022`
- 其他 Azure 支持的 Claude 模型

## 使用示例

```bash
# 设置环境变量
set PROVIDER=claude
set AZURE_CLAUDE_API_KEY=your-api-key
set AZURE_CLAUDE_ENDPOINT=https://your-resource.openai.azure.com/
set AZURE_CLAUDE_DEPLOYMENT_NAME=claude-3-5-sonnet-20241022
set RPM=5
set MAX_TOKENS=100
set STREAM=true
set PROMPT=写一首关于春天的短诗

# 运行测试
cargo run
```

## API 格式差异

- **OpenAI/Azure OpenAI (GPT)**: 使用 `chat/completions` 端点，OpenAI 格式
- **Azure Claude**: 使用 `messages` 端点，Anthropic v1/messages 格式

## 性能指标

- **TTFT** (Time To First Token): 首个 token 的响应时间
- **TPOT** (Time Per Output Token): 每个输出 token 的平均时间
- 支持 QPS、延迟、吞吐量统计

## 注意事项

1. 确保 Azure 资源已部署对应的 Claude 模型
2. API 密钥需要有访问 Claude 模型的权限
3. 建议先用较小的 RPM 值进行测试
4. 流式和非流式模式都支持