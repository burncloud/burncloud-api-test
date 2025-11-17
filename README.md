# OpenAI API测试工具

使用Rust编写的OpenAI格式API测试工具，支持RPM限制测试。

## 配置

编辑`.env`文件：
```
API_TOKEN=你的API密钥
API_URL=https://api.openai.com/v1/chat/completions
RPM=60
```

## 运行

```bash
cargo run
```

## 功能

- 发送测试请求到OpenAI格式的API
- 根据RPM设置自动控制请求间隔
- 中文回复测试