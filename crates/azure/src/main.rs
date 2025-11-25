use azure::AzureOpenAI;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 启动 Azure OpenAI 测试程序");

    // 创建 Azure OpenAI 客户端
    let azure = match AzureOpenAI::new() {
        Ok(client) => {
            println!("✅ Azure OpenAI 客户端创建成功");
            client
        }
        Err(e) => {
            println!("❌ 创建 Azure OpenAI 客户端失败: {}", e);
            println!();
            println!("请确保 .env 文件包含以下配置:");
            println!("AZURE_OPENAI_API_KEY=your_api_key_here");
            println!("AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/");
            println!("AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name");
            println!("AZURE_OPENAI_API_VERSION=2024-02-15-preview");
            return Err(e);
        }
    };

    println!();

    // 测试连接
    println!("🔍 测试 Azure OpenAI 连接...");
    match azure.test_connection().await {
        Ok(_) => println!("✅ 连接测试通过"),
        Err(e) => {
            println!("❌ 连接测试失败: {}", e);
            return Err(e);
        }
    }

    println!();

    // 进行简单的对话测试
    println!("💬 开始对话测试...");
    let test_questions = vec![
        "请用一句话介绍你自己",
        "什么是人工智能？",
        "写一个简单的 Rust 函数来计算斐波那契数列",
    ];

    for (i, question) in test_questions.iter().enumerate() {
        println!("问题 {}: {}", i + 1, question);
        match azure.simple_chat(question, None, Some(200)).await {
            Ok(response) => {
                println!("回答: {}", response);
            }
            Err(e) => {
                println!("❌ 获取回答失败: {}", e);
            }
        }
        println!();
    }

    println!("🎉 Azure OpenAI 测试完成!");

    Ok(())
}