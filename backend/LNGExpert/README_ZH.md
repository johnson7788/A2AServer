# 比较复杂的问答示例(灵活的工作流)

本仓库提供了多个MCP的工具的示例， 让Agent按一定工作流完成任务。工作流的示例在Prompt中给出，未给出的将使用思考方式灵活解答。

---

## 🚀 开始使用

### 1. 安装依赖
进入 backend 目录并安装所需依赖包：
```bash
cd backend/A2AServer
pip install -e .
```

### 2. 配置环境变量
复制环境变量模板并根据需要进行修改：
```bash
cp env_template.txt .env
```

### 3. 更新 Prompt 文件
编辑 `prompt.txt` 文件以定义代理的行为。

---

## ⚙️ MCP 服务器配置

### 4.1 设置自定义 MCP 服务器
1. 创建 `mcpserver` 目录。
2. 在 `mcpserver` 目录中添加 MCP 服务器文件（例如 `search_tool.py`）。
3. 在 `mcp_config.json` 中配置多个 MCP 文件。

### 4.2 配置 `mcp_config.json`
更新 `mcp_config.json` 以包含您的 MCP 多个服务器。使用 `server-sequential-thinking` 进行序列化思考：
```json
{
  "mcpServers": {
    "FactoryProfit": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "fastmcp",
        "fastmcp",
        "run",
        "mcpserver/Simulate_Factory_Profit.py"
      ]
    },
    "LNGPrice": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "fastmcp",
        "fastmcp",
        "run",
        "mcpserver/Simulate_LNG_Price.py"
      ]
    },
    "sequentialThinking": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-sequential-thinking"
      ]
    }
  }
}

```

---

## 🌐 启动 A2A 服务器
启动 A2A 服务器以处理代理任务：
```bash
python main.py --port 10003
```

---

### 验证 MCP 依赖
在启动之前，检查所需依赖是否正确安装：
```bash
uv run --with fastmcp fastmcp run mcpserver/Simulate_Factory_Profit.py
uv run --with fastmcp fastmcp run mcpserver/Simulate_LNG_Price.py.py
```
或者使用
```bash
bash test_mcp_server.sh
```

## 🧪 UI测试
```bash
cd frontend/single_agent
npm run dev
```

---
## 📖 命令行帮助
查看 A2A 服务器的可用选项：
```bash
python main.py --help
```

**输出：**
```
用法: main.py [OPTIONS]

  启动 A2A 服务器，用于加载智能代理并响应任务请求。

选项:
  --host TEXT        服务器绑定的主机名（默认: localhost）
  --port INTEGER     服务器监听的端口号（默认: 10004）
  --prompt TEXT      代理的 prompt 文件路径（默认: prompt.txt）
  --model TEXT       使用的模型名称（例如 deepseek-chat）
  --provider TEXT    模型提供方名称（例如 deepseek、openai 等）
  --mcp_config TEXT  MCP 配置文件路径（默认: mcp_config.json）
  --help             显示此帮助信息并退出。
```

---

## 💡 使用提示
- 与 `server-sequential-thinking` 搭配使用可增强深度搜索能力。
- 确保配置文件中的所有路径正确，以避免运行时错误。
- 在与 A2A 服务器集成之前，建议独立测试 MCP 服务器。