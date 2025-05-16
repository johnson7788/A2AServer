# 深度搜索示例

本仓库提供了深度搜索系统的示例实现，包括 A2A 服务器和 MCP 服务器的配置。请按照以下步骤设置和运行项目。

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

### 4.2 启动 MCP 服务器（SSE 模式）
使用 SSE 传输方式运行 MCP 服务器：
```bash
cd mcpserver
fastmcp run --transport sse --port 7001 search_tool.py
```

### 4.3 配置 `mcp_config.json`
更新 `mcp_config.json` 以包含您的 MCP 服务器。为获得最佳深度搜索效果，建议与 `server-sequential-thinking` 搭配使用：
```json
{
  "mcpServers": {
    "SearchTool": {
      "url": "http://127.0.0.1:7001/sse"
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
python main.py
```

---

## 🧪 测试
使用提供的客户端脚本测试 A2A 服务器：
```bash
python client.py --agent http://localhost:10004
```

---

## 🔧 替代 MCP 服务器设置（Stdio 模式）
为了获得更好的深度搜索性能，可以使用 Stdio 模式启动 MCP 服务器，而非 SSE 模式。更新 `mcp_config.json` 如下：
```json
{
  "mcpServers": {
    "SearchTool": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "fastmcp",
        "fastmcp",
        "run",
        "mcpserver/search_tool.py"
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

### 验证 MCP 依赖
在启动之前，检查所需依赖是否正确安装：
```bash
uv run --with fastmcp fastmcp run mcpserver/search_tool.py
```
或者使用
```bash
bash test_mcp_server.sh
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