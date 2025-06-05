# 🚀 A2A 服务器开发指南

本项目为您提供开发和测试 A2A 服务器的完整指导，涵盖项目结构、开发步骤和测试方法，帮助您快速构建自定义的 A2A 智能代理。

---

## 📂 项目结构

```
backend
├── A2AServer           # A2A 服务器依赖包
├── DeepSearch          # 示例项目：实现深度搜索功能
├── AgentRAG            # 示例项目：实现Agent RAG功能
├── LNGExpert           # 示例项目：实现Agent的灵活工作流
├── A2Aclient.py        # 被其它程序调用时使用这个，可以集成到其它应用
└── client.py           # 用于测试 A2A 服务器的客户端脚本
```

---

## 🛠️ 开发自己的 A2A 服务器

通过参考 `DeepSearch` 示例，您可以快速创建自己的 A2A 服务器。以下是具体步骤：

### 1. 复制 DeepSearch 目录
将 `DeepSearch` 目录复制为您的项目起点：

```
DeepSearch
├── .env                # 存储模型的 API 密钥
├── main.py             # A2A 服务器的启动文件
├── mcp_config.json     # MCP 服务器的配置文件
├── mcpserver/          # 可选：MCP 服务器代码
│   └── search_tool.py  # 示例 MCP 工具（可包含多个工具）
└── prompt.txt          # Agent 的 Prompt 配置文件
```

### 2. 自定义服务器
- **配置 `.env`**：添加您的模型 API 密钥。
- **修改 `main.py`**：实现您的 A2A 服务器逻辑。
- **更新 `mcp_config.json`**：配置 MCP 工具和相关设置。
- **扩展 `mcpserver/`**：根据需要添加更多工具（如 `search_tool.py`）。
- **调整 `prompt.txt`**：根据您的场景自定义 Agent 的 Prompt。

---

## 🧪 测试您的 A2A 服务器

您可以通过命令行或前端界面测试您的 A2A 服务器。

### 方式 1：命令行测试
使用 `client.py` 脚本测试您的 Agent：

```bash
python client.py --agent http://localhost:10004
```

### 方式 2：前端界面测试
使用单 Agent 前端界面进行测试：
1. 进入前端目录并启动：
   ```bash
   cd frontend/single_agent
   npm install
   npm run dev
   ```
2. 在浏览器中打开前端页面，输入 Agent 的 URL（例如 `http://localhost:10004`），开始测试。

---

## ⚠️ 注意事项

- **工具命名规则**：
  - 在 `mcp_config.json` 中，工具名称应使用驼峰式或帕斯卡式命名（如 `SearchTool`、`RAGTool`）。
  - 避免使用下划线（如 `Search_Tool`、`RAG_Tool`），否则可能导致工具无法被识别。

---

## 💡 快速上手与后续步骤

### 快速上手
1. 运行 `DeepSearch` 示例，熟悉 A2A 服务器的运行流程。
2. 根据您的需求，修改 `prompt.txt` 和 `mcp_config.json`。
3. 使用 `client.py` 或前端界面验证您的服务器功能。

### 后续步骤
- 深入研究 `DeepSearch` 示例，了解 A2A 服务器的实现细节。
- 在 `mcpserver/` 目录中添加更多自定义工具，扩展功能。
- 集成多 Agent 前端（`frontend/multiagent_front`），体验多 Agent 协作。