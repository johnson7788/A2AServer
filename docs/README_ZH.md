# 文档

本仓库包含关于用户界面 (UI) 和不同大型语言模型 (LLM) 配置的文档。

## UI 界面

- [中文用户界面指南](./Chinese/UI.md)

这两个文档详细介绍了应用程序的用户界面，包括单 Agent 和多 Agent 模式的使用示例，以及对话和思考过程的界面展示。文档中使用了图片进行说明，图片位于 `images` 目录下。

## 不同的 LLM 配置

- [中文 LLM 配置指南](./Chinese/LLM_configuration.md)

这两个文档提供了关于不同大型语言模型配置的详细信息，包括配置参数、使用方法以及相关的注意事项。

## Debug技巧
记录MCP的工具的输入和输出,修改backend/DeepSearch/mcp_config.json的启动从uv启动到debug_utils.py启动，
```
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
    }
```
```
    "SearchTool": {
      "command": "mcpserver/debug_utils.py",
      "args": [
        "fastmcp",
        "run",
        "mcpserver/search_tool.py"
      ]
    },
```
backend/DeepSearch/mcpserver/mcp.log中显示了详细的使用日志。

## 问题总结
1. Deepseek官方Deepseek-R1模型不支持函数调用，因此使用Deepseek的R1模型会报错。但是火山引擎的deepseek-r1模型可以支持函数调用和思考，推荐。
2. 保持传入Agent的会话的session_id的唯一，这是必须的。
3. Agent Card中的url是最终对外提供的接口，当前端访问时，会访问这个Agent Card中的地址，可以按需传入。
4. DeepSeek V3的函数名称不支持中文, ByteDance的Deepseek R1函数支持中文

# Bug 修复
1. backend/A2AServer/src/A2AServer/mcp_client/client.py, 修复MCP客端建立session后的self.responses直不删除，导致mcpserver的数据越来越多卡死的问题，添加了时间戳控制收到的数据
请查阅相应的文档以获取更详细的信息。