#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/5/16 08:37
# @File  : adk_agent.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : Google ADK实现的Agent

import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import AsyncIterable, Any, Dict

import google.auth
from google.adk.agents import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters, SseServerParams
from google.adk.models.lite_llm import LiteLlm  #谷歌进行了封装的Litellm
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class ADKAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, config_path="mcp_config.json", model_name="gemini-2.0-flash", prompt_file="prompt.txt",
                 provider="google", quiet_mode=False, log_messages_path=None):
        """
        初始化ADKAgent，加载配置并设置模型和MCP工具。
        """
        load_dotenv()
        self.config_path = config_path
        self.model_name = model_name
        self.prompt_file = prompt_file
        self.provider = provider
        self.quiet_mode = quiet_mode
        self.log_messages_path = log_messages_path

        # 加载MCP配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.file = json.load(f)
        self.servers_cfg = self.file.get("mcpServers", {})
        assert os.path.exists(prompt_file), f"Prompt file must exist: {prompt_file}"

        # 初始化ADK组件
        self.agent = None
        self.runner = None
        self.exit_stack = AsyncExitStack()
        self.tools = []
        self.tool_ready = False


    async def setup_agent(self):
        """
        异步设置模型和MCP工具。
        """
        # 加载系统提示
        system_prompt = "You are a helpful assistant."
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        except Exception as e:
            logger.warning(f"Failed to read prompt file: {e}")

        # 设置模型
        model = self._configure_model()

        # 初始化MCP工具
        self.tools = await self._setup_mcp_tools()

        # 创建LlmAgent
        self.agent = LlmAgent(
            model=model,
            name="adk_agent",
            instruction=system_prompt,
            tools=self.tools,
        )

        # 初始化Runner
        self.runner = Runner(
            app_name="adk_agent",
            agent=self.agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

        self.tool_ready = True
        if not self.quiet_mode:
            print(f"Initialized ADKAgent with model {self.model_name} and {len(self.tools)} tools.")

    def _configure_model(self):
        """
        根据provider配置模型。
        """
        if self.provider == "google":
            return self.model_name  # 直接使用Gemini模型ID
        elif self.provider == "openai":
            return LiteLlm(model=f"openai/{self.model_name}")
        elif self.provider == "anthropic":
            return LiteLlm(model=f"anthropic/{self.model_name}")
        elif self.provider == "ollama":
            return LiteLlm(model=f"ollama_chat/{self.model_name}")
        elif self.provider == "deepseek":
            return LiteLlm(model=f"deepseek/{self.model_name}")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _setup_mcp_tools(self):
        """
        初始化MCP工具，连接到MCP服务器。
        """
        tools = []
        for server_name, conf in self.servers_cfg.items():
            try:
                if "url" in conf:
                    # 远程MCP服务器（SSE）
                    connection_params = SseServerParams(url=conf["url"])
                elif "command" in conf:
                    # 本地MCP服务器
                    connection_params = StdioServerParameters(
                        command=conf.get("command"),
                        args=conf.get("args", []),
                        env=conf.get("env", {})
                    )
                else:
                    if not self.quiet_mode:
                        print(f"[WARN] Skipping server {server_name}: No 'url' or 'command' specified.")
                    continue
                logger.info(f"连接MCP Server:{server_name}")
                server_tools, _ = await MCPToolset.from_server(
                    connection_params=connection_params,
                    async_exit_stack=self.exit_stack
                )
                tools.extend(server_tools)
                if not self.quiet_mode:
                    print(f"[MCP Tool OK] {server_name} with {len(server_tools)} tools.")
            except Exception as e:
                if not self.quiet_mode:
                    print(f"[WARN] Exception setting up server {server_name}: {e}")

        return tools

    async def stream(self, query: str, session_id: str) -> AsyncIterable[Dict[str, Any]]:
        """
        流式处理用户查询，返回响应。
        """
        if not self.tool_ready:
            await self._setup()

        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "updates": "Processing request..."
        }

        try:
            # 使用Runner处理查询
            async for update in self.runner.stream(query, user_id=session_id):
                content = update.get("content", "")
                response_type = update.get("type", "normal")
                yield {
                    "is_task_complete": update.get("is_task_complete", False),
                    "require_user_input": update.get("require_user_input", False),
                    "content": content,
                    "type": response_type,
                }
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "updates": f"Error processing request: {str(e)}"
            }

    async def cleanup(self):
        """
        清理资源，关闭MCP服务器连接。
        """
        if not self.quiet_mode:
            print("Cleaning up...")
        await self.exit_stack.aclose()
        if self.log_messages_path and self.runner:
            # 保存会话日志（可选）
            with open(self.log_messages_path, "w", encoding="utf-8") as f:
                json.dump(self.runner.session_service.get_session(self.runner.user_id), f, ensure_ascii=False)
        if not self.quiet_mode:
            print("Cleanup complete.")
async def main():
    """
    示例运行方法。
    """
    agent = ADKAgent(
        config_path="mcp_config.json",
        model_name="gemini-2.0-flash",
        prompt_file="prompt.txt",
        provider="google"
    )
    query = "你好啊"
    session_id = "test_session"

    async def run_stream():
        async for response in agent.stream(query, session_id):
            print(response)

    await run_stream()
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
