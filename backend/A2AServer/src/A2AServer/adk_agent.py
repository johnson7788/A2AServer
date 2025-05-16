#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2025/5/16 08:37
# @File  : adk_agent.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : Google ADK实现的Agent，代替自定义实现的agent.py

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
from google.adk.models.lite_llm import LiteLlm
from google.genai import types  # Import types for Content
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

class ADKAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, config_path="mcp_config.json", model_name="gemini-2.0-flash", prompt_file="prompt.txt",
                 provider="google", quiet_mode=False, log_messages_path=None):
        load_dotenv()
        self.config_path = config_path
        self.model_name = model_name
        self.prompt_file = prompt_file
        self.provider = provider
        self.quiet_mode = quiet_mode
        self.log_messages_path = log_messages_path

        with open(config_path, "r", encoding="utf-8") as f:
            self.file = json.load(f)
        self.servers_cfg = self.file.get("mcpServers", {})
        assert os.path.exists(prompt_file), f"Prompt file must exist: {prompt_file}"

        self.agent = None
        self.runner = None
        self.exit_stack = AsyncExitStack()
        self.tools = []
        self.tool_ready = False

    async def setup_agent(self):
        system_prompt = "You are a helpful assistant."
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        except Exception as e:
            logger.warning(f"Failed to read prompt file: {e}")
        logger.info(f"使用的系统提示词是: {system_prompt}")
        model = self._configure_model()
        self.tools = await self._setup_mcp_tools()

        self.agent = LlmAgent(
            model=model,
            name="adk_agent",
            instruction=system_prompt,
            tools=self.tools,
        )

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
        if self.provider == "google":
            return self.model_name
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
        tools = []
        for server_name, conf in self.servers_cfg.items():
            try:
                if "url" in conf:
                    connection_params = SseServerParams(url=conf["url"])
                elif "command" in conf:
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
        if not self.tool_ready:
            await self.setup_agent()

        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "updates": "Processing request..."
        }

        try:
            # Create a session if it doesn't exist
            user_id = session_id  # Assuming user_id is same as session_id for simplicity
            session = self.runner.session_service.get_session(
                app_name="adk_agent", user_id=user_id, session_id=session_id
            )
            if not session:
                # Initialize a new session (simplified, adjust based on your needs)
                self.runner.session_service.create_session(
                    app_name="adk_agent", user_id=user_id, session_id=session_id
                )

            # Convert query to types.Content
            new_message = types.Content(parts=[types.Part(text=query)])

            # Use run_async instead of stream
            async for event in self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=new_message,
            ):
                logger.info(f"返回的event: {event}")
                content = event.content.parts[0].text if event.content and event.content.parts else ""
                yield {
                    "is_task_complete": event.is_task_complete if hasattr(event, 'is_task_complete') else False,
                    "require_user_input": event.require_user_input if hasattr(event, 'require_user_input') else False,
                    "content": content,
                    "type": event.type if hasattr(event, 'type') else "normal",
                }
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": " ",
            }
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "updates": f"Error processing request: {str(e)}"
            }

    async def cleanup(self):
        if not self.quiet_mode:
            print("Cleaning up...")
        await self.exit_stack.aclose()
        if self.log_messages_path and self.runner:
            with open(self.log_messages_path, "w", encoding="utf-8") as f:
                json.dump(self.runner.session_service.get_session("adk_agent"), f, ensure_ascii=False)
        if not self.quiet_mode:
            print("Cleanup complete.")

async def main():
    agent = ADKAgent(
        config_path="mcp_config.json",
        model_name="gemini-2.0-flash",
        prompt_file="prompt.txt",
        provider="google"
    )
    await agent.setup_agent()  # Ensure agent is set up
    query = "你好啊"
    session_id = "test_session"

    async def run_stream():
        async for response in agent.stream(query, session_id):
            print(response)

    await run_stream()
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())