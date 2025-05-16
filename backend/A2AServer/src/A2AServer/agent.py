import logging
import os
import traceback
import json
import base64
from dotenv import load_dotenv
from typing import AsyncIterable, Any, Literal
from pydantic import BaseModel
from datetime import datetime

from A2AServer.mcp_client.client import *

logger = logging.getLogger(__name__)


def base64_to_dict(base64_str: str) -> dict:
    """将 Base64 字符串还原为 Python 字典"""
    try:
        json_bytes = base64.b64decode(base64_str)
        json_str = json_bytes.decode('utf-8')
        data = json.loads(json_str)
    except Exception as e:
        print(f"Error decoding Base64: {e}")
        return {}
    return data


class BasicAgent:
    """Agent to access Deep Search"""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self, config_path="mcp_config.json", model_name="deepseek-chat", prompt_file="prompt.txt",
                 provider="deepseek",
                 quiet_mode=False, log_messages_path=None):
        """
        Synchronous initialization.
        Loads config and sets up basic attributes.
        Asynchronous setup (starting servers, listing tools) is done in the 'setup' method.
        """
        self.config_path = config_path
        self.model_name = model_name
        self.quiet_mode = quiet_mode
        self.log_messages_path = log_messages_path

        self.file = load_mcp_config_from_file(config_path)
        config = self.file
        self.servers_cfg = config.get("mcpServers", {})
        self.models_cfg = config.get("models", [])
        assert os.path.exists(prompt_file), f"Agent prompt file 必须存在，请检查: {prompt_file}"
        # Choose a model (synchronous)
        self.chosen_model = {"model": model_name, "provider": provider, "prompt_file": prompt_file}
        self.is_ready = True

        # Initialize attributes that will be populated asynchronously in setup()
        self.servers = {}
        self.all_functions = []
        self.conversation = []  # Initial conversation might be built later in run() or here
        self.tool_ready = False
        # 为啥在这里初始化工具后，工具就无法被调通呢，
        # loop = asyncio.get_event_loop()
        # try:
        #     self.tool_ready = loop.run_until_complete(self.setup_tools())
        # except RuntimeError:
        #     self.tool_ready = asyncio.run(self.setup_tools())

    async def setup_tools(self):
        """
        Asynchronous setup method.
        Starts servers and gathers tools.
        Returns True if setup was successful, False otherwise.
        """
        if not self.is_ready:
            print("Agent cannot be set up: Model not found.")
            return False

        print("Starting MCP servers...")
        successful_servers = {}
        all_functions = []
        # 初始化MCP的server
        for server_name, conf in self.servers_cfg.items():
            client = None
            if "url" in conf:  # SSE server
                client = SSEMCPClient(server_name, conf["url"])
            elif "command" in conf:  # Local process-based server
                client = MCPClient(
                    server_name=server_name,
                    command=conf.get("command"),
                    args=conf.get("args", []),
                    env=conf.get("env", {})
                )
            else:
                if not self.quiet_mode:
                    print(f"[WARN] Skipping server {server_name}: No 'url' or 'command' specified.")
                continue

            try:
                ok = await client.start()  # <-- AWAIT is valid here (inside async def)
                if not ok:
                    if not self.quiet_mode:
                        print(f"[WARN] Could not start server {server_name}")
                    # Ensure client is stopped even if start failed
                    if client: await client.stop()
                    continue
                else:
                    print(f"[MCP Tool OK] {server_name}")
                    successful_servers[server_name] = client

                    # gather tools
                    try:
                        tools = await client.list_tools()  # <-- AWAIT is valid here
                        for t in tools:
                            input_schema = t.get("inputSchema") or {"type": "object", "properties": {}}
                            fn_def = {
                                "name": f"{server_name}_{t['name']}",
                                "description": t.get("description", ""),
                                "parameters": input_schema
                            }
                            all_functions.append(fn_def)
                    except Exception as e:
                        if not self.quiet_mode:
                            print(f"[WARN] Error listing tools for {server_name}: {e}")
                        # Consider if failing to list tools should stop processing for this server


            except Exception as e:  # Catch potential errors during client creation or start
                if not self.quiet_mode:
                    print(f"[WARN] Exception starting server {server_name}: {e}")
                # Ensure client is stopped if created before exception
                if 'client' in locals() and client: await client.stop()

        self.servers = successful_servers
        self.all_functions = all_functions

        if not self.servers:
            error_msg = "No MCP servers could be started."
            print(f"[ERROR] {error_msg}")
            self.tool_ready = False  # Cannot run without servers
            return False

        print(f"Found {len(self.all_functions)} tools.")
        self.tool_ready = True  # Setup was successful
        return True

    async def run_inference(self, user_query, history, sessionId, stream=True):
        """
        推理和工具的设置
        """
        if not self.tool_ready:
            #  如果没设置过相关的MCP工具
            await self.setup_tools()
        if not self.is_ready:
            print("Agent is not ready. Setup failed or model not found.")
            # Depending on requirements, you might return an error or raise an exception
            if stream:
                async def error_gen(): yield "Agent setup failed."

                return error_gen()
            return "Agent setup failed."

        # Build initial conversation (system message + user query)
        self._build_initial_conversation(user_query, history)  # This helper can be synchronous

        # try:
        if stream:
            return self._stream_response_generator(sessionId)  # Returns an async generator
        else:
            return await self._non_stream_response()  # Returns the final text
        # finally:
        #     # Ensure cleanup is called when run() finishes or an exception occurs
        #     await self.cleanup() # <-- AWAIT is valid here

    def _build_initial_conversation(self, user_query, history=[]):
        """
        user_query: str, 用户的问题
        history: 历史记录
        """
        # Helper method to build the initial conversation list (synchronous)
        self.conversation = []
        # 默认的prompt
        agent_prompt = "You are a helpful assistant."
        try:
            with open(self.chosen_model["prompt_file"], "r", encoding="utf-8") as f:
                agent_prompt = f.read()
            self.conversation.append({"role": "system", "content": agent_prompt})
        except Exception as e:
            logger.warning(f"Failed to read Agent prompt file: {e}")
            self.conversation.append({"role": "system", "content": agent_prompt})
        # 加上当前的时间
        self.conversation[0]['content'] = f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}。" + \
                                          self.conversation[0]['content']
        # 这里加上历史的记录
        self.conversation.extend(history)
        self.conversation.append({"role": "user", "content": user_query})
        print(f"发起的conversation: {self.conversation}")

    async def _stream_response_generator(self, sessionId):
        """Handles the streaming response logic (async generator)."""
        # 分5种返回类型，1. reasoning, 2. normal,  4. tool_call, 5. tool_result
        while True:
            generator = await generate_text(self.conversation, self.chosen_model, self.all_functions, stream=True)
            accumulated_text = ""
            tool_calls_processed = False

            async for chunk in generator:  # AWAIT is used to iterate over the async generator
                if chunk.get("is_chunk", False):
                    if chunk.get("token", False):
                        if chunk.get("is_reasoning"):
                            yield {"text": chunk["assistant_text"], "type": "reasoning"}
                        else:
                            yield {"text": chunk["assistant_text"], "type": "normal"}  # YIELD is used in a generator
                    if not chunk.get("is_reasoning"):
                        accumulated_text += chunk["assistant_text"]
                else:
                    remaining = chunk["assistant_text"][len(accumulated_text):]
                    if remaining:
                        yield {"text": remaining, "type": "normal"}  # YIELD here as well 剩余文本

                    tool_calls = chunk.get("tool_calls", [])
                    if tool_calls:
                        for tc in tool_calls:
                            tc["type"] = "function"
                        assistant_message = {
                            "role": "assistant",
                            "content": chunk["assistant_text"],
                            "tool_calls": tool_calls
                        }
                        self.conversation.append(assistant_message)
                        yield {"text": f"{json.dumps(tool_calls, ensure_ascii=False)}", "type": "tool_call"}

                        for tc in tool_calls:
                            if tc.get("function", {}).get("name"):
                                # 对工具进行参数的修改
                                result = await process_tool_call(tc, self.servers, self.quiet_mode)  # AWAIT valid here
                                if result:
                                    self.conversation.append(result)
                                    tool_calls_processed = True
                                    yield {"text": f"{json.dumps(result)}", "type": "tool_result"}
            if not tool_calls_processed:
                break


    async def _non_stream_response(self):
        """Handles the non-streaming response logic."""
        # Move the non-stream logic from original init here
        final_text = ""
        while True:
            gen_result = await generate_text(self.conversation, self.chosen_model, self.all_functions,
                                             stream=False)  # AWAIT valid here

            assistant_text = gen_result["assistant_text"]
            final_text = assistant_text
            tool_calls = gen_result.get("tool_calls", [])

            assistant_message = {"role": "assistant", "content": assistant_text}
            if tool_calls:
                for tc in tool_calls:
                    tc["type"] = "function"
                assistant_message["tool_calls"] = tool_calls
            self.conversation.append(assistant_message)
            logger.info(f"Added assistant message: {json.dumps(assistant_message, indent=2)}")

            if not tool_calls:
                break

            for tc in tool_calls:
                result = await process_tool_call(tc, self.servers, self.quiet_mode)  # AWAIT valid here
                if result:
                    self.conversation.append(result)
                    logger.info(f"Added tool result: {json.dumps(result, indent=2)}")

        return final_text


    async def cleanup(self):
        """Clean up servers and log messages."""
        print("Cleaning up servers...")
        if self.log_messages_path:
            # Pass attributes needed for logging
            await log_messages_to_file(self.conversation, self.all_functions, self.log_messages_path)  # AWAIT valid here
        for cli in self.servers.values():
            await cli.stop()  # AWAIT valid here
        print("Cleanup complete.")


    def get_agent_response(self, response: str) -> dict[str, Any]:
        """Format agent response in a consistent structure."""
        try:
            # All final responses should be treated as complete
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": response
            }
        except Exception as e:
            # Log but continue with best-effort fallback
            logger.error(f"Error parsing response: {e}, response: {response}")

            # Default to treating it as a completed response
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": response
            }


    async def stream(self, query: str, history: list[dict], sessionId: str) -> AsyncIterable[dict[str, Any]]:
        """Stream updates from the MCP agent.
        # sessionId作为知识库的关联信息, 重新初始化知识库的mcpClient
        """
        print(f"问题: {query}的sessionId为： {sessionId}")
        try:
            # Initial response to acknowledge the query
            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "updates": "Processing request..."
            }

            logger.info(f"Processing query: {query[:50]}...")

            try:

                response_generator = await self.run_inference(
                    user_query=query,
                    history=history,
                    sessionId=sessionId,
                    stream=True,
                )
                # Iterate through the chunks yielded by the response_generator
                chunks = []
                async for chunk in response_generator:
                    # Yield each chunk as it arrives.
                    yield {
                        "is_task_complete": False,  # Indicate it's an intermediate part
                        "require_user_input": False,
                        "content": chunk["text"],  # Yield the actual content chunk
                        "type": chunk["type"],
                    }
                yield {
                    "is_task_complete": True,  # Indicate it's an intermediate part
                    "require_user_input": False,
                    "content": " "
                }
            except Exception as e:
                logger.error(f"Error during processing: {traceback.format_exc()}")
                yield {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "updates": f"Error processing request: {str(e)}"
                }
        except Exception as e:
            logger.error(f"Error in streaming agent: {traceback.format_exc()}")
            yield {
                "is_task_complete": False,
                "require_user_input": True,
                "updates": f"Error processing request: {str(e)}"
            }


    def invoke(self, query: str, sessionId: str) -> dict[str, Any]:
        """Synchronous invocation of the MCP agent."""
        raise NotImplementedError(
            "Synchronous invocation is not supported by this agent. Use the streaming endpoint (tasks/sendSubscribe) instead."
        )
