import click
import os
import sys
import logging
import asyncio
from A2AServer.adk_agent import ADKAgent
from dotenv import load_dotenv

load_dotenv()

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# 强制配置 root logger
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    # Ascasync def main():
    """
    示例运行方法。
    """
    agent = ADKAgent(
        config_path="mcp_config.json",
        model_name="deepseek-chat",
        prompt_file="prompt.txt",
        provider="deepseek"
    )
    await agent.setup_agent()
    query = "你好啊"
    session_id = "test_session"

    async def run_stream():
        async for response in agent.stream(query, session_id):
            print(response)

    await run_stream()
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
