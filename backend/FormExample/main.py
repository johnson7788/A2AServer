import click
import os
import sys
import logging
from A2AServer.common.server import A2AServer
from A2AServer.common.A2Atypes import AgentCard, AgentCapabilities, AgentSkill, MissingAPIKeyError
from A2AServer.task_manager import AgentTaskManager
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

@click.command(help="启动 A2A Server，用于加载 ADKAgent 并响应任务请求")
@click.option("--host", "host", default="localhost", help="服务器绑定的主机名（默认为 localhost,可以指定具体本机ip）")
@click.option("--port", "port", default=10005, help="服务器监听的端口号（默认为 10005）")
@click.option("--prompt", "agent_prompt_file", default="prompt.txt", help="Agent 的 prompt 文件路径（默认为 prompt.txt）")
@click.option("--model", "model_name", default="deepseek-chat", help="使用的模型名称（如 deepseek-chat）")
@click.option("--provider", "provider", default="deepseek", help="模型提供方名称（如 deepseek、openai 等）")
@click.option("--mcp_config", "mcp_config_path", default="mcp_config.json", help="MCP 配置文件路径（默认为 mcp_config.json）")
@click.option("--agent_url", "agent_url", default="", help="Agent Card中对外展示和访问的地址")
def main(host, port, agent_prompt_file, model_name, provider, mcp_config_path, agent_url=""):
    """启动 A2A Server
    host: 启动的Agent的主机
    port: 启动的端口
    agent_prompt_file: prompt文件
    """
    input_mode, output_mode = ["text", "text/plain"], ["text", "text/plain"]
    try:
        # 定义 Agent 能力和技能，支持流式响应
        capabilities = AgentCapabilities(streaming=True)
        skill = AgentSkill(
            id="receiptAssistant",
            name="receiptAssistant",
            description="发票报销助手",
            tags=["receipt", "invoice"],
            examples=[
                "我想报销1个200元的餐饮发票。"
            ]
        )
        if not agent_url:
            agent_url = f"http://{host}:{port}/"
        # 配置 AgentCard，包含名称、描述、URL、输入输出格式等
        agent_card = AgentCard(
            name="receiptAssistant",
            description="发票报销助手",
            url=agent_url,
            version="1.0.0",
            defaultInputModes=input_mode,
            defaultOutputModes=output_mode,
            capabilities=capabilities,
            skills=[skill],
        )
        print(f"AgentCard信息: {agent_card}")
        # 使用 ADKAgent 替代 BasicAgent
        agent = ADKAgent(
            config_path=mcp_config_path,
            model_name=model_name,
            prompt_file=agent_prompt_file,
            provider=provider
        )
        # 启动 A2A 服务器
        server = A2AServer(
            agent_card=agent_card,
            task_manager=AgentTaskManager(agent=agent),
            host=host,
            port=port,
        )

        logger.info(f"Starting agent on {host}:{port}")
        server.start()
    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        exit(1)

if __name__ == "__main__":
    main()