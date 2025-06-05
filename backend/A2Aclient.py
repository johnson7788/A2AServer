import random
import time
import json
import asyncio
import urllib
from uuid import uuid4
from A2AServer.common.client import A2AClient, A2ACardResolver
from A2AServer.common.A2Atypes import TaskState,TextPart,TaskArtifactUpdateEvent,TaskStatusUpdateEvent
# from common.types import Task, TextPart, FilePart, FileContent # 如有需要可以加上
from typing import AsyncIterator


class A2AClientWrapper:
    def __init__(self,
                 session_id,
                 agent_url="http://localhost:10006",
                 push_notification_receiver="http://localhost:5000",
                 use_push_notifications=False,
                 use_history=False):
        self.agent_url = agent_url
        self.use_push_notifications = use_push_notifications
        self.use_history = use_history
        self.push_notification_receiver = push_notification_receiver
        self.session_id = session_id
        self.client = None
        self.streaming = False
        self.notification_receiver_host = None
        self.notification_receiver_port = None

    async def setup(self):
        resolver = A2ACardResolver(self.agent_url)
        card = resolver.get_agent_card()
        print("======= Agent Card ========")
        print(card.model_dump_json(exclude_none=True))
        self.client = A2AClient(agent_card=card)
        self.streaming = card.capabilities.streaming

        notif_receiver_parsed = urllib.parse.urlparse(self.push_notification_receiver)
        self.notification_receiver_host = notif_receiver_parsed.hostname
        self.notification_receiver_port = notif_receiver_parsed.port

    async def run(self, prompt: str):
        if self.client is None:
            await self.setup()

        task_id = uuid4().hex
        print("=========  starting a new task ======== ")
        stream_response = self.generate(messages=[{"role": "user", "content": prompt}])
        async for result in stream_response:
            print(f"stream event => {result}")



    async def generate(self, messages: list[dict]) -> AsyncIterator[dict]:
        """SSE 流式生成器"""
        if self.client is None:
            await self.setup()
        task_id = uuid4().hex
        prompt = messages[-1]["content"]
        print(f"发送的问题是: {prompt}")
        message = {
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
        }
        payload = {
            "id": task_id,
            "sessionId": self.session_id,
            "acceptedOutputModes": ["text"],
            "message": message,
        }

        # 如果支持streaming，就一边收，一边yield
        response_stream = self.client.send_task_streaming(payload)
        async for result in response_stream:
            print(f"收到的结果信息是: {result}")
            if result and result.result:
                if isinstance(result.result, TaskStatusUpdateEvent):
                    # print(f"收到状态更新TaskStatusUpdateEvent",result.result)
                    if result.result and result.result.status.message and result.result.status.message.parts:
                        type = result.result.status.message.parts[0].type
                        if type == "text":
                            text = result.result.status.message.parts[0].text
                            # print(f"TaskStatusUpdateEvent,Text内容: {text}")
                            if not text:
                                # 去掉空的思考
                                continue
                            yield {
                                "type": "reasoning", "reasoning": text
                            }
                        elif type == "data":
                            data = result.result.status.message.parts[0].data
                            # print(f"TaskStatusUpdateEvent,data内容: {data}")
                            yield {
                                "type": "data", "data": data
                            }
                        else:
                            print(f"收到的数据类型未知: {type}")
                elif isinstance(result.result, TaskArtifactUpdateEvent):
                    for part in result.result.artifact.parts:
                        if isinstance(part, TextPart) and part.text:
                            # print(f"TaskArtifactUpdateEvent,Text内容: {part.text}")
                            yield {
                                "type": "text", "text": part.text,
                            }
            else:
                print(f"收到了其它消息内容，{result}")
        print(f"任务结束，返回flag为done的信息。")
        yield {
            "type": "final", "text": "[DONE]",
        }

    async def _complete_task(self, prompt: str, task_id: str):
        message = {
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
        }

        payload = {
            "id": task_id,
            "sessionId": self.session_id,
            "acceptedOutputModes": ["text"],
            "message": message,
        }

        if self.use_push_notifications:
            payload["pushNotification"] = {
                "url": f"http://{self.notification_receiver_host}:{self.notification_receiver_port}/notify",
                "authentication": {
                    "schemes": ["bearer"],
                },
            }

        if self.streaming:
            response_stream = self.client.send_task_streaming(payload)
            async for result in response_stream:
                print(f"stream event => {result.model_dump_json(exclude_none=True)}")
            task_result = await self.client.get_task({"id": task_id})
        else:
            task_result = await self.client.send_task(payload)
            print(f"\n{task_result.model_dump_json(exclude_none=True)}")

        state = TaskState(task_result.result.status.state)
        if state.name == TaskState.INPUT_REQUIRED.name:
            # 递归重新调用直到任务完成
            return await self._complete_task(prompt, task_id)
        else:
            return True


# 作为脚本运行时调用
if __name__ == "__main__":
    async def main():
        session_id = time.strftime("%Y%m%d%H%M%S",time.localtime())
        wrapper = A2AClientWrapper(session_id=session_id, agent_url="http://localhost:10006")
        await wrapper.run("乳腺癌的治疗方案有哪些?")
    asyncio.run(main())
