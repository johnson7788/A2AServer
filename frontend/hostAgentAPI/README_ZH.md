# Host Agent API 接口

**项目简介：**

本项目的目标是为 A2A的协调者和组织者的 Agent 提供启动和管理功能的 API 接口。通过这些接口，可以方便地和其它Agent进行交互，从而实现对其它Agent控制。


**快速开始：**

1.  **环境准备：**
    * 确保你的系统已经安装了 Python 3。
    * （如果项目有依赖）使用 pip 安装所需的依赖包：
        ```bash
        pip install -r requirements.txt
        ```
2. **配置模型：**
    * cp env_template.txt 为 .env 
    * 修改/hostAgentAPI/hosts/multiagent/host_agent.py中的模型，
    * model=LiteLlm(model="deepseek/deepseek-chat", api_key="xxx", api_base="")


3. **启动 API 服务：**
    * 在项目根目录下，运行以下命令启动 API 服务：
        ```bash
        python api.py
        ```
    * 默认情况下，API 服务可能会在本地的某个端口（例如 `http://localhost:13000`）启动。

**API 接口测试：**

项目提供了一个 `test_api.py` 脚本用于测试各个 API 接口的功能是否正常。

1.  **运行测试脚本：**
    * 确保 API 服务已经成功启动（见上面的“启动”步骤）。
    * 在项目根目录下，运行测试脚本：
        ```bash
        python test_api.py
        ```
    * 测试脚本将会输出每个接口的测试结果，帮助你验证 API 的可用性。


**API 接口测试：**

项目提供了一个 `test_api.py` 脚本用于测试各个 API 接口的功能是否正常。该脚本使用了 `unittest` 框架，对每个接口发送请求并验证响应。

1.  **运行测试脚本：**
    * 确保 API 服务已经成功启动（见上面的“启动”步骤）。
    * 在项目根目录下，运行测试脚本：
        ```bash
        python test_api.py
        ```
    * 测试脚本将会自动执行所有测试用例，并输出每个接口的测试结果，包括状态码、响应内容和耗时等信息，帮助你验证 API 的可用性。

**API 端点说明：**

以下是根据 `test_api.py` 文件分析出的 API 端点及其功能说明：

* **`/ping` (GET)**
    * **功能：** 测试 API 服务是否启动并正常运行。
    * **请求示例：** `GET http://127.0.0.1:13000/ping`
    * **响应示例：** `"Pong"`

* **`/conversation/create` (POST)**
    * **功能：** 创建一个新的对话。
    * **请求体：** 无
    * **响应示例：**
        ```json
        {
          "result": {
            "conversation_id": "生成的唯一对话ID"
          }
        }
        ```

* **`/conversation/list` (POST)**
    * **功能：** 列出当前所有的对话。
    * **请求体：** 无
    * **响应示例：**
        ```json
        {
          "result": [
            "conversation_id_1",
            "conversation_id_2",
            ...
          ]
        }
        ```

* **`/message/send` (POST)**
    * **功能：** 发送消息到指定的对话。
    * **请求体 (application/json)：**
        ```json
        {
          "params": {
            "role": "user",
            "parts": [{"type": "text", "text": "发送的消息内容"}],
            "metadata": {"conversation_id": "目标对话ID"}
          }
        }
        ```
    * **响应示例：**
        ```json
        {
          "result": {
            "message_id": "生成的消息ID",
            "conversation_id": "对应的对话ID"
          }
        }
        ```

* **`/message/list` (POST)**
    * **功能：** 列出指定对话中的所有消息。
    * **请求体 (application/json)：**
        ```json
        {
          "params": "目标对话ID"
        }
        ```
    * **响应示例：**
        ```json
        {
          "result": [
            {
              "metadata": {"message_id": "消息ID_1", ...},
              "role": "user",
              "parts": [{"type": "text", "text": "消息内容_1"}]
            },
            {
              "metadata": {"message_id": "消息ID_2", ...},
              "role": "assistant",
              "parts": [{"type": "text", "text": "回复内容_1"}]
            },
            ...
          ]
        }
        ```

* **`/message/pending` (POST)**
    * **功能：** 获取当前正在处理（pending）的消息。
    * **请求体：** 无
    * **响应示例：**
        ```json
        {
          "result": [
            ["conversation_id_1", "message_id_1"],
            ["conversation_id_2", "message_id_2"],
            ...
          ]
        }
        ```
        *注意：如果 `result` 为空列表，则表示当前没有正在处理的消息。*

* **`/events/get` (POST)**
    * **功能：** 获取发生的事件列表（例如，新消息事件）。
    * **请求体：** 无
    * **响应示例：**
        ```json
        {
          "result": [
            {"event_type": "new_message", "data": {...}},
            ...
          ]
        }
        ```
        *注意：当有新的提问或交互发生时，`result` 中可能会包含相应的事件信息。*

* **`/task/list` (POST)**
    * **功能：** 列出当前的任务。
    * **请求体：** 无
    * **响应示例：**
        ```json
        {
          "result": [
            {"task_id": "task_id_1", "status": "running", ...},
            ...
          ]
        }
        ```

* **`/agent/register` (POST)**
    * **功能：** 注册一个新的 Agent。
    * **请求体 (application/json)：**
        ```json
        {
          "params": "Agent的URL地址 (例如: 127.0.0.1:10003)"
        }
        ```
    * **响应示例：**
        ```json
        {
          "result": "注册结果信息"
        }
        ```

* **`/agent/list` (POST)**
    * **功能：** 列出所有已注册的 Agent。
    * **请求体：** 无
    * **响应示例：**
        ```json
        {
          "result": ["agent_url_1", "agent_url_2", ...]
        }
        ```

* **`/api_key/update` (POST)**
    * **功能：** 更新 API Key。
    * **请求体 (application/json)：**
        ```json
        {
          "api_key": "新的API Key"
        }
        ```
    * **响应示例：**
        ```json
        {
          "status": "success"
        }
        ```