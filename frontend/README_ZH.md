# 单Agent和多Agent前端
本项目展示了如何通过多个 A2A Agent 协同完成任务，包含多个 Agent 后端服务、统一调度接口以及前端界面。支持单个或多个 Agent 模式。

## 📁 项目结构
* hostAgentAPI：协调多个 Agent 的中控 API，决定调用哪个 Agent，查看状态等
* multiagent_front：多 Agent 协作模式的前端界面
* single_agent：单 Agent 模式下的前端界面

## 🚀 快速开始
### 一、启动多个 Agent 模式

#### 1. 启动1个A2A的Agent
启动Agent RAG
```
cd backend/AgentRAG
python main.py --port 10005
```

#### 2. 启动第二个Agent， DeepSearch
```
cd backend/DeepSearch
python main.py --port 10004
```

### 3. 启动host Agent， 用于协调多个Agent，决定使用哪个Agent和查看Agent的状态等
```
cd hostAgentAPI
pip install -r requirements.txt
python api.py
```

### 4. 启动前端
```
cd multiagent_front
npm install
npm run dev
```
### 5. 在网页中添加 Agent 配置并开始问答
打开前端页面后，添加各个 Agent 的地址及信息
输入问题，观察多个 Agent 的协作响应过程


### 二、单个A2A模式

### 1. 启动1个A2A的Agent，例如启动Agent RAG
```
cd backend/AgentRAG
python main.py --port 10005
```
### 2. 启动前端
```
cd single_agent
npm install
npm run dev
```

### 3. 打开前端页面，输入要使用的Agent的URl地址，开始问答


## 💡 项目亮点
多 Agent 调度与协作框架，易于扩展和集成更多智能 Agent

前后端分离，界面清晰，支持动态添加 Agent

支持单独测试某个 Agent 的能力与效果

## 📌 注意事项
所有服务默认本地运行，请确保端口未被占用