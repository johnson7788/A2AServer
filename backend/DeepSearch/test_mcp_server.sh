#!/bin/bash
# 测试是否mcp的脚本服务能正常运行
# 定义配置项
declare -A scripts=(
  ["search_tool"]="mcpserver/search_tool.py"
)

# 循环执行所有脚本
for name in "${!scripts[@]}"; do
  echo "----------------------------"
  echo "Running: $name"
  echo "----------------------------"
  python "${scripts[$name]}"
  # 后台运行 uv 命令
  uv run --with fastmcp fastmcp run "${scripts[$name]}" &
  pid=$!

  # 等待 2 秒
  sleep 2

  # 终止后台进程
  kill $pid 2>/dev/null

  # 等待进程彻底退出（可选）
  wait $pid 2>/dev/null

  echo "$name stopped after 2 seconds."
  echo ""
done
