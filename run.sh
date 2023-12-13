#!/bin/bash

# 使用pgrep查找正在运行的Qwen.py的进程ID
PID=$(pgrep -f Qwen.py)

# 如果找到了进程ID，则杀死该进程
if [ ! -z "$PID" ]; then
    echo "Killing Qwen.py with PID $PID"
    kill $PID
fi

# 等待一小段时间以确保进程已经被杀死
sleep 2

# 启动Qwen.py Python脚本
echo "Starting Qwen.py"
python Qwen.py > qwen.log 2>&1 &
