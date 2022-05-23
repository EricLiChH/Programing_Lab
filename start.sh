#!/bin/bash


if [[ "$1" == "start" ]]; 
then
cd backend
nohup python3 bot.py > /dev/null 2>log & 
cd ../frontend
nohup node server.js > /dev/null 2>log &
echo "服务已启动"

elif [[ "$1" == "stop" ]]; 
then
killall python3
killall node
echo "服务已停止"

else
echo "参数错误"
fi
