#!/bin/bash

cd backend
nohup python3 bot.py &
cd ../
node frontend/server.js
