#!/bin/bash
# $1,数据集名称
cd ../
# Timestamp
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
filename="./log/train_$1_$timestamp.log"
# Train
nohup python main.py --cfg_file="./config/$1/config_server.json" --run_type="train" --model="gaf" > $filename &