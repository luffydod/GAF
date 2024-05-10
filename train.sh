#!/bin/bash

# Timestamp
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
filename="./log/train_$timestamp.log"
# Train
nohup python main.py --cfg_file="./config/METR-LA/config_server.json" --run_type="train" > $filename &