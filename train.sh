#!/bin/bash

# Timestamp
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
filename="./log/train_$timestamp.log"
# Train
nohup python main.py > $filename &