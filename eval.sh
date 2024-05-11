#!/bin/bash
# $1,数据集名称

# Eval
python main.py --cfg_file="./config/$1/config_server.json" --run_type="eval" --model="gaf"