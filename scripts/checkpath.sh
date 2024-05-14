#!/bin/bash

cd ../

if [ ! -d "ckpt" ]; then
    echo "Creating ckpt directory..."
    mkdir ckpt
else
    echo "ckpt directory already exists."
fi

if [ ! -d "img/plot" ]; then
    echo "Creating img/log directory..."
    mkdir -p img/plot
else
    echo "img directory already exists."
fi

if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir data
else
    echo "data directory already exists."
fi

if [ ! -d "log" ]; then
    echo "Creating log directory..."
    mkdir log
else
    echo "log directory already exists."
fi

echo "Initialization completed."
