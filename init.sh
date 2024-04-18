#!/bin/bash

# Check if ckpt directory exists
if [ ! -d "ckpt" ]; then
    echo "Creating ckpt directory..."
    mkdir ckpt
else
    echo "ckpt directory already exists."
fi

# Check if img directory exists
if [ ! -d "img" ]; then
    echo "Creating img directory..."
    mkdir img
else
    echo "img directory already exists."
fi

# Check if plot directory exists
if [ ! -d "plot" ]; then
    echo "Creating plot directory..."
    mkdir plot
else
    echo "plot directory already exists."
fi

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir data
else
    echo "data directory already exists."
fi

# Check if log directory exists
if [ ! -d "log" ]; then
    echo "Creating log directory..."
    mkdir log
else
    echo "log directory already exists."
fi

echo "Initialization completed."
