#!/bin/bash
#SBATCH --job-name=whattrack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --qos=students
#SBATCH --output=logs_out/%x-%j.out
#SBATCH --error=logs_out/%x-%j.err
#SBATCH --time=1-00:00:00

uv run main.py
