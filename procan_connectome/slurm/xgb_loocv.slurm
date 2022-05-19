#!/bin/bash

## GET RESOURCES ##

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G    
#SBATCH --time=120:00:00
#SBATCH --job-name=XGB_log_min_grid

## SET ENV ##
export PATH=~/miniconda3/bin:$PATH
source activate ML

## RUN SCRIPT ##
python3 ../model_training/main_xgb.py
