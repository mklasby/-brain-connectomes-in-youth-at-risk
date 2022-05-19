#!/bin/bash

## GET RESOURCES ##

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G    
#SBATCH --time=48:00:00
#SBATCH --job-name=RF-cf+pt+pca+if

## SET ENV ##
export PATH=~/miniconda3/bin:$PATH
source activate ML

## RUN SCRIPT ##
python3 ../model_training/main_rf.py
