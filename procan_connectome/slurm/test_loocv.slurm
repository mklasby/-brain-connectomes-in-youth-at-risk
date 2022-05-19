#!/bin/bash

## GET RESOURCES ##

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G    
#SBATCH --time=72:00:00
#SBATCH --job-name=SVCLOOCV

## SET ENV ##
export PATH=~/miniconda3/bin:$PATH
source activate ML

## RUN SCRIPT ##
python3 ../model_training/toy_dataset_test.py
