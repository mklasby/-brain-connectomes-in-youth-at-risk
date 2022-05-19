#!/bin/bash

## GET RESOURCES ##

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=20G    
#SBATCH --time=25:00:00

## SET ENV ##
export PATH=~/software/miniconda3/bin:$PATH
conda activate ML

## RUN SCRIPT ##
python3 ../model_training/tpot_trial.py
