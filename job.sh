#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=1:0:0    
#SBATCH --mail-user=aligholamzadeh42@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:p100:1


T