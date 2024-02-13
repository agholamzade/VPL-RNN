#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=1:0:0    
#SBATCH --mail-user=aligholamzadeh42@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100l:1


cd $SLURM_TMPDIR
git clone git@github.com:aligh42/VPL-RNN.git
cd ./VPL-RNN
mkdir -p data/output
module purge
module load python/3.7.9 scipy-stack
source $project/py37/bin/activate

pip install wandb --upgrade

python main.py

tar -zcf $project/VPL/VPL-RNN/result-GRU-BCE-MSE.tar data/output/
tar -zcf $project/VPL/VPL-RNN/wand_res.tar ./wandb/

