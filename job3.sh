#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=6:0:0    
#SBATCH --mail-user=aligholamzadeh42@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:v100l:1


cd $SLURM_TMPDIR
git clone git@github.com:aligh42/VPL-RNN.git
cd ./VPL-RNN

mkdir -p data/output0
mkdir -p data/output1
mkdir -p data/output2
mkdir -p data/output3
mkdir -p data/output4

module purge
module load python/3.7.9 scipy-stack
source $project/py37/bin/activate

python main_MSE.py

tar -zcf $project/VPL/VPL-RNN/result-GRU-MSE0.tar data/output0/
tar -zcf $project/VPL/VPL-RNN/result-GRU-MSE1.tar data/output1/
tar -zcf $project/VPL/VPL-RNN/result-GRU-MSE2.tar data/output2/
tar -zcf $project/VPL/VPL-RNN/result-GRU-MSE3.tar data/output3/
tar -zcf $project/VPL/VPL-RNN/result-GRU-MSE4.tar data/output4/

