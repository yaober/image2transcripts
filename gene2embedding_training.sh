#!/bin/bash
#SBATCH --job-name gene2embedding_training 

# Name of the SLURM partition that this job should run on.
#SBATCH -p  GPU4A100   # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%jgene2embedding_training.out
#SBATCH -e job_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu
module load gpu_prepare
export CUDA_VISIBLE_DEVICES=0,1,2,3
module load python/3.8.x-anaconda
conda activate scGPT
cd /archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript
python gene2embedding_training.py