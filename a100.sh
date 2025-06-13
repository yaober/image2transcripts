#!/bin/bash
#SBATCH --job-name train_img2trans 

# Name of the SLURM partition that this job should run on.
#SBATCH -p GPU4v100    # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%j_train.out
#SBATCH -e job_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

module load gpu_prepare
module load python/3.8.x-anaconda
source activate image2transcripts
cd /archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript
python train_ddp.py