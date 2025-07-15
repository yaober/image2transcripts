#!/bin/bash
#SBATCH --job-name train_img2trans 

# Name of the SLURM partition that this job should run on.
#SBATCH -p GPUA100    # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%j_train_zinb.out
#SBATCH -e job_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

module load gpu_prepare
module load python/3.8.x-anaconda

source activate /archive/DPDS/Xiao_lab/shared/jia_yao/envs/image2transcripts
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=/archive/DPDS/Xiao_lab/shared/jia_yao/envs/image2transcripts/lib/python3.9/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
cd /archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript/model
python main.py --gene_dir ../data/demo/gene_expression/ --img_dir ../data/demo/images/ --out_dir ../test