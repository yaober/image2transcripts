#!/bin/bash
#SBATCH --job-name create_hdf5 

# Name of the SLURM partition that this job should run on.
#SBATCH -p 512GB    # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%j_create_hdf5.out
#SBATCH -e job_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

module load python/3.8.x-anaconda
conda activate scGPT
cd /archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript/scripts
python create_hdf5.py