#!/bin/bash
#SBATCH --job-name data_process

# Name of the SLURM partition that this job should run on.
#SBATCH -p GPU4A100    # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%j.out
#SBATCH -e job_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

for (( ; ; ))
do
  sleep 10
done
