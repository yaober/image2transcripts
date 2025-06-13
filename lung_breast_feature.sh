#!/bin/bash
#SBATCH --job-name lung_breast_feature

# Name of the SLURM partition that this job should run on.
#SBATCH -p 512GB    # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%j.out
#SBATCH -e job_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu
module load python/3.8.x-anaconda
source activate image2transcripts
cd /archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript/colon
python -u crop_patch.py

#python -u summarize_tme_features.py --model_res_path ./test_lung --output_dir ./test_output_tme_gpu --n_patches 100 --patch_size 512 --score_thresh 10 --scale_factor 16 --save_images --save_nuclei

