#!/bin/bash
#SBATCH --job-name train_model2_img2trans 

# Name of the SLURM partition that this job should run on.
#SBATCH -p GPU4v100    # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%j_train_94.out
#SBATCH -e job_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu


module load gpu_prepare
module load python/3.8.x-anaconda

source activate /archive/DPDS/Xiao_lab/shared/jia_yao/envs/image2transcripts

export CUDA_VISIBLE_DEVICES=0,1,2,3

# --- UPDATED SECTION START ---
# Define paths for clarity
NVRTC_LIB=/endosome/archive/DPDS/Xiao_lab/shared/jia_yao/envs/image2transcripts/lib/python3.9/site-packages/nvidia/cuda_nvrtc/lib
CUDNN_LIB=/endosome/archive/DPDS/Xiao_lab/shared/jia_yao/envs/image2transcripts/lib/python3.9/site-packages/nvidia/cudnn/lib

# Add BOTH paths to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NVRTC_LIB:$CUDNN_LIB:$LD_LIBRARY_PATH
# --- UPDATED SECTION END ---

cd /archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript/model
python main.py \
    --gene_dir ../data/gene_expression/ \
    --img_dir ../data/images/ \
    --out_dir ../output_improved \
    --batch 128 \
    --epochs 100 \
    --lr_backbone 1e-5 \
    --lr_head 5e-4 \
    --weight_decay 0.05 \
    --warmup_ratio 0.05 \
    --grad_clip 1.0 \
    --w_contrast 0.1 \
    --w_align 0.1 \
    --w_zinb 1.0 \
    --gene_mask_ratio 0.15 \
    --patience 15