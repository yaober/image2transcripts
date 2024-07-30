#!/bin/bash
#SBATCH --job-name xenium_brca_0 

# Name of the SLURM partition that this job should run on.
#SBATCH -p 512GB    # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%j_xenium_brca_0.out
#SBATCH -e job_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

module load python/3.8.x-anaconda
conda activate CONCH
cd /archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript
python image2embedding.py --image_path ./data/xenium_brca_0 --output_path ./data/xenium_brca_0_embeddings
python image2embedding.py --image_path ./data/xenium_brca_1 --output_path ./data/xenium_brca_1_embeddings
python image2embedding.py --image_path ./data/xenium_brca_10 --output_path ./data/xenium_brca_10_embeddings