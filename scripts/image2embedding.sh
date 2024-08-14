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
cd /archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript/scripts
python image2embedding.py --image_path ../data/xenium_brca_rep1_0_padding --output_path ../data/image_embeddings/xenium_brca_rep1_0_padding
python image2embedding.py --image_path ../data/xenium_brca_rep1_10_padding --output_path ../data/image_embeddings/xenium_brca_rep1_10_padding
python image2embedding.py --image_path ../data/xenium_brca_rep2_0_padding --output_path ../data/image_embeddings/xenium_brca_rep2_0_padding
python image2embedding.py --image_path ../data/xenium_brca_rep2_10_padding --output_path ../data/image_embeddings/xenium_brca_rep2_10_padding
python image2embedding.py --image_path ../data/xenium_brca_sample2_0_padding --output_path ../data/image_embeddings/xenium_brca_sample2_0_padding
python image2embedding.py --image_path ../data/xenium_brca_sample2_10_padding --output_path ../data/image_embeddings/xenium_brca_sample2_10_padding
python image2embedding.py --image_path ../data/xenium_lung_sample1_0_padding --output_path ../data/image_embeddings/xenium_lung_sample1_0_padding
python image2embedding.py --image_path ../data/xenium_lung_sample1_10_padding --output_path ../data/image_embeddings/xenium_lung_sample1_10_padding
python image2embedding.py --image_path ../data/xenium_lung_sample2_0_padding --output_path ../data/image_embeddings/xenium_lung_sample2_0_padding
python image2embedding.py --image_path ../data/xenium_lung_sample2_10_padding --output_path ../data/image_embeddings/xenium_lung_sample2_10_padding