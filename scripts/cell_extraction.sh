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
conda activate image2transcripts
cd /archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript
python cell_extraction.py --wsi ../data/Xenium/breast_cancer_rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tiff --mask ../data/Xenium/breast_cancer_rep1/outs/cells.zarr.zip --output ../data/xenium_brca_rep1_0_padding --cell_count 0
python cell_extraction.py --wsi ../data/Xenium/breast_cancer_rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.tiff --mask ../data/Xenium/breast_cancer_rep1/outs/cells.zarr.zip --output ../data/xenium_brca_rep1_10_padding --cell_count 10
python cell_extraction.py --wsi ../data/Xenium/breast_cancer_rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.tif --mask ../data/Xenium/breast_cancer_rep2/outs/cells.zarr.zip --output ../data/xenium_brca_rep2_0_padding --cell_count 0
python cell_extraction.py --wsi ../data/Xenium/breast_cancer_rep2/Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.tif --mask ../data/Xenium/breast_cancer_rep2/outs/cells.zarr.zip --output ../data/xenium_brca_rep2_10_padding --cell_count 10
