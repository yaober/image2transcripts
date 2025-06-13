#!/bin/bash
#SBATCH --job-name colon_cancer 

# Name of the SLURM partition that this job should run on.
#SBATCH -p 512GB    # partition (queue)
# Number of nodes required to run this job
#SBATCH -N 1

#SBATCH -t 100-23:0:00

#SBATCH -o job_%j_colon_cancer_image2embedding.out
#SBATCH -e job_%j.err

#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu
source activate CONCH

cd /archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript/scripts
python image2embedding.py --image_path ../data/colon_cancer/output-XETG00248__0014303__CA432__20240306__011452 --output_path ../data/image_embeddings/output-XETG00248__0014303__CA432__20240306__011452
python image2embedding.py --image_path ../data/colon_cancer/output-XETG00248__0014303__CA564__20240306__011452 --output_path ../data/image_embeddings/output-XETG00248__0014303__CA564__20240306__011452
python image2embedding.py --image_path ../data/colon_cancer/output-XETG00248__0014303__NL432__20240306__011452 --output_path ../data/image_embeddings/output-XETG00248__0014303__NL432__20240306__011452
python image2embedding.py --image_path ../data/colon_cancer/output-XETG00248__0014303__NL564__20240306__011452 --output_path ../data/image_embeddings/ooutput-XETG00248__0014303__NL564__20240306__011452
python image2embedding.py --image_path ../data/colon_cancer/output-XETG00248__0014306__CA497__20240306__011452 --output_path ../data/image_embeddings/output-XETG00248__0014306__CA497__20240306__011452
python image2embedding.py --image_path ../data/colon_cancer/output-XETG00248__0014306__CA560__20240306__011452 --output_path ../data/image_embeddings/output-XETG00248__0014306__CA560__20240306__011452
python image2embedding.py --image_path ../data/colon_cancer/output-XETG00248__0014306__NL497__20240306__011452 --output_path ../data/image_embeddings/output-XETG00248__0014306__NL497__20240306__011452
python image2embedding.py --image_path ../data/colon_cancer/output-XETG00248__0014306__NL560__20240306__011452 --output_path ../data/image_embeddings/output-XETG00248__0014306__NL560__20240306__011452
