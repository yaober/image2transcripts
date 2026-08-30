#!/bin/bash
#SBATCH --job-name stai_fm_zinb_pg
#SBATCH -p GPU4v100
#SBATCH -N 1
#SBATCH -t 3-00:00:00
#SBATCH -o job_%j_fm_zinb_pg.out
#SBATCH -e job_%j_fm_zinb_pg.err
#SBATCH --mail-type ALL
#SBATCH --mail-user jia.yao@utsouthwestern.edu

# Single-backbone wrapper around scripts/4v100_fixedsplit_foundation.sh so
# prov-gigapath gets its own 3-day budget and can't be killed by an
# unrelated stall in another backbone (cf. job 10476797).
export FOUNDATION_MODELS="prov-gigapath"
exec bash /archive/DPDS/Xiao_lab/shared/jia_yao/Image2Transcript/scripts/4v100_fixedsplit_foundation.sh
