#!/usr/bin/env bash
#SBATCH --job-name=ssl-ft-c24
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesv100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lkoffma2@jh.edu
#SBATCH -o eofiles/%x_%A.out
#SBATCH -e eofiles/%x_%A.err

set -euo pipefail

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

module load conda/3-24.3.0
conda activate ssl_env

# Helpful to avoid CPU oversubscription
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Optional binding (JHPCE recommends it for performance)
# If this causes issues on your cluster, remove it.
SRUN_ARGS="--gpu-bind=closest"

REPO="/dcs05/ciprian/smart/ssl-wearables"
cd "${REPO}"

nvidia-smi

# ---- EDIT THESE PATHS ----
DATA_ROOT="/dcs05/ciprian/smart/ssl_wearables/capture24_100hz_w10_o0_small/"   # must contain X.npy, Y.npy, pid.npy
REPORT_ROOT="/dcs05/ciprian/smart/ssl_wearables/temp_reports/"                    # output CSVs
WEIGHTS="/dcs05/ciprian/smart/ssl_wearables/weights/mtl_best.mdl"            # pretrained UKB model

# Recommended: start modest, then scale up
# - num_split=1 for speed
# - increase evaluation.num_epoch for real run
srun ${SRUN_ARGS} python downstream_task_evaluation_v2.py \
  data=capture24_10s \
  data.data_root="${DATA_ROOT}" \
  data.Y_path="${DATA_ROOT}/Y.npy" \
  report_root="${REPORT_ROOT}" \
  evaluation=all \
  evaluation.flip_net_path="${WEIGHTS}" \
  gpu=0 \
  model=resnet \
  num_split=1 \
  evaluation.num_epoch=50 \
  evaluation.patience=5 \
  evaluation.num_workers=6

echo "End: $(date)"