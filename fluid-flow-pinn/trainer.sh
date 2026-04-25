#!/bin/bash
#SBATCH --job-name=fluid_pinn
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --mail-type=END
#SBATCH --mail-user=dns5508@nyu.edu
#SBATCH --account=csci_3033_109-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --requeue

mkdir -p ./logs

singularity exec --bind /scratch --nv \
  --overlay /scratch/dns5508/env/another__overlay-25GB-500K.ext3:ro \
  /scratch/dns5508/ubuntu-20.04.3.sif \
  /bin/bash -c "
  source /ext3/miniconda3/etc/profile.d/conda.sh
  export PATH=/home/dns5508/.local/bin:\$PATH
  conda activate cv
  cd /scratch/dns5508/FluidFlowPinn/fluid-flow-pinn

  python train.py \
    --config configs/default.yaml \
    --fdst-path /scratch/dns5508/FDST_Data/ \
    --checkpoint-dir /scratch/dns5508/FluidFlowPinn/checkpoints/ \
    --epochs 20 \
    --wandb-project fluid-pinn
"
