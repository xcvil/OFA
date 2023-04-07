#!/bin/bash
#SBATCH --output=/cluster/work/medinfmk/MedVLM/output/output_%J.txt
#SBATCH --error=/cluster/work/medinfmk/MedVLM/error/error_%j.txt
#SBATCH --job-name=mod_tra              # create a short name for your job
#SBATCH --partition=gpu
#SBATCH --nodes=1                       # node count
#SBATCH --gres=gpu:rtx1080ti:3            # titan_rtx & geforce_rtx_3090 & tesla_v100 & geforce_rtx_2080_ti & rtx_a6000
#SBATCH --cpus-per-task=3               # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G               # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=48:00:00                 # total run time limit (HH:MM:SS)

# Send more noteworthy information to the output log
echo "Started at:     $(date)"

source ~/.bashrc
source ~/.bashrc.xzheng
conda activate base

python test.py

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0