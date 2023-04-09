#!/bin/bash
#SBATCH --output=/cluster/work/medinfmk/MedVLM/output/output_%J.txt
#SBATCH --error=/cluster/work/medinfmk/MedVLM/error/error_%j.txt
#SBATCH --job-name=eval              # create a short name for your job
#SBATCH --nodes=1                       # node count
#SBATCH --gres=gpu:rtx1080ti:1               # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G               # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=18:00:00                 # total run time limit (HH:MM:SS)

# Exit on errors
echo "Started at:     $(date)"

source ~/.bashrc
source ~/.bashrc.xzheng
conda activate med

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1081

bpe_dir=/cluster/customapps/medinfmk/xiaochen/OFA/utils/BPE
user_dir=/cluster/customapps/medinfmk/xiaochen/OFA/ofa_module

data=/cluster/work/medinfmk/MedVLM/dataset/caption_data/caption_test.tsv
path=/cluster/work/medinfmk/MedVLM/ckpt/leomed-medium-1080-4x4x8-from-tik/checkpoint45.pt
result_path=/cluster/work/medinfmk/MedVLM/results/caption
selected_cols=1,4,2
split='test'

CUDA_VISIBLE_DEVICES=0 python3 /cluster/customapps/medinfmk/xiaochen/OFA/evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --selected-cols=${selected_cols} \
    --bpe-dir=${bpe_dir} \
    --task=caption \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=72 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --zero-shot \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"

python coco_eval.py ../../results/caption/test_predict.json ../../dataset/caption_data/test_caption_coco_format.json

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0