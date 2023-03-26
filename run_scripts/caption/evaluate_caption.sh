#!/bin/bash
#SBATCH --mail-type=FAIL                # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=output/%j.out
#SBATCH --error=log/%j.err
#SBATCH --job-name=mod_tra              # create a short name for your job
#SBATCH --nodes=1                       # node count
#SBATCH --gres=gpu:1   # titan_rtx & geforce_rtx_3090 & tesla_v100 & geforce_rtx_2080_ti & rtx_a6000
#SBATCH --cpus-per-task=3               # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=24G               # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=18:00:00                 # total run time limit (HH:MM:SS)

# Exit on errors
set -o errexit

source ~/.bashrc.xzheng
conda activate med

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1081

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

data=../../dataset/caption_data/caption_test.tsv
path=../../checkpoints/caption_base_best.pt
result_path=../../results/caption
selected_cols=1,4,2
split='test'

CUDA_VISIBLE_DEVICES=0 python3 ../../evaluate.py \
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