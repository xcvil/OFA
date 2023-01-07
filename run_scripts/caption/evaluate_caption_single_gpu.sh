#!/bin/bash
#SBATCH --job-name=ofa-single    # create a short name for your job
#SBATCH --partition gpu          # request a specific partition for the resource allocation.
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gres=gpu:rtx3090:1     # number of gpus per node with specified GPU tpyes
#SBATCH --cpus-per-task=3        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=ofa_single.out.txt

source ~/.bashrc
enable_modules

module load python/3.8
source /cluster/work/medinfmk/x_pipeline/multimodal/OFA-main/ofa/bin/activate

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

data=../../dataset/caption_data/caption_test.tsv
path=../../checkpoints/caption_large_best_clean.pt
result_path=../../results/caption
selected_cols=1,4,2
split='test'

CUDA_VISIBLE_DEVICES=0 python3 ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=caption \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=16 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"

python coco_eval.py ../../results/caption/test_predict.json ../../dataset/caption_data/test_caption_coco_format.json