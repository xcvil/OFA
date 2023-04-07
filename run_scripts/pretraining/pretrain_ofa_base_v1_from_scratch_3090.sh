#!/bin/bash
#SBATCH --output=/cluster/work/medinfmk/MedVLM/output/output_%J.txt
#SBATCH --error=/cluster/work/medinfmk/MedVLM/error/error_%j.txt
#SBATCH --job-name=mod_tra              # create a short name for your job
#SBATCH --partition=gpu
#SBATCH --nodes=1                       # node count
#SBATCH --gres=gpu:rtx3090:3            # titan_rtx & geforce_rtx_3090 & tesla_v100 & geforce_rtx_2080_ti & rtx_a6000
#SBATCH --cpus-per-task=3               # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G               # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=96:00:00                 # total run time limit (HH:MM:SS)

# Send more noteworthy information to the output log
echo "Started at:     $(date)"

source ~/.bashrc
source ~/.bashrc.xzheng
conda activate med

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=9063
export CUDA_VISIBLE_DEVICES=0,1,2
export GPUS_PER_NODE=3

bpe_dir=/cluster/customapps/medinfmk/xiaochen/OFA/utils/BPE
user_dir=/cluster/customapps/medinfmk/xiaochen/OFA/ofa_module
data_dir=/cluster/work/medinfmk/MedVLM/dataset/ofa-pretrain-v1
neg_sample_dir=${data_dir}/negative_sample
data=${data_dir}/vision_language_examples.tsv
text_data=${data_dir}/text_examples.tsv
detection_data=${data_dir}/detection_examples.tsv

selected_cols=0,1,2,3,4,5,6,7
text_selected_cols=0,1
image_selected_cols=0,1,2
detection_selected_cols=0,1,2


task=unify_task
arch=ofa_base
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.0
lr=1e-4
max_epoch=200
warmup_ratio=0.01
batch_size=11
update_freq=1
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=30
num_bins=1000
patch_image_size=384
sample_patch_num=196
max_image_size=512

save_path=/cluster/work/medinfmk/MedVLM/ckpt/leomed-base-3090-3-11-from-scratch

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} /cluster/customapps/medinfmk/xiaochen/OFA/train.py \
  $data \
  --text-data=${text_data} \
  --detection-data=${detection_data} \
  --selected-cols=${selected_cols} \
  --text-selected-cols=${text_selected_cols} \
  --detection-selected-cols=${detection_selected_cols} \
  --bpe-dir=${bpe_dir} \
  --user-dir=${user_dir} \
  --reset-optimizer --reset-dataloader --reset-meters \
  --save-dir=${save_path} \
  --neg-sample-dir=${neg_sample_dir} \
  --task=${task} \
  --arch=${arch} \
  --criterion=${criterion} \
  --label-smoothing=${label_smoothing} \
  --batch-size=${batch_size} \
  --update-freq=${update_freq} \
  --encoder-normalize-before \
  --decoder-normalize-before \
  --share-decoder-input-output-embed \
  --share-all-embeddings \
  --layernorm-embedding \
  --patch-layernorm-embedding \
  --code-layernorm-embedding \
  --resnet-drop-path-rate=${resnet_drop_path_rate} \
  --encoder-drop-path-rate=${encoder_drop_path_rate} \
  --decoder-drop-path-rate=${decoder_drop_path_rate} \
  --dropout=${dropout} \
  --attention-dropout=${attention_dropout} \
  --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=5.0 \
  --lr-scheduler=polynomial_decay --lr=${lr} \
  --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
  --log-format=simple --log-interval=10 \
  --fixed-validation-seed=7 \
  --keep-last-epochs=15 \
  --save-interval=1 \
  --save-interval-updates=6000 \
  --disable-validation \
  --max-src-length=${max_src_length} \
  --max-tgt-length=${max_tgt_length} \
  --add-type-embedding \
  --scale-attn \
  --scale-fc \
  --scale-heads \
  --disable-entangle \
  --num-bins=${num_bins} \
  --patch-image-size=${patch_image_size} \
  --sample-patch-num=${sample_patch_num} \
  --max-image-size=${max_image_size} \
  --fp16 \
  --fp16-scale-window=128 \
  --num-workers=0 \
  --ddp-backend=no_c10d \

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0