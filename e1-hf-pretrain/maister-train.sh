#!/bin/bash


HF_CACHE="$PWD/hfcache"
HF_HOME="$PWD/hfhome"
mkdir -p $HF_CACHE
mkdir -p $HF_HOME
mkdir -p torch_temp
mkdir -p cupy_cache
mkdir -p nltk_data
singularity exec nvhf2.sif python -m nltk.downloader -d nltk_data punkt
OUTPUT_DIR="output/"
mkdir -p $OUTPUT_DIR
mkdir -p logs

tar xzf slen-from-cse-bert.tar.gz # pre-trained model to further train

# if checkpoint files are provided, uncomment the following two lines
#tar xzf slen4-from-pre-checkpoint.tar.gz # checkpoint files of currently trained model, if existing. Will automatically restart from it if in output dir
#mv checkpoint-* $OUTPUT_DIR
tar xzf slen-lbl-eval-preprocessed.tar.gz
tar xzf slen-lbl-train-preprocessed.tar.gz

SINGULARITYENV_HF_CACHE=$HF_CACHE \
SINGULARITYENV_HF_HOME=$HF_HOME \
SINGULARITYENV_NLTK_DATA=$PWD/nltk_data \
SINGULARITYENV_CUPY_CACHE_DIR=$PWD/cupy_cache \
SINGULARITYENV_TORCH_EXTENSIONS_DIR=$PWD/torch_temp \
SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
timeout 46h singularity exec --nv nvhf2.sif \
deepspeed run_mlm_preprocessed.py \
 --deepspeed ds_config.json \
 --model_name_or_path ./slen-from-cse-bert \
 --train_file slen-lbl-train-preprocessed \
 --validation_file slen-lbl-eval-preprocessed \
 --do_train \
 --do_eval \
 --line_by_line \
 --max_seq_length 512 \
 --output_dir $OUTPUT_DIR \
 --per_device_train_batch_size 32 \
 --per_device_eval_batch_size 32 \
 --evaluation_strategy steps \
 --eval_steps 150000 \
 --adam_beta1 0.9 \
 --adam_beta2 0.999 \
 --adam_epsilon 1e-8 \
 --num_train_epochs 1 \
 --fp16 \
 --lr_scheduler_type cosine \
 --warmup_ratio 0.01 \
 --mlm_probability 0.15 \
 --logging_dir logs \
 --logging_strategy steps \
 --logging_steps 1000 \
 --save_strategy steps \
 --save_steps 10000 \
 --save_total_limit 1 \
 --gradient_accumulation_steps 2 \
 --dataloader_num_workers 2


tar czf output.tar.gz $OUTPUT_DIR
tar czf logs.tar.gz logs
