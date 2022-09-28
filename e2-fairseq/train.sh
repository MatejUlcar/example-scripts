#!/bin/bash

TOTAL_UPDATES=204640            # Total number of training steps (5116 per epoch for en-hi)
WARMUP_UPDATES=10000            # Warmup the learning rate over this many updates
PEAK_LR=0.0005                  # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512           # Max sequence length
MAX_POSITIONS=512               # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16                # Number of sequences per batch (batch size)
NUM_GPUS=4
MAX_TOKENS=$((5120*$NUM_GPUS))  # For 4 gpus use 20480, for 1 use 5120, for 2 use 10240
UPDATE_FREQ=$((128/$NUM_GPUS))  # Increase the batch size 32x (for 4 gpus), 64x for 2 gpus, 128x for 1 gpu


CURR_EPOCH=21 # How many epochs already trained for

DATA_DIR=enhi/binary_data
CHECKPOINT_DIR=enhi/checkpoints

# the following 2 should be done by hand in user's home folder
mkdir -p enhi/checkpoints
tar xf indo-slov-fairseq-data.tar -C enhi/
mv dict.txt $DATA_DIR/dict.txt

export SINGULARITYENV_PYTHONPATH=$SINGULARITYENV_PYTHONPATH:/fairseq/

if [ $CURR_EPOCH -gt 0 ]
then
    tar xzf enhi_checkpoints.tar.gz
    timeout 47h singularity exec --nv fairseq.sif \
    fairseq-train --fp16 $DATA_DIR \
        --task masked_lm --criterion masked_lm \
        --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
        --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --max-tokens $MAX_TOKENS \
        --update-freq $UPDATE_FREQ \
        --max-update $TOTAL_UPDATES --log-format simple --log-interval 100 \
        --skip-invalid-size-inputs-valid-test \
        --mask-whole-words \
        --bpe sentencepiece \
        --sentencepiece-model en-hi_spm.model \
        --keep-last-epochs 1 \
        --num-workers 2 \
        --restore-file ${CHECKPOINT_DIR}/checkpoint${CURR_EPOCH}.pt \
        --save-dir $CHECKPOINT_DIR
else
    timeout 47h singularity exec --nv fairseq.sif \
    fairseq-train --fp16 $DATA_DIR \
        --task masked_lm --criterion masked_lm \
        --arch roberta_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
        --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
        --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
        --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
        --max-tokens $MAX_TOKENS \
        --update-freq $UPDATE_FREQ \
        --max-update $TOTAL_UPDATES --log-format simple --log-interval 10 \
        --skip-invalid-size-inputs-valid-test \
        --mask-whole-words \
        --bpe sentencepiece \
        --sentencepiece-model en-hi_spm.model \
        --keep-last-epochs 1 \
        --num-workers 2 \
        --save-dir $CHECKPOINT_DIR
fi
tar czf output.tar.gz $CHECKPOINT_DIR
