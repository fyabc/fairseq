#!/usr/bin/env bash
# Checkpoints: 43, 58, 73, 85, 88
REMOTE_DATA_PATH="/home/v-yaf/DataTransfer/fairseq/wmt14_en_de_joined_dict"
REMOTE_MODEL_PATH="/home/v-yaf/DataTransfer/fairseq/WMT14_EN_DE_CYCLE_4.5k"

checkpoint=${1}
shift

python generate.py ${REMOTE_DATA_PATH} --path ${REMOTE_MODEL_PATH}/checkpoint${checkpoint}.pt --batch-size 128 \
    --beam 5 --lenpen 0.6 --remove-bpe $*
#    --quiet --no-progress-bar
