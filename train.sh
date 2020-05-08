#!/bin/bash
export PYTHONIOENCODING=utf-8

if [ -z "$CUDA_VISIBLE_DEVICES" ];then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"


if [ -z "$PRETRAINED_MODEL_PATH" ];then
    PRETRAINED_MODEL_PATH="./pretrained_model"
fi
echo "PRETRAINED_MODEL_PATH=$PRETRAINED_MODEL_PATH"

python -u src/run_mrc.py --use_cuda true \
                --batch_size 32 \
                --checkpoints output \
                --init_pretraining_params ${PRETRAINED_MODEL_PATH}/params \
                --vocab_path ${PRETRAINED_MODEL_PATH}/vocab.txt \
                --ernie_config ${PRETRAINED_MODEL_PATH}/ernie_config.json \
                --save_steps 100 \
                --warmup_proportion 0.1 \
                --weight_decay  0.01 \
                --epoch 10 \
                --max_seq_len 512 \
                --do_lower_case true \
                --doc_stride 128 \
                --learning_rate 3e-5 \
                --skip_steps 25 \
                --max_answer_length 40 \
                --do_train true \
                --do_predict false \
                --use_fp16 false \
                --train_file data/train.json \
                $@