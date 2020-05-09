export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DATASET_PATH="../data"

python -u reading_comprehension.py \
                   --dataset_path=${DATASET_PATH} \
                   --batch_size=32 \
                   --use_gpu=True \
                   --checkpoint_dir="./ckpt_dureader" \
                   --learning_rate=1e-6 \
                   --weight_decay=0.01 \
                   --warmup_proportion=0.1 \
                   --num_epoch=5 \
                   --max_seq_len=512 \
                   --use_data_parallel=True