export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=0,1,2,3

DATASET_PATH="../data"

python -u predict.py \
                   --dataset_path=${DATASET_PATH} \
                   --batch_size=8 \
                   --use_gpu=True \
                   --checkpoint_dir="./ckpt_dureader40" \
                   --max_seq_len=512 \