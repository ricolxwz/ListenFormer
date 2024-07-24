
CUDA_VISIBLE_DEVICES=0 python -u -m torch.distributed.launch --nproc_per_node 1 --master_port 2343 train.py \
    --batch_size 128 \
    --time_size 90 \
    --max_epochs 200 \
    --change_epoch 100 \
    --lr 0.002 \
    --task listener \
    --output_path saved/listenformer

