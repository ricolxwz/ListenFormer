
CUDA_VISIBLE_DEVICES=0 python -u eval.py \
  --batch_size 1 \
  --output_path saved/listenformer_E200 \
  --resume saved/listenformer/checkpoints/Epoch_200.bin \
  --task listener
