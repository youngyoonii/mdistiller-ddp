torchrun --nproc-per-node=$CUDA_DEVICE_COUNT  tools/train.py --cfg test_clip.yaml ./configs/imagenet/optim/adamw.yaml

torchrun --nproc-per-node=4  tools/train.py --cfg ./configs/imagenet/dinov2/test.yaml ./configs/imagenet/optim/adamw.yaml


export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_DEVICE_COUNT=4
export OMP_NUM_THREADS=4
torchrun --nproc-per-node=$CUDA_DEVICE_COUNT --master_port=6789  tools/train.py --cfg ./configs/imagenet/clip/fitvit_remove.yaml ./configs/imagenet/optim/adamw.yaml

# TODO: for check
# baseline: clip_base >> clip_tiny feat_weight 1.0, 5.0, 10.0


