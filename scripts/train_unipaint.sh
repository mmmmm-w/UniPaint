export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
torchrun --nnodes=1 --nproc_per_node=4 train_unipaint.py --config configs/training/v1/unipaint_training.yaml