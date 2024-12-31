export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export OMP_NUM_THREADS=1
torchrun --nnodes=1 --nproc_per_node=6 --master_port=29500 train_unipaint_mask.py --config configs/unipaint/training_mix_data.yaml