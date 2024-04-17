path="--pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 --output_dir=results/logs/cifar10 --dataset_name=cifar10 --data_dir=~/tensorflow_datasets"
args="--gradient_accumulation_steps=1 --num_tokens=5 --resolution=128 --train_batch_size=50 --num_emb=100 --max_train_steps=8000"
lr="--lr_warmup_steps=0 --interpolation=bicubic --lr_scheduler=constant --learning_rate=1e-02"
log="--checkpointing_steps=1000 --save_steps=1000 --save_image_steps=400 --resume_from_checkpoint=latest"

export CUDA_VISIBLE_DEVICES=5
for i in {27..49}
do
	accelerate launch src/diffuser_inversion.py $path $args $lr $log --group_id=${i}
done