path="--dataset_name=cifar10 --model_root_dir=results/logs/cifar10/res128_bicubic/emb100_token5_lr0.03_constant --dm_name=CompVis/stable-diffusion-v1-4"
train_config="--emb_ch=768 --num_tokens=5 --num_classes=10 --num_emb=100 --sampling_resolution=128 --save_resolution=32  "
sampling_config="--num_inference_steps=50 --batch_size=100 --interpolation_strength=0.1 --num_samples=5 --emb_noise=0.1 --train_steps=3000 --seed=42"


export CUDA_VISIBLE_DEVICES=5
for i in {0..50..5}
do
    accelerate launch src/sample_dataset.py $path $train_config $sampling_config --t_step=${i} --outdir=results/inversion_data/cifar10/scaling/${i}t_step
done
