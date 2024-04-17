
path="--output arch/tiny-imagenet-syn-frog-snake"
data="--dataset-name=tiny-imagenet --group-size=500 --num-steps=100000"
args="--batch-size=128 --warmup-steps=1000 --num-steps=20000 --optimizer=sgd --weight-decay=5e-4 --real-bs=128 --syn-bs=128"
log="--num-evals=20 --seed=42 --wandb-name=DI-tiny-imagenet --log-wandb"
syn="--syn-data-dir=/home/docker_user/lan/diffusion_inversion/results/tiny-imagenet-frog-snake"

# Train on real data
export CUDA_VISIBLE_DEVICES=0
# python src/train_net.py $path $data $args $log --model=resnet18 --lr=1e-2
python src/train_net_and_selection.py $path $data $args $log $syn --model=resnet18 --lr=1e-2 --specific-class=1,5

#python src/train_net_and_selection.py $path $data $args $log $syn --model=resnet18 --lr=1e-2

# Train on synthetic data
#python src/train_net.py $path $data $syn $args $log --model=resnet18 --lr=5e-2
# for i in {0..50..5}
# do
#     python src/train_net.py $data $syn $args $log --model=resnet18 --lr=2e-1 --output=arch/${i}tstep-lr-2e-1 --syn-data-dir=results/inversion_data/cifar10/scaling/${i}t_step/res32_bicubic
# done

# path="--output arch/tiny-imagenet-syn-baseline"
# data="--dataset-name=tiny-imagenet --group-size=500 --num-steps=100000"
# args="--batch-size=128 --warmup-steps=1000 --num-data=5000 --num-steps=20000 --optimizer=sgd --weight-decay=5e-4 --real-bs=128 --syn-bs=128"
# log="--num-evals=20 --seed=42 --wandb-name=DI-tiny-imagenet --log-wandb"
# syn="--syn-data-dir=results/tiny-imagenet-syn-baseline"

# # Train on real data
# export CUDA_VISIBLE_DEVICES=0
# # python src/train_net.py $path $data $args $log --model=resnet18 --lr=1e-2
# python src/tiny_imagenet_train.py $path $data $args $log --model=resnet18 --lr=1e-2

# Train on synthetic data
#python src/train_net.py $path $data $syn $args $log --model=resnet18 --lr=5e-2
# for i in {0..50..5}
# do
#     python src/train_net.py $data $syn $args $log --model=resnet18 --lr=2e-1 --output=arch/${i}tstep-lr-2e-1 --syn-data-dir=results/inversion_data/cifar10/scaling/${i}t_step/res32_bicubic
# done