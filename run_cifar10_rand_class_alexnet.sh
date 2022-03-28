# CIFAR-100 10 class Alexnet runs

# train dcgan for cifar-100 10 classes
CUDA_VISIBLE_DEVICES=0 python ./code/train_generator/dcgan.py --dataroot ./data/ --dataset cifar100 --imageSize 32 --cuda --outf ./cifar100_run_models/alexnet/cifar_10_rand/dcgan/ --manualSeed 108 --niter 200 --batchSize 64 --network alexnet  --proxy_ds_name 10_class_rand --teacher_path ./teacher_models/cifar10_alexnet_teacher_79.pth --num_classes 10

mkdir val_data

CUDA_VISIBLE_DEVICES=0 python ./code/train_student/generate_val_data.py  --dcgan_netG ./cifar100_run_models/alexnet/cifar_10_rand/dcgan/netG_epoch_199.pth --dcgan_out ./val_data/dcgan_val_data_cifar10_rand_alexnet.pkl --teacher_path ./teacher_models/cifar10_alexnet_teacher_79.pth  --network alexnet


# Hard label runs

CUDA_VISIBLE_DEVICES=0 python ./code/train_student/train_student.py --from_scratch --dcgan_netG ./cifar100_run_models/alexnet/cifar_10_rand/dcgan/netG_epoch_199.pth --count 1 --network alexnet --dcgan_data --dcgan_data_ratio 0.8  --proxy_data --proxy_data_ratio 1 --pad_crop --name proxy_dcgan_45k_10_class_rand --proxy_ds_name 10_class_rand --val_data_dcgan ./val_data/dcgan_val_data_cifar10_rand_alexnet.pkl --teacher_path ./teacher_models/cifar10_alexnet_teacher_79.pth --true_dataset cifar10 --num_classes 10 --proxy_dataset cifar100

# ./train_student/checkpoint/alexnet/cifar10_cifar100/1/last_epoch_best_student_model_proxy_dcgan_45k_10_class_rand.pth


# Run div gan

CUDA_VISIBLE_DEVICES=0 python ./code/train_generator/train_gen.py --dataroot ./data/  --dataset cifar100 --imageSize 32 --cuda --outf ./cifar100_run_models/alexnet/out_step2_0_10/10_class_rand/2/ --manualSeed 108 --niter 100 --batchSize 64 --netG ./cifar100_run_models/alexnet/cifar_10_rand/dcgan/netG_epoch_199.pth --student_path ./train_student/checkpoint/alexnet/cifar10_cifar100/1/last_epoch_best_student_model_proxy_dcgan_45k_10_class_rand.pth  --network alexnet --c_l 0 --d_l 10 --proxy_ds_name 10_class_rand --true_dataset cifar10 --num_classes 10


# Div GAN + Random Crop (student trained from scratch on 0.5*Proxy + 0.5*DivGAN )
CUDA_VISIBLE_DEVICES=0 python code/train_student/train_student.py --div_gan_netG  ./cifar100_run_models/alexnet/out_step2_0_10/10_class_rand/2/netG_epoch_99.pth --dcgan_netG ./cifar100_run_models/alexnet/cifar_10_rand/dcgan/netG_epoch_199.pth  --count 2 --network alexnet --pad_crop  --div_gan_data --div_gan_data_ratio 0.8 --proxy_data --proxy_data_ratio 1 --from_scratch --name from_scratch_div_gan_05_10_class_rand  --proxy_ds_name 10_class_rand --val_data_dcgan ./val_data/dcgan_val_data_cifar10_rand_alexnet.pkl --teacher_path ./teacher_models/cifar10_alexnet_teacher_79.pth --proxy_dataset cifar100 --true_dataset cifar10 --num_classes 10


CUDA_VISIBLE_DEVICES=0 python ./code/train_student/generate_val_data.py  --div_gan_netG ./cifar100_run_models/alexnet/out_step2_0_10/10_class_rand/2/netG_epoch_99.pth  --network alexnet --div_gan_out ./val_data/div_gan_val_data_cifar10_rand_alexnet.pkl --teacher_path ./teacher_models/cifar10_alexnet_teacher_79.pth


# ./train_student/checkpoint/alexnet/cifar10_cifar100_div_gan/2/last_epoch_best_student_model_from_scratch_div_gan_05_10_class_rand.pth
# Alternate training

CUDA_VISIBLE_DEVICES=0 python ./code/train_generator/train_generator_clone.py --dataroot ./data/  --dataset cifar100 --imageSize 32 --cuda --outf ./cifar100_run_models/alexnet/out_step2/cifar_10_rand/0_500/ --manualSeed 108 --niter 800 --batchSize 64 --netG  ./cifar100_run_models/alexnet/out_step2_0_10/10_class_rand/2/netG_epoch_99.pth  --student_path ./train_student/checkpoint/alexnet/cifar10_cifar100_div_gan/2/last_epoch_best_student_model_from_scratch_div_gan_05_10_class_rand.pth  --network alexnet --teacher_path ./teacher_models/cifar10_alexnet_teacher_79.pth --warmup --name warmup_altr_0_500_1_p10  --auto-augment --c_l 0 --d_l 500  --proxy_ds_name 10_class_rand --val_data_dcgan ./val_data/dcgan_val_data_cifar10_rand_alexnet.pkl --val_data_degan ./val_data/div_gan_val_data_cifar10_rand_alexnet.pkl --true_dataset cifar10 --num_classes 10
