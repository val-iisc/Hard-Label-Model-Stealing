
# train dcgan for cifar-100 40 classes

CUDA_VISIBLE_DEVICES=0 python ./code/train_generator/dcgan.py --dataroot ./data/ --dataset cifar100 --imageSize 32 --cuda --outf ./cifar100_run_models/alexnet/cifar_40/dcgan/ --manualSeed 108 --niter 200 --batchSize 64 --network resnet  --proxy_ds_name 40_class --teacher_path ./teacher_models/cifar10_resnet18_teacher_93.pth --num_classes 10

CUDA_VISIBLE_DEVICES=0 python ./code/train_student/generate_val_data.py  --dcgan_netG ./cifar100_run_models/alexnet/cifar_40/dcgan/netG_epoch_199.pth --dcgan_out ./val_data/dcgan_val_data_cifar_40_resnet.pkl --teacher_path ./teacher_models/cifar10_resnet18_teacher_93.pth  --network resnet


# Hard label runs

CUDA_VISIBLE_DEVICES=0 python ./code/train_student/train_student.py --from_scratch --dcgan_netG ./cifar100_run_models/alexnet/cifar_40/dcgan/netG_epoch_199.pth --count 1 --network resnet --dcgan_data --dcgan_data_ratio 0.5  --proxy_data --proxy_data_ratio 1 --pad_crop --name proxy_dcgan_45k_40_class_rand --proxy_ds_name 40_class --val_data_dcgan ./val_data/dcgan_val_data_cifar_40_resnet.pkl --teacher_path ./teacher_models/cifar10_resnet18_teacher_93.pth --true_dataset cifar10 --num_classes 10 --proxy_dataset cifar100


# ./train_student/checkpoint/resnet/cifar10_cifar100/1/last_epoch_best_student_model_proxy_dcgan_45k_40_class_rand.pth


# Run div gan

CUDA_VISIBLE_DEVICES=0 python ./code/train_generator/train_gen.py --dataroot ./data/  --dataset cifar100 --imageSize 32 --cuda --outf ./cifar100_run_models/resnet/out_step2_0_10/40_class/2/ --manualSeed 108 --niter 100 --batchSize 64 --netG ./cifar100_run_models/alexnet/cifar_40/dcgan/netG_epoch_199.pth --student_path ./train_student/checkpoint/resnet/cifar10_cifar100/1/last_epoch_best_student_model_proxy_dcgan_45k_40_class_rand.pth  --network resnet --c_l 0 --d_l 10 --proxy_ds_name 40_class --true_dataset cifar10 --num_classes 10


# DivGAN + Random Crop (student trained from scratch on 0.5*Proxy + 0.5*DivGAN )
CUDA_VISIBLE_DEVICES=0 python code/train_student/train_student.py --div_gan_netG  ./cifar100_run_models/resnet/out_step2_0_10/40_class/2/netG_epoch_99.pth --dcgan_netG ./cifar100_run_models/alexnet/cifar_40/dcgan/netG_epoch_199.pth --count 2 --network resnet --pad_crop  --div_gan_data --div_gan_data_ratio 0.5 --proxy_data --proxy_data_ratio 1 --from_scratch --name from_scratch_div_gan_05_40_class  --proxy_ds_name 40_class --val_data_dcgan ./val_data/dcgan_val_data_cifar_40_resnet.pkl --teacher_path ./teacher_models/cifar10_resnet18_teacher_93.pth --proxy_dataset cifar100 --true_dataset cifar10 --num_classes 10


CUDA_VISIBLE_DEVICES=0 python ./code/train_student/generate_val_data.py  --div_gan_netG ./cifar100_run_models/resnet/out_step2_0_10/40_class/2/netG_epoch_99.pth  --network resnet --div_gan_out ./val_data/degan_val_data_cifar_40_resnet.pkl --teacher_path ./teacher_models/cifar10_resnet18_teacher_93.pth


# ./train_student/checkpoint/resnet/cifar10_cifar100_div_gan/2/last_epoch_best_student_model_from_scratch_div_gan_05_40_class.pth


# Alternate training

CUDA_VISIBLE_DEVICES=0 python ./code/train_generator/train_generator_clone.py --dataroot ./data/  --dataset cifar100 --imageSize 32 --cuda --outf ./cifar100_run_models/resnet/out_step2/cifar_40/0_500/ --manualSeed 108 --niter 400 --batchSize 64 --netG  ./cifar100_run_models/resnet/out_step2_0_10/40_class/2/netG_epoch_99.pth  --student_path ./train_student/checkpoint/resnet/cifar10_cifar100_div_gan/2/last_epoch_best_student_model_from_scratch_div_gan_05_40_class.pth  --network resnet --teacher_path ./teacher_models/cifar10_resnet18_teacher_93.pth --warmup --name warmup_altr_0_500_1_p10  --auto-augment --c_l 0 --d_l 500  --proxy_ds_name 40_class --val_data_dcgan ./val_data/dcgan_val_data_cifar_40_resnet.pkl --val_data_degan ./val_data/degan_val_data_cifar_40_resnet.pkl --true_dataset cifar10 --num_classes 10